#include <Gale.h>
#include <Relations.cuh>
#include <TestTopoRelations.h>
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLGenericDataObjectReader.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <set>
#include <vector>
#include <vtkPointData.h>
//#include <MemoryUsage.h>


#include "ttkUtils.h"

#define CUDA_ASSERT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(err); \
    } \
} while (0)

#define CUDA_WARN(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Warning: %s\n", cudaGetErrorString(err)); \
    } \
} while (0)

// Class definitions as enums, basically
#define MAXIMUM_CLASS 1
#define MINIMUM_CLASS 2
#define REGULAR_CLASS 3
#define SADDLE_CLASS  4
int SFCP_ColumnNum = 80;
//#define FORCE_BLOCK_IDX 13

using namespace ttk;


__global__ void critPoints(const vtkIdType * __restrict__ VV, // formerly int
                           const unsigned int * __restrict__ VV_index, // formerly int
                           int * __restrict__ valences,
                           const int points,
                           const int max_VV_guess, // formerly SFCP_ColumnNum
                           const double * __restrict__ scalar_values,
                           const int * __restrict__ partition, // NEW
                           const vtkIdType * __restrict__ vvi, // NEW
                           const vtkIdType * __restrict__ ivvi, // NEW
                           unsigned int * __restrict__ classes) {
    extern __shared__ unsigned int block_shared[];

    unsigned int *component_edits = &block_shared[0],
                 *upper_edits = &block_shared[0],
                 *lower_edits = &block_shared[1];
    vtkIdType *neighborhood = (vtkIdType*)(block_shared+2); // NEW: Multiple datatype shared

    // Set execution mask to EXCLUDE no work / out-of-partition points
    bool execution_mask = true;
    /*
        1) Parallelize VV on second-dimension

           my_1d and my_2d have the GLOBAL vertex IDs
           indirect_my*    have the LOCAL vertex IDs used to index in all
                            arrays except ivvi
    */
    const int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    const vtkIdType indirect_my_1d = tid / max_VV_guess,
                    my_1d = vvi[indirect_my_1d],
                    my_2d = VV[tid];
    vtkIdType indirect_my_2d;
    if (my_2d < 0) {
        execution_mask = false;
    }
    else {
        indirect_my_2d = ivvi[my_2d];
    }

    if (VV_index[indirect_my_1d] <= 0 || partition[indirect_my_1d] == 0) {
        #if PRINT_ON
        /*
        printf("Block %d Thread %02d exits due to partitioning or OOB data\n",
               blockIdx.x, threadIdx.x);
        */
        #endif
        execution_mask = false;
    }
    // Prefix scan as anti-duplication
    for (vtkIdType i = indirect_my_1d * max_VV_guess; i < tid; i++) {
        if (VV[i] == my_2d) {
            #if PRINT_ON
            /*
            printf("Block %d Thread %02d exits from prefix scan\n",
                   blockIdx.x, threadIdx.x);
            */
            #endif
            execution_mask = false;
        }
    }

    #if PRINT_ON
    /*
    printf("Block %d Thread %02d remains to participate in computation (VV_index=%u)\n",
           blockIdx.x, threadIdx.x, VV_index[indirect_my_1d]);
    */
    #endif

    /*
        2) Read the scalar value used for point classification and classify
           your 2nd point relative to your 1st scalar value as upper (-1) or
           lower (1) neighbor

           For parity with Paraview, tied scalar values are tie-broken such
           that the lesser vertex ID is assigned as lower
    */

    /* These variables COULD be const, but non-executing threads cannot be
       allowed to potentially read bad memory addresses */
    int my_class;
    vtkIdType min_my_1d, max_my_1d;
    if (execution_mask) {
        my_class = 1 - (
                        (scalar_values[indirect_my_2d] == scalar_values[indirect_my_1d] ?
                            my_2d < my_1d :
                            scalar_values[indirect_my_2d] < scalar_values[indirect_my_1d])
                        << 1);
        // Allow other threads in the block to read your class for matching
        // Not SHMEM yet -- could be as there are no cross-block reads here
        valences[tid] = my_class;
        min_my_1d = (indirect_my_1d * max_VV_guess);
        max_my_1d = min_my_1d + VV_index[indirect_my_1d];
        // Initialize your neighborhood with your own 2d value
        // If you END the union-find without editing your neighborhood, you are
        // the root of a component of your class.
        neighborhood[threadIdx.x] = my_2d;
    }
    __syncthreads();
    #if PRINT_ON
    if (execution_mask) {
        printf("Block %d Thread %02d B init on my_1d %ld and my_2d %ld with "
               "valence %d based on %lf (block) vs %lf (thread)\n",
               blockIdx.x, threadIdx.x, my_1d, my_2d, my_class,
               scalar_values[indirect_my_1d], scalar_values[indirect_my_2d]);
    }
    #endif
    /*
        3) Union-Find: All threads with a common classification attempt to
            union such that the lowest-connected point with the same class
            is set as your neighborhood value (accelerating future decisions)

            Repeat until convergence. The `to_converge` variable should NEVER
            be a termination condition but guarantees kernel termination (even
            if all points share the same class, Union-Find CANNOT execute more
            iterations than this maximum).
    */
    bool upper_converge = false, lower_converge = false;
    int to_converge = 0;

    while (!(upper_converge && lower_converge) && to_converge < max_VV_guess) {
        // Reset convergence indicators for the iteration
        if (threadIdx.x == 0) {
            #if PRINT_ON
            /*
            printf("Block %d Iteration %d: upper_edits %u lower_edits %u "
                   "Neighborhood %ld %ld %ld %ld %ld %ld %ld %ld "
                                "%ld %ld %ld %ld %ld %ld %ld %ld "
                                "%ld %ld %ld %ld %ld %ld %ld %ld "
                                "%ld %ld %ld %ld %ld %ld %ld %ld\n",
                    blockIdx.x, to_converge, *upper_edits, *lower_edits,
                    neighborhood[0],neighborhood[1],neighborhood[2],neighborhood[3],
                    neighborhood[4],neighborhood[5],neighborhood[6],neighborhood[7],
                    neighborhood[8],neighborhood[9],neighborhood[10],neighborhood[11],
                    neighborhood[12],neighborhood[13],neighborhood[14],neighborhood[15],
                    neighborhood[16],neighborhood[17],neighborhood[18],neighborhood[19],
                    neighborhood[20],neighborhood[21],neighborhood[22],neighborhood[23],
                    neighborhood[24],neighborhood[25],neighborhood[26],neighborhood[27],
                    neighborhood[28],neighborhood[29],neighborhood[30],neighborhood[31]);
            */
            #endif
            *upper_edits = 0;
            *lower_edits = 0;
        }
        to_converge++;
        __syncthreads();

        if (execution_mask) {
            #if PRINT_ON
            printf("Block %d thread %02d clears START UF sync %d\n", blockIdx.x,
                    threadIdx.x, to_converge);
            #endif

            // Union-Find iteration
            for(vtkIdType i = min_my_1d; i < max_my_1d; i++) {
                // Find a matching valence
                const vtkIdType candidate_component_2d = VV[i],
                                indirect_candidate_component_2d = ivvi[candidate_component_2d];
                if (i != tid && valences[i] == my_class) {
                    const vtkIdType start_2d = indirect_candidate_component_2d*max_VV_guess,
                                    stop_2d  = start_2d + VV_index[indirect_candidate_component_2d];
                    #if PRINT_ON
                    /*
                    if (start_2d > blockDim.x * gridDim.x) {
                        printf("Block %d Thread %02d has bad iterator from %ld "
                                "to %ld based on direct %ld and indirect %ld "
                                "at iteration %ld\n",
                                blockIdx.x, threadIdx.x, start_2d, stop_2d,
                                candidate_component_2d,
                                indirect_candidate_component_2d, i);
                    }
                    */
                    #endif
                    /* Do you see your 2d in their 2d? If so, update your
                       neighborhood with the minimum of your two points
                    */
                    for(vtkIdType j = start_2d; j < stop_2d; j++) {
                        const vtkIdType theirNeighborhood = neighborhood[i-min_my_1d];
                        if (VV[j] == my_2d && neighborhood[threadIdx.x] > theirNeighborhood) {
                            #if PRINT_ON
                            /*
                            printf("Block %d Thread %02d neighborhood update "
                                    "iteration %0d moves %ld --> %ld\n",
                                    blockIdx.x, threadIdx.x, to_converge,
                                    neighborhood[threadIdx.x],
                                    theirNeighborhood);
                            */
                            #endif
                            neighborhood[threadIdx.x] = theirNeighborhood;
                            // Denote LACK of convergence this iteration
                            atomicAdd(component_edits+((my_class+1)/2), 1);
                        }
                    }
                }
            }
        }
        // End Union-Find iteration

        __syncthreads();
        upper_converge = (*upper_edits == 0);
        lower_converge = (*lower_edits == 0);
        #if PRINT_ON
        if (execution_mask) {
            printf("Block %d Thread %02d ends UF loop %02d with upper_edits %u "
                    "lower_edits %u (loop condition: %d = [%d[!%d && %d] && %d])\n",
                    blockIdx.x, threadIdx.x, to_converge, *upper_edits, *lower_edits,
                    !(upper_converge && lower_converge) && to_converge < max_VV_guess,
                    !(upper_converge && lower_converge),
                    upper_converge, lower_converge,
                    to_converge < max_VV_guess
                    );
        }
        #endif
    }
    // End all Union-Finds

    /*
       4) Each root of each component increments the counter for its class to
            contribute to the final classification of the per-block point
    */
    if (execution_mask && neighborhood[threadIdx.x] == my_2d) {
        const vtkIdType memoffset = ((indirect_my_1d*3)+((my_class+1)/2));
        const unsigned int old = atomicAdd(classes+memoffset,1);
        #if PRINT_ON
        printf("Block %d Thread %02d Writes %s component @ mem offset %ld "
                "(old: %u) after %02d iterations\n",
                blockIdx.x, threadIdx.x,
                (my_class == -1 ? "Upper" : "Lower"), memoffset, old, to_converge);
        #endif
    }
    __syncthreads();

    /*
        5) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
    */
    if (threadIdx.x == 0) {
        const vtkIdType my_classes = indirect_my_1d*3;
        const unsigned int upper = classes[my_classes],
                           lower = classes[my_classes+1];
        #if PRINT_ON
        printf("Block %d Thread %02d Verdict: my_1d (%ld) has %u upper and %u "
                "lower after %02d iterations\n",
                blockIdx.x, threadIdx.x, my_1d, upper, lower, to_converge);
        #endif

        // Set classification
        if (upper >= 1 && lower == 0) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "MINIMUM");
            #endif
            classes[my_classes+2] = MINIMUM_CLASS;
        }
        else if (upper == 0 && lower >= 1) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "MAXIMUM");
            #endif
            classes[my_classes+2] = MAXIMUM_CLASS;
        }
        else if (upper == 1 && lower == 1) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "REGULAR");
            #endif
            classes[my_classes+2] = REGULAR_CLASS;
        }
        else {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n",
                    blockIdx.x, threadIdx.x, "SADDLE");
            #endif
            classes[my_classes+2] = SADDLE_CLASS;
        }
    }
    // END OF KERNEL
    #if PRINT_ON
    printf("Block %d Thread %02d exits normally at end of computation with "
            "%02d iterations\n",
            blockIdx.x, threadIdx.x, to_converge);
    #endif
}

/**
 * Main entry function.
 */
int main(int argc, char *argv[])
{
    ttk::Timer compat_timer;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    int c,
        blockNumber = 8,
        blockSize = 512,
        threadNumber = 1,
        numThreadsPerCluster = 32;
    std::string fileName = "",
                inputArrayName = "",
                partitionArrayName = "";

    while ((c = getopt(argc, argv, "i:a:p:b:c:s:t:v:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
                break;
            case 'a':
                inputArrayName = optarg;
                break;
            case 'p':
                partitionArrayName = optarg;
                break;
            case 'b':
                blockNumber = atoi(optarg);
                break;
            case 'c':
                numThreadsPerCluster = atoi(optarg);
                break;
            case 's':
                blockSize = atoi(optarg);
                break;
            case 't':
                threadNumber = atoi(optarg);
                break;
            case 'v':
                SFCP_ColumnNum = atoi(optarg);
                break;
        }
    }
    std::cout << "CMDLINE ARGS:" << std::endl;
    std::cout << "Parsing vtu file (-i): " << fileName << std::endl;
    std::cout << "Look for data array (-a): " << inputArrayName << std::endl;
    std::cout << "Look for partition array (-p): " << partitionArrayName << std::endl;
    std::cout << "Block Number (-b): " << blockNumber << std::endl;
    std::cout << "numThreadsPerCluster (-c): " << numThreadsPerCluster << std::endl;
    std::cout << "Block Size (-s):" << blockSize << std::endl;
    std::cout << "CPU threads (-t):" << threadNumber << std::endl;
    std::cout << "SFCP Column Num (-v):" << SFCP_ColumnNum << std::endl;
    std::cout << "[TIMER] Argument parsing: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();
    // Create CUDA Context
    (void)cudaFree(0);
    std::cout << "[TIMER] Create CUDA Context: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    //read all the data from the file
    vtkSmartPointer<vtkXMLGenericDataObjectReader> reader =
        vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();

    MemoryUsage m;

    int scalarFieldIndex = 0;
    vtkDataObject *inputDataObject = reader->GetOutput();
    if(!inputDataObject) {
        std::cerr << "Unable to read input file `" << fileName << "' :(" << std::endl;
        return 1;
    }
    vtkDataSet *inputDataSet = vtkDataSet::SafeDownCast(inputDataObject);
    vtkDataArray *inputArray = GetInputArray(inputArrayName, inputDataSet, scalarFieldIndex);

    ttk::Gale gale(threadNumber, numThreadsPerCluster, SFCP_ColumnNum);
    gale.initialize(inputDataSet, blockNumber, blockSize, true);
    std::cout << "[TIMER] Initialize VTK: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    // TODO: Set up VV on just the one partition
    const int FORCE_PARTITION = 1;
    int * HVertexIntervals = gale.getHVertexIntervals(),
        * HCellIntervals   = gale.getHCellIntervals(),
        * HExternalCells   = gale.getHExternalCells();
    int n_in_partition = HVertexIntervals[FORCE_PARTITION] - HVertexIntervals[FORCE_PARTITION-1];
    // Other vertices included (as upper bound)
    int vertexCount = n_in_partition + 3 * ((HCellIntervals[FORCE_PARTITION] - HCellInteravls[FORCE_PARTITION-1])
                                           + (HExternalCells[FORCE_PARTITION] - HExternalCells[FORCE_PARTITION-1]));
    std::cout << "Found " << n_in_partition << " points in partition "
              << FORCE_PARTITION << " (at most " << vertexCount << " vertices)"
              << std::endl;

    // Locla variables for kernel
    int *dVertexMap = nullptr,
        *hVertexMap = new int[vertexCount],
        *dTotalVertexNum = nullptr,
        hTotalVertexNum = n_in_partition,
        *dSizeArray = nullptr,
        *hSizeArray = new int[vertexCount],
        *dDataArray = nullptr,
        *hDataArray = new int[vertexCount * SFCP_ColumnNum];

    size_t vertices_size = vertexCount * sizeof(int),
           vv_data_size  = vertexCount * SFCP_ColumnNum * sizeof(int);
    CHECK(cudaMalloc(&dVertexMap, vertices_size));
    CHECK(cudaMalloc(&dTotalVertexNum, sizeof(int)));
    CHECK(cudaMalloc(&dSizeArray, vertices_size));
    CHECK(cudaMalloc(&dDataArray, vv_data_size));
    // Copy & Set
    CHECK(cudaMemcpy(dTotalVertexNum, &hTotalVertexNum, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dVertexMap, -1, vertices_size));
    CHECK(cudaMemset(dSizeArray, 0, vertices_size));
    CHECK(cudaMemset(dDataArray, -1, vv_data_size));

    // Call to kernel
    /* PROBLEM WITH PREVIOUS KERNEL WAS: block size too big! Have to limit
       thread_block_size ( -b * -s ) / (-p) = #clusters to compute in one
                                              kernel */
    computeVertexNeighborsWithExternals<<<1, 32>>>(
            dDataArray,
            dSizeArray,
            dVertexMap,
            dTotalVertexNum,
            FORCE_PARTITION,
            gale.getNumberOfClusters(),
            gale.getDConnectivity(),
            gale.getDOffsets(),
            gale.getDVertexIntervals(),
            gale.getDCellIntervals(),
            gale.getDExternalCells(),
            gale.getDExternalCellOffsets(),
            3, /* cellDimension */
            SFCP_ColumnNum,
            );
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        std::cerr << "VV kernel launch failed (" << code << ") "
                  << cudaGetErrorString(code) << std::endl;
    }

    // No need to pull back to CPU, just synchronize with the device
    cudaDeviceSynchronize();

    std::cout << "[TIMER] GALE VV: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    /* BEYOND THIS POINT IS FOR CLARITY ON THE MEMORY LAYOUT AND KERNEL INVOCATION * /
    // TODO: Ensure these are partition-local counts
    int nCells = static_cast<int>(gale.getNumberOfCells()), // cellNumber_
        nPoints = static_cast<int>(gale.getNumberOfVertices()); // vertexNumber_

    size_t classes_size = sizeof(unsigned int) * nPoints * 3, // #Upper, #Lower, Classification
           valences_size = sizeof(vtkIdType) * nPoints * SFCP_ColumnNum,
           scalars_size = sizeof(double) * nPoints;

    // Set scalars here
    double * vertexAttributes = new double[nPoints];
    // TODO: May be able to yank this from gale internals easily
    for (vtkIdType i = 0; i < nPoints; i++) {
        vertexAttributes[i] = inputArray->GetTuple1(i);
    }

    // Allocations
    unsigned int * host_CPC = new unsigned int[classes_size],
                 * device_CPC = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_CPC, classes_size));
    // valences = adjacency upper/lower classification PRIOR to point classification
    int * device_valences = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_valences, valences_size));
    double * device_scalar_values = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_scalar_values, scalars_size));
    // Scalar values from VTK
    CUDA_WARN(cudaMemcpyAsync(device_scalar_values, vertexAttributes, scalars_size, cudaMemcpyHostToDevice));

    std::cout << "[TIMER] GALE SFCP preparation: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    // Critical Points
    // Set kernel launch parameters here
    // Parallelized VV on second-dimension (number of adjacenies to each vertex)
    #ifdef FORCE_BLOCK_IDX
    dim3 grid_size = 1,
    #else
    int n_to_compute = nPoints * SFCP_ColumnNum;
    dim3 grid_size = (n_to_compute + SFCP_ColumnNum - 1) / SFCP_ColumnNum,
    #endif
         thread_block_size = SFCP_ColumnNum;
    // Need one entry per block (VV width) + 2 entries for tracking edits to # upper and #lower components
    const int shared_mem_size = (sizeof(unsigned int) * 2) + (sizeof(vtkIdType) * SFCP_ColumnNum);

    // TODO: Convert to partition-based kernel!
    critPoints<<<grid_size, thread_block_size, shared_mem_size>>>(
            dataArray,
            sizeArray,
            device_valences,
            nPoints,
            SFCP_ColumnNum,
            device_scalar_values,
            device_CPC);
    CUDA_WARN(cudaDeviceSynchronize()); // Make algorithm timing accurate
    // // Results are retrieved from GPU
    std::cout << "[ScalarFieldCriticalPoints] Time usage for computation: " << compat_timer.getElapsedTime() << " s." << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Memory usage: " << m.getValue_in_MB(false) << " MB." << std::endl;
    compat_timer.reStart();
    CUDA_WARN(cudaMemcpy(host_CPC, device_CPC, classes_size, cudaMemcpyDeviceToHost));
    std::cout << "[TIMER] Retrieve results from GPU: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    // TODO: Make output comparable to standalone
    int numMinimum = 0, numMaximum = 0, numSaddles = 0;
    for (int i = 0; i < nPoints; i++) {
        if (host_CPC[(i*3)+2] == MINIMUM_CLASS) {
            numMinimum++;
        } else if (host_CPC[(i*3)+2] == MAXIMUM_CLASS) {
            numMaximum++;
        } else if (host_CPC[(i*3)+2] == SADDLE_CLASS) {
            numSaddles++;
        }
    }
    std::cout << "[ScalarFieldCriticalPoints] Number of minimum points: " << numMinimum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of maximum points: " << numMaximum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of saddle points: " << numSaddles << std::endl;
    */
    return 0;
}
