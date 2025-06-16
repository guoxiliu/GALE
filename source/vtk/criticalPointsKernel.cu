#include <Relations.cuh>
#include <array>
#include <cstddef>
#include <iostream>
#include <set>
#include <stdlib.h>
#include <vector>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLGenericDataObjectReader.h>
#include <vector>
#include <Gale.h>
#include <MemoryUsage.h>
#include <TestTopoRelations.h>


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

using namespace ttk;


__global__ void critPointsA(const int * __restrict__ VV,                // FLAT array of <#points, #adjacent to point>
                            const int * __restrict__ VV_index, // Offset array into VV
                            int * __restrict__ valences,                // FLAT array same size/indexing as VV, used to reference relationship between point pairs based on their Scalar Values
                            const int points,                    // Number of points (== #blocks)
                            const int vvColumnNum,              // Padding factor for my algorithm (#adjacent to point)
                            const double * __restrict__ scalar_values, // Actual scalar values, vector of size #points
                            unsigned int * __restrict__ classes       // Output as vector of size #points
                            ) {
    extern __shared__ unsigned long long block_shared[]; // Used for GPU Union-Find!

    unsigned long long *component_edits = &block_shared[0], // Handle that can be mathematically offset (compiler can probably optimize out)
                       *upper_edits = &block_shared[0],  // Number of edits to the blocks' upper components in a cycle
                       *lower_edits = &block_shared[1],  // Number of edits to the blocks' lower components in a cycle
                       *neighborhood = &block_shared[2]; // Tracks current components within the thread block

    // Parallelized VV on second-dimension (number of adjacenies to each vertex)
    const int tid = (blockDim.x * blockIdx.x) + threadIdx.x,
                    // VV[tid] shows the SECOND point, the block ID is the first point
                    my_1d = blockIdx.x,
                    my_2d = VV[tid];
    // No work for this point's valence (final thread block may be too big)
    if (VV_index[my_1d] <= 0 || my_2d < 0) return;

    /*
        1) Read the scalar value used for point classification and classify
           yourself relative to your primary-dimension scalar value as upper (-1)
           or lower (1) neighbor. Store the result in shmem for Union-Find
           visibility across the block.
    */
    // Classify yourself as an upper or lower valence neighbor to your 1d point
    // Upper = -1, Lower = 1
    // This little ternary gadget makes my algorithm match paraview's tiebreakers on flat portions of the mesh
    const int my_class = 1 - ((scalar_values[my_2d] == scalar_values[my_1d] ? \
                                    my_2d < my_1d : \
                                    scalar_values[my_2d] < scalar_values[my_1d]) \
                                    << 1);
    // Global mesh visibility of what component you belong to
    valences[tid] = my_class;

    // Iteration bounds for Union-Find
    const int min_my_1d = (my_1d * vvColumnNum),
                    max_my_1d = min_my_1d + VV_index[my_1d];

    // Initialize neighborhood with YOURSELF, barrier for block visibility
    neighborhood[threadIdx.x] = my_2d;
    __syncthreads();
    /*
        2) For all other threads sharing your neighborhood classification, scan
           their connectivity in VV and update your "lowest classifying point"
           to be the lowest IDX (including yourself). Repeat until convergence
           where no edits are made, then if you have your own IDX at your
           location in memory, you log +1 component of your type.
    */
    bool upper_converge = false, lower_converge = false;
    // Repeat until both component directions converge
    while (!(upper_converge && lower_converge)) {
        // Sanity: Guarantee zero-init
        if (threadIdx.x == 0) {
            *upper_edits = 0;
            *lower_edits = 0;
        }
        __syncthreads();


        // Union-Find iteration
        for(int i = min_my_1d; i < max_my_1d; i++) {
            // Found same valence
            const int candidate_component_2d = VV[i];
            if (i != tid && valences[i] == my_class) {
                // Find yourself in their adjacency to become a shared component and release burden
                const int start_2d = candidate_component_2d*vvColumnNum,
                                stop_2d  = start_2d + VV_index[candidate_component_2d];
                for(int j = start_2d; j < stop_2d; j++) {
                    /* Do you see your 2d in their 2d? If so, update your
                       neighborhood with the minimum of your two points
                    */
                    const int theirNeighborhood = neighborhood[i-min_my_1d];
                    if (VV[j] == my_2d && neighborhood[threadIdx.x] > theirNeighborhood) {
                        neighborhood[threadIdx.x] = theirNeighborhood;
                        // +1 if lower, else +0
                        atomicAdd(component_edits+((my_class+1)/2), 1);
                    }
                }
            }
        }
        __syncthreads();
        upper_converge = *upper_edits == 0;
        lower_converge = *lower_edits == 0;
    }
    // End Union-Find

    if (neighborhood[threadIdx.x] == my_2d) { // You are the root of a component!
        // Memory layout is [#upper, #lower, <CLASSIFICATION>], need to aggregate
        // the component counts at block-visible memory
        const int memoffset = ((my_1d*3)+((my_class+1)/2));
        const unsigned int old = atomicAdd(classes+memoffset,1);
    }
    __syncthreads();
    /*
        3) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
    */
    // Limit classification to lowest-ranked thread for single write
    if (threadIdx.x == 0) {
        const int my_classes = my_1d*3;
        const unsigned int upper = classes[my_classes],
                           lower = classes[my_classes+1];
        if (upper == 1 && lower == 1) classes[my_classes+2] = REGULAR_CLASS;
        else if (upper == 0 && lower >= 1) classes[my_classes+2] = MAXIMUM_CLASS;
        else if (upper >= 1 && lower == 0) classes[my_classes+2] = MINIMUM_CLASS;
        else classes[my_classes+2] = SADDLE_CLASS;
    }
}

/**
 * Main entry function.
 */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    int c;
    int blockNumber = 8;
    int blockSize = 512;
    int threadNumber = 1;
    int numThreadsPerCluster = 32;
    std::string fileName = "";
    std::string inputArrayName = "";

    while ((c = getopt(argc, argv, "a:b:i:p:s:t:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
                break;
            case 'a':
                inputArrayName = optarg;
                break;
            case 'b':
                blockNumber = atoi(optarg);
                break;
            case 'p':
                numThreadsPerCluster = atoi(optarg);
                break;
            case 's':
                blockSize = atoi(optarg);
                break;
            case 't':
                threadNumber = atoi(optarg);
                break;
        }
    }

    std::cout << "Parsing vtu file: " << fileName << std::endl;
    
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
    double *vertexAttributes = static_cast<double*>(inputArray->GetVoidPointer(0));

    ttk::Gale gale(threadNumber, numThreadsPerCluster);
    if (blockNumber > 0 && blockSize > 0)
        gale.initialize(inputDataSet, blockNumber, blockSize, false);

    // the following code of calling preconditionVertexNeighbor() does not work somehow
    // gale.preconditionVertexNeighbors();
        
    // using testing plugin to precompute the VV relation does not work either
    ttk::TestTopoRelations testPlugin;
    testPlugin.preconditionTriangulation(&gale);
    testPlugin.setThreadNumber(threadNumber);

    /* BEYOND THIS POINT IS FOR CLARITY ON THE MEMORY LAYOUT AND KERNEL INVOCATION */
    int nCells = gale.getNumberOfCells();
    int nPoints = gale.getNumberOfVertices();
    size_t classes_size = sizeof(unsigned int) * nPoints * 3, // #Upper, #Lower, Classification
        valences_size = sizeof(vtkIdType) * nPoints * vvColumnNum,
        scalars_size = sizeof(double) * nPoints;

    // Allocations
    unsigned int *host_CPC = new unsigned int[classes_size];
    unsigned int *device_CPC = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_CPC, classes_size));
    // valences = adjacency upper/lower classification PRIOR to point classification
    int *device_valences = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_valences, valences_size));
    double *device_scalar_values = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_scalar_values, scalars_size));
    // Scalar values from VTK
    CUDA_WARN(cudaMemcpyAsync(device_scalar_values, vertexAttributes, scalars_size, cudaMemcpyHostToDevice));

    ttk::Timer t;

    // Critical Points
    // Set kernel launch parameters here
    // Parallelized VV on second-dimension (number of adjacenies to each vertex)
    int n_to_compute = nPoints * vvColumnNum;
    dim3 thread_block_size = vvColumnNum,
        grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    // Need one entry per block (VV width) + 2 entries for tracking edits to # upper and #lower components
    const int shared_mem_size = sizeof(int) * (2+vvColumnNum);

    critPointsA<<<grid_size, thread_block_size, shared_mem_size>>>(
            gale.dVVDataArray_,
            gale.dVVIndexArray_,
            device_valences,
            nPoints,
            vvColumnNum,
            device_scalar_values,
            device_CPC);
    CUDA_WARN(cudaDeviceSynchronize()); // Make algorithm timing accurate
    // // Results are retrieved from GPU
    CUDA_WARN(cudaMemcpy(host_CPC, device_CPC, classes_size, cudaMemcpyDeviceToHost));

    std::cout << "[ScalarFieldCriticalPoints] Time usage for computation: " << t.getElapsedTime() << " s." << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Memory usage: " << m.getValue_in_MB(false) << " MB." << std::endl;


    int numMinimum = 0, numMaximum = 0, numSaddles = 0;
    for (int i = 0; i < nPoints; i++) {
        if (host_CPC[i*3+2] == MINIMUM_CLASS) {
            numMinimum++;
        } else if (host_CPC[i*3+2] == MAXIMUM_CLASS) {
            numMaximum++;
        } else if (host_CPC[i*3+2] == SADDLE_CLASS) {
            numSaddles++;
        }
    }
    std::cout << "[ScalarFieldCriticalPoints] Number of minimum points: " << numMinimum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of maximum points: " << numMaximum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of saddle points: " << numSaddles << std::endl;

    return 0;
}
