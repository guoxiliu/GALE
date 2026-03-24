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
#include "class_enum.h"
char *class_strmap[5] = {"VOID", "MAXIMUM", "MINIMUM", "REGULAR", "SADDLE"};

// Kernel separated to reduce burden of recompiling
#include "gpu_kernel.h"
int SFCP_ColumnNum = 80;

using namespace ttk;

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
    int FORCE_PARTITION = 1;
    std::string fileName = "",
                inputArrayName = "",
                partitionArrayName = "";

    while ((c = getopt(argc, argv, "i:a:p:b:c:s:t:v:f:")) != -1) {
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
            case 'f':
                FORCE_PARTITION = atoi(optarg);
                break;
        }
    }
    std::cout << "CMDLINE ARGS:" << std::endl;
    std::cout << "Parsing vtu file (-i): " << fileName << std::endl;
    std::cout << "Look for data array (-a): " << inputArrayName << std::endl;
    std::cout << "Look for partition array (-p): " << partitionArrayName << std::endl;
    std::cout << "Block Number (-b): " << blockNumber << std::endl;
    std::cout << "numThreadsPerCluster (-c): " << numThreadsPerCluster << std::endl;
    std::cout << "Block Size (-s): " << blockSize << std::endl;
    std::cout << "CPU threads (-t): " << threadNumber << std::endl;
    std::cout << "SFCP Column Num (-v): " << SFCP_ColumnNum << std::endl;
    std::cout << "FORCED PARTITION (-f): " << FORCE_PARTITION << std::endl;
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
    /* Partitions set in gale.dataArray, copied into hIndexArray_, exists on device as dIndexArray_ (1-based indexing?)
       gale.clusterNumber_ has # of partitions
       VVI = mapping of contiguous points [0,nPoints) -> mesh-global VID
            |VVI| = #points in partition
            ReorderVertices() sets this, but it doesn't persist in Gale object?
                dVertexIntervals_ = size of partitions
                clusterVertices[partitionID][nth_point] = mesh-global-VID
                foreach partition in clusterVertices
                    vertexMap[global-id] = new partition-sequential-IDs
                hVertexIntervals_[partition+1] = end of partition in new partition-sequenced-ID order
       IVVI = dense array mapping mesh-global VID -> nth point for partition
            |IVVI| = #points in mesh
     */
    // TODO: Ensure these are partition-local counts
    int nCells = static_cast<int>(gale.getNumberOfCells()), // cellNumber_
        nPoints = static_cast<int>(gale.getNumberOfVertices()); // vertexNumber_
    std::cout << "Gale reports " << nCells << " cells and " << nPoints << " points"
              << std::endl;

    std::cout << "[TIMER] Initialize VTK: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    // TODO: Set up VV on just the one partition
    std::cout << "Set partition to " << FORCE_PARTITION << std::endl;
    int * HVertexIntervals = gale.getHVertexIntervals(),
        * HCellIntervals   = gale.getHCellIntervals(),
        * HExternalCells   = gale.getHExternalCells();
    // Gale-like way to determine partition sizes and contents
    // total number of partitions in Gale is clusterNumber_
    int n_in_partition = HVertexIntervals[FORCE_PARTITION] - HVertexIntervals[FORCE_PARTITION-1];
    // Other vertices included (as upper bound)
    int vertexCount = n_in_partition + 3 * ((HCellIntervals[FORCE_PARTITION] - HCellIntervals[FORCE_PARTITION-1])
                                           + (HExternalCells[FORCE_PARTITION] - HExternalCells[FORCE_PARTITION-1]));
    std::cout << "Found " << n_in_partition << " points in partition "
              << FORCE_PARTITION << " (at most " << vertexCount << " vertices "
              << "(" << HCellIntervals[FORCE_PARTITION]-HCellIntervals[FORCE_PARTITION-1] << " internal and "
              << HExternalCells[FORCE_PARTITION]-HExternalCells[FORCE_PARTITION-1] << " external))"
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
            dDataArray,                     /* int* vertexNeighborMatrix = {-1}, to be filled with VV */
            dSizeArray,                     /* int* vertexNeighborSizes = {0}, to be filled with VV lengths */
            dVertexMap,                     /* int* vertexMap = {-1}, map local->global vertex ID (local implicit by index) */
            dTotalVertexNum,                /* int* totalVertexNum = n_in_partition, but as a pointer instead of regular int? */
            FORCE_PARTITION,                /* int  baseCluster = target cluster/partition to work on */
            gale.getNumberOfClusters(),     /* int  clusterNumber = total number of clusters/partitions available */
            gale.getDConnectivity(),        /* int* connectivity = TV from VTK */
            gale.getDOffsets(),             /* int* offsets = TV from VTK */
            gale.getDVertexIntervals(),     /* int* vertexIntervals */
            gale.getDCellIntervals(),       /* int* cellIntervals */
            gale.getDExternalCells(),       /* int* externalCells */
            gale.getDExternalCellOffsets(), /* int* externalCellOffsets */
            3,                              /* int  dimension = points are in 3D space */
            SFCP_ColumnNum                  /* int  sfcp_ColumnNum = max number of VV neighbors */
            );
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        std::cerr << "VV kernel launch failed (" << code << ") "
                  << cudaGetErrorString(code) << std::endl;
    }

    // No need to pull back to CPU, just synchronize with the device
    cudaDeviceSynchronize();

    // DEBUG: Yank back to Host and check answer
    if (FORCE_PARTITION == 2) {
        CHECK(cudaMemcpy(hDataArray, dDataArray, vv_data_size, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hSizeArray, dSizeArray, vertices_size, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hVertexMap, dVertexMap, vertices_size, cudaMemcpyDeviceToHost));

        // Find double-listed vertexMaps?
        std::vector<int> known_vertices(nPoints, 0);
        int max_known_count = 0;
        for (int i = 0; i < vertexCount; i++) {
            if (hVertexMap[i] != -1) {
                known_vertices[hVertexMap[i]]++;
                if (known_vertices[hVertexMap[i]] > max_known_count) {
                    max_known_count = known_vertices[hVertexMap[i]];
                    std::cout << "New max known count set at vertex "
                              << hVertexMap[i] << ": " << max_known_count
                              << std::endl;
                }
            }
        }
        std::map<int,int> unVertexMap;
        for (int i = 0; i < vertexCount; i++) {
            if (hVertexMap[i] == -1) continue;
            unVertexMap[hVertexMap[i]] = i;
        }

        const int N_EXPECT = 24,
                  TARGET = 7;
        // Find target in vertexMap
        int INDIRECT_TARGET = -1;
        for (int i = 0; i < vertexCount; i++) {
            if (hVertexMap[i] == TARGET) {
                INDIRECT_TARGET = i;
            }
        }
        // Points in the dataArray have to be remapped in order to get the proper global values I know back
        int expect[N_EXPECT] = {0, 26, 48, 486, 1073, 1074, 1088, 1126, 1187, 1199, 1204, 1283, 1296, 1428, 5708, 5710, 6781, 6790, 6880, 6921, 6928, 6930, 6947, 7203},
            found[N_EXPECT]  = {0};
        for (int i = 0 ; i < hSizeArray[INDIRECT_TARGET]; i++) {
            std::cout << "All neighbors of " << TARGET << " ith entry: " << hDataArray[(INDIRECT_TARGET * SFCP_ColumnNum) + i] << std::endl;
        }
        for (int i = 0; i < hSizeArray[INDIRECT_TARGET]; i++) {
            // Note neighbor AS GLOBAL VALUE
            int neighbor = hDataArray[(INDIRECT_TARGET * SFCP_ColumnNum) + i];
            for (int j = 0; j < N_EXPECT; j++) {
                if (neighbor == unVertexMap[expect[j]]) {
                    found[j] = 1;
                }
            }
        }
        int found_count = 0;
        for (int i = 0; i < N_EXPECT; i++) {
            if (!found[i]) {
                std::cout << "! Failed to find " << TARGET << " expected neighbor: " << expect[i] << std::endl;
            }
            else {
                std::cout << "+ Did find " << TARGET << " expected neighbor: " << expect[i] << std::endl;
                found_count++;
            }
        }
        if (found_count == N_EXPECT) {
            std::cout << "Found all expected neighbors for " << TARGET << std::endl;
            // Check for roots of components
            const int N_ROOTS = 2;
            int roots[N_ROOTS] = {0,26};
            std::cout << "Considering global root IDs: ";
            for (int i = 0; i < N_ROOTS; i++) {
                if (i > 0) std::cout << ", ";
                std::cout << roots[i];
            }
            std::cout << std::endl;
            for (int i = 0; i < N_EXPECT; i++) {
                bool non_root = true;
                for (int &root : roots) {
                    if (expect[i] == root) {
                        non_root = false;
                        break;
                    }
                }
                if (!non_root) continue;
                // See if this connects TO a root
                bool connect_to_root = false;
                for (int &root : roots) {
                    if (connect_to_root) break;
                    int indirect_root = unVertexMap[root];
                    for (int j = 0; j < hSizeArray[indirect_root]; j++) {
                        if (hDataArray[(indirect_root * SFCP_ColumnNum) + j] == unVertexMap[expect[i]]) {
                            std::cout << "Expected neighbor " << expect[i] << " has a connection to root " << root << std::endl;
                            connect_to_root = true;
                            break;
                        }
                    }
                }
                if (!connect_to_root) {
                    std::cout << "Expected neighbor " << expect[i] << " has NO CONNECTIONS to any roots" << std::endl;
                }
            }
        }
        else {
            std::cout << "Found " << found_count << " / " << N_EXPECT << " neighbors for " << TARGET << std::endl;
        }
    }

    std::cout << "[TIMER] GALE VV: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    /* BEYOND THIS POINT IS FOR CLARITY ON THE MEMORY LAYOUT AND KERNEL INVOCATION */
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

    /* VVI, IVVI, partitions */
    vtkIdType * host_vvi,
              * devi_vvi,
              * host_ivvi,
              * devi_ivvi;
    int * host_partition,
        * devi_partition;
    size_t vvi_size = sizeof(vtkIdType) * vertexCount,
           ivvi_size = sizeof(vtkIdType) * nPoints,
           partition_size = sizeof(int) * vertexCount;
    host_vvi = new vtkIdType[vertexCount];
    CUDA_ASSERT(cudaMalloc((void**)&devi_vvi, vvi_size));
    host_ivvi = new vtkIdType[nPoints];
    CUDA_ASSERT(cudaMalloc((void**)&devi_ivvi, ivvi_size));
    host_partition = new int[vertexCount]();
    CUDA_ASSERT(cudaMalloc((void**)&devi_partition, partition_size));
    int nth_point = 0, gale_n_in_partition = 0;
    int search_point = 2665, force_point = -1;
    std::cout << "Gale has " << gale.clusterVertices[FORCE_PARTITION].size() << " points in the partition" << std::endl;
    // Have to maintain this for SFCP, but may be able to steal from GALE internals to save processing
    for (SimplexId &vid : gale.clusterVertices[FORCE_PARTITION]) {
        host_vvi[nth_point] = vid;
        host_ivvi[vid] = nth_point;
        host_partition[nth_point] = (gale.hIndexArray_[vid] == FORCE_PARTITION);
        gale_n_in_partition += (gale.hIndexArray_[vid] == FORCE_PARTITION);
        #if CONSTRAIN_BLOCK
        if (vid == search_point) {
            force_point = nth_point;
            std::cout << "Targeting global id " << search_point << " as block " << nth_point << std::endl;
        }
        #endif
        nth_point++;
    }
    std::cout << "Gale found " << gale_n_in_partition << " of its points to be in partition " << FORCE_PARTITION << std::endl;
    CUDA_WARN(cudaMemcpyAsync(devi_vvi, host_vvi, vvi_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpyAsync(devi_ivvi, host_ivvi, ivvi_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpyAsync(devi_partition, host_partition, partition_size, cudaMemcpyHostToDevice));

    std::cout << "[TIMER] GALE SFCP preparation: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    // Critical Points
    // Set kernel launch parameters here
    // Parallelized VV on second-dimension (number of adjacenies to each vertex)
    dim3 grid_size = vertexCount,
         thread_block_size = SFCP_ColumnNum;
    #if CONSTRAIN_BLOCK
    grid_size.x = 1;
    #endif
    // Need one entry per block (VV width) + 2 entries for tracking edits to # upper and #lower components
    const int shared_mem_size = (sizeof(unsigned int) * 2) + (sizeof(vtkIdType) * SFCP_ColumnNum);

    // vertexMap serves as local->global (vvi for me)
    // inverse VertexMap not defined

    critPoints<<<grid_size, thread_block_size, shared_mem_size>>>(
            dDataArray,
            dSizeArray,
            dVertexMap,
            device_valences,
            nPoints,
            SFCP_ColumnNum,
            device_scalar_values,
            devi_partition,
            devi_vvi, // vvi for me, should be 0-init with nth entry == global_id
            devi_ivvi, // 0 by default, global_id'th entry == nth point
            #if CONSTRAIN_BLOCK
            force_point,
            #endif
            device_CPC);
    CUDA_WARN(cudaDeviceSynchronize()); // Make algorithm timing accurate
    // Results are retrieved from GPU
    std::cout << "[ScalarFieldCriticalPoints] Time usage for computation: " << compat_timer.getElapsedTime() << " s." << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Memory usage: " << m.getValue_in_MB(false) << " MB." << std::endl;
    compat_timer.reStart();
    CUDA_WARN(cudaMemcpy(host_CPC, device_CPC, classes_size, cudaMemcpyDeviceToHost));
    std::cout << "[TIMER] Retrieve results from GPU: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();

    // TODO: Make output comparable to standalone
    int numMinimum = 0, numMaximum = 0, numSaddles = 0, numRegular = 0, numVoid = 0;

    for (int i = 0; i < gale.clusterVertices[FORCE_PARTITION].size(); i++) {
        int global_idx = gale.clusterVertices[FORCE_PARTITION][i],
            idx = (i*3)+2;
        if (host_CPC[idx] == MINIMUM_CLASS) {
            numMinimum++;
            std::cout << "Minimum noted at i=" << i << ", global_idx=" << global_idx << std::endl;
        } else if (host_CPC[idx] == MAXIMUM_CLASS) {
            numMaximum++;
        } else if (host_CPC[idx] == SADDLE_CLASS) {
            numSaddles++;
        } else if (host_CPC[idx] == REGULAR_CLASS) {
            numRegular++;
        }
        else {
            numVoid++; // You don't want this to ever trip!
        }
    }
    std::cout << "[ScalarFieldCriticalPoints] Number of minimum points: " << numMinimum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of maximum points: " << numMaximum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of saddle points: " << numSaddles << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of regular points: " << numRegular << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of void points: " << numVoid << std::endl;
    return 0;
}
