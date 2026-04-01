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
#include <pthread.h>

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
//char *class_strmap[5] = {"VOID", "MAXIMUM", "MINIMUM", "REGULAR", "SADDLE"};

// Kernel separated to reduce burden of recompiling
#include "gpu_kernel.h"

using namespace ttk;

typedef struct {
    int blockNumber = 1,
        blockSize = 128,
        threadNumber = 1,
        SFCP_threads = -1,
        SFCP_GPUs = -1,
        numThreadsPerCluster = 32,
        specific_partition = -1,
        SFCP_ColumnNum = 80;
    std::string fileName = "",
                inputArrayName = "",
                partitionArrayName = "";
} cli_args;

int parse(cli_args *args, int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    int c;

    while ((c = getopt(argc, argv, "i:a:p:b:c:s:t:v:f:u:g:")) != -1) {
        switch (c) {
            case 'i':
                args->fileName = optarg;
                break;
            case 'a':
                args->inputArrayName = optarg;
                break;
            case 'p':
                args->partitionArrayName = optarg;
                break;
            case 'b':
                args->blockNumber = atoi(optarg);
                break;
            case 'c':
                args->numThreadsPerCluster = atoi(optarg);
                break;
            case 's':
                args->blockSize = atoi(optarg);
                break;
            case 't':
                args->threadNumber = atoi(optarg);
                break;
            case 'v':
                args->SFCP_ColumnNum = atoi(optarg);
                break;
            case 'f':
                args->specific_partition = atoi(optarg);
                break;
            case 'u':
                args->SFCP_threads = atoi(optarg);
                break;
            case 'g':
                args->SFCP_GPUs = atoi(optarg);
                break;
        }
    }
    if (args->SFCP_threads < 1) {
        args->SFCP_threads = std::thread::hardware_concurrency();
    }
    if (args->SFCP_GPUs < 1) {
        CUDA_WARN(cudaGetDeviceCount(&args->SFCP_GPUs));
    }
    std::cout << "CMDLINE ARGS:" << std::endl
              << "Parsing vtu file (-i): " << args->fileName << std::endl
              << "Look for data array (-a): " << args->inputArrayName << std::endl
              << "Look for partition array (-p): " << args->partitionArrayName << std::endl
              << "Block Number (-b): " << args->blockNumber << std::endl
              << "numThreadsPerCluster (-c): " << args->numThreadsPerCluster << std::endl
              << "Block Size (-s): " << args->blockSize << std::endl
              << "CPU threads (-t): " << args->threadNumber << std::endl
              << "SFCP Column Num (-v): " << args->SFCP_ColumnNum << std::endl
              << "Target Partition (-f): " << args->specific_partition << std::endl
              << "SFCP CPU threads (-u): " << args->SFCP_threads << std::endl
              << "SFCP GPUs (-g): " << args->SFCP_GPUs << std::endl;
    return 0;
}

typedef struct {
    int numMinimum = 0,
        numMaximum = 0,
        numSaddles = 0,
        numRegular = 0,
        numVoid    = 0;
} sfcp_classification_counter;
/* Global memory for threads to utilize */
ttk::Gale                   * global_gale = NULL;
vtkDataArray                * global_inputArray = NULL;
cli_args                    * global_args = NULL;
sfcp_classification_counter * global_classifications = NULL;

// Parallelizable worker function
void * classify_partition(void * fn_args) {
    // Argument retrieval
    const int thread_id = *((int *)fn_args);
    vtkDataArray *inputArray = global_inputArray;
    cli_args *args = global_args;
    sfcp_classification_counter * partitionCount = &global_classifications[thread_id-1];
    ttk::Gale *gale = global_gale;
    const int nCells = static_cast<int>(gale->getNumberOfCells()),
              nPoints = static_cast<int>(gale->getNumberOfVertices()),
              nPartitions = gale->getNumberOfClusters();
    const int partition_burden = (nPartitions / args->SFCP_threads),
              previous_burden = 1 + (thread_id-1 < nPartitions % args->SFCP_threads ? thread_id-1 : nPartitions % args->SFCP_threads),
              bonus_burden = (thread_id-1 < nPartitions % args->SFCP_threads),
              initial_partition_id = previous_burden + ((thread_id-1) * partition_burden),
              final_partition_id = (initial_partition_id + partition_burden + bonus_burden > nPartitions) ? nPartitions : initial_partition_id + partition_burden + bonus_burden;

    ttk::Timer compat_timer;
    // GPU Identity and Stream setup
    int gpu_id = (thread_id % args->SFCP_GPUs);
    std::cout << "[CPU " << thread_id << "] Classify partitions "
              << initial_partition_id << " - " << final_partition_id
              << " on GPU " << gpu_id << std::endl;
    cudaSetDevice(gpu_id);
    // TODO: Allow use of different GPUs (requires GALE structures to be copied x-GPU if not resident on same GPU as GALE)
    cudaStream_t thread_stream;
    CUDA_WARN(cudaStreamCreate(&thread_stream));

    // One-time allocations and pointers for global mesh use
    int * HVertexIntervals     = gale->getHVertexIntervals(),
        * HCellIntervals       = gale->getHCellIntervals(),
        * HExternalCells       = gale->getHExternalCells(),
        * HExternalCellOffsets = gale->getHExternalCellOffsets(),
        clusterNum             = gale->getNumberOfClusters(),
        // THESE GALE STRUCTURES HAVE TO SUPPORT MULTI-GPU
        * dConnectivity        = gale->getDConnectivity(),
        * dOffsets             = gale->getDOffsets(),
        * dVertexIntervals     = gale->getDVertexIntervals(),
        * dCellIntervals       = gale->getDCellIntervals(),
        * dExternalCells       = gale->getDExternalCells(),
        * dExternalCellOffsets = gale->getDExternalCellOffsets(),
        * hIndexArray          = gale->hIndexArray_,
        // Local to partition, but allocate once to limit memory
        used_vertexCount  = -1,
        * dVertexMap      = nullptr,
        * dTotalVertexNum = nullptr,
        * dSizeArray      = nullptr,
        * dDataArray      = nullptr,
        * host_partition  = nullptr,
        * devi_partition  = nullptr;
    std::vector<std::vector<SimplexId>> clusterVertices = gale->clusterVertices;
    vtkIdType * host_vvi  = nullptr,
              * devi_vvi  = nullptr;
    // Some sizes are known at this point -- initialize these ones here
    size_t classes_size = sizeof(unsigned int) * nPoints * 3;
    unsigned int * host_CPC   = new unsigned int[nPoints*3],
                 * device_CPC = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_CPC, classes_size));

    size_t valences_size = sizeof(int) * nPoints * args->SFCP_ColumnNum;
    int * device_valences = nullptr; // No host-side version, just GPU buffer
    CUDA_ASSERT(cudaMalloc((void**)&device_valences, valences_size));

    size_t scalars_size = sizeof(double) * nPoints;
    double * device_scalar_values = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_scalar_values, scalars_size));

    size_t ivvi_size = sizeof(vtkIdType) * nPoints;
    vtkIdType * host_ivvi = new vtkIdType[nPoints],
              * devi_ivvi = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&devi_ivvi, ivvi_size));

    double * vertexAttributes = new double[nPoints];
    for (vtkIdType i = 0; i < nPoints; i++) {
        vertexAttributes[i] = inputArray->GetTuple1(i);
    }

    // This single int never has to be reallocated, so allocate it once here
    CHECK(cudaMalloc(&dTotalVertexNum, sizeof(int)));



    for (int FORCE_PARTITION = initial_partition_id;
             FORCE_PARTITION < nPartitions && FORCE_PARTITION < final_partition_id;
             FORCE_PARTITION += 1) // TODO: Convert increment to args->SFCP_threads when ready for bulk VV
    {
        if (args->specific_partition >= 0) FORCE_PARTITION = args->specific_partition;

        int n_in_partition = HVertexIntervals[FORCE_PARTITION] - HVertexIntervals[FORCE_PARTITION-1],
            internal_partition_cells = HCellIntervals[FORCE_PARTITION] - HCellIntervals[FORCE_PARTITION-1],
            external_partition_cells = HExternalCellOffsets[FORCE_PARTITION] - HExternalCellOffsets[FORCE_PARTITION-1],
            vertexCount = n_in_partition + 3 * (internal_partition_cells + external_partition_cells);
        if (n_in_partition == 0) continue;
        /*
        std::cout << "[CPU " << thread_id << "] Found " << n_in_partition
                  << " points in partition " << FORCE_PARTITION
                  << " with at most " << vertexCount
                  << " vertices (" << internal_partition_cells
                  << " internal and " << external_partition_cells
                  << " external)" << std::endl;
        */
        // Per-partition allocations can persist between loops unless they need to grow

        size_t vertices_size = vertexCount * sizeof(int),
               vv_data_size  = vertexCount * args->SFCP_ColumnNum * sizeof(int);
        if (vertexCount > used_vertexCount) {
            if (dVertexMap != nullptr) {
                CHECK(cudaFree(dVertexMap));
            }
            CHECK(cudaMalloc(&dVertexMap, vertices_size));
            if (dSizeArray != nullptr) {
                CHECK(cudaFree(dSizeArray));
            }
            CHECK(cudaMalloc(&dSizeArray, vertices_size));
            if (dDataArray != nullptr) {
                CHECK(cudaFree(dDataArray));
            }
            CHECK(cudaMalloc(&dDataArray, vv_data_size));

            // DO NOT update used_vertexCount here -- it needs to be referenced later for other allocations!
        }
        // Copy & (Re)set
        CHECK(cudaMemcpyAsync(dTotalVertexNum, &n_in_partition, sizeof(int), cudaMemcpyHostToDevice, thread_stream));
        CHECK(cudaMemsetAsync(dVertexMap, -1, vertices_size, thread_stream));
        CHECK(cudaMemsetAsync(dSizeArray, 0, vertices_size, thread_stream));
        CHECK(cudaMemsetAsync(dDataArray, -1, vv_data_size, thread_stream));

        // Call to kernel
        /* NOTE: block size can be too big! Have to limit
           thread_block_size ( -b * -s ) / (-p) = #clusters to compute in one kernel
         */
        //std::cout << "Start cVNWE() kernel" << std::endl;
        computeVertexNeighborsWithExternals<<<args->blockNumber, args->blockSize, 0, thread_stream>>>(
                dDataArray, dSizeArray, dVertexMap, dTotalVertexNum,
                FORCE_PARTITION, // target partition
                /* n_partitions_to_compute, */
                clusterNum, // total number of partitions
                dConnectivity, dOffsets, dVertexIntervals,
                dCellIntervals, dExternalCells, dExternalCellOffsets, 3,
                args->SFCP_ColumnNum
                );
        cudaError_t code = cudaGetLastError();
        if (code != cudaSuccess) {
            std::cerr << "[CPU " << thread_id
                      << "] VV kernel launch failed (" << code << ") "
                      << cudaGetErrorString(code) << std::endl;
        }

        // No need to pull back to CPU, just synchronize with the device for timing purposes
        CUDA_WARN(cudaStreamSynchronize(thread_stream));
        std::cout << "[TIMER] GALE VV: [CPU " << thread_id << "] " << compat_timer.getElapsedTime()
                  << std::endl;
        compat_timer.reStart();

        /* VVI, IVVI, partitions */
        size_t vvi_size = sizeof(vtkIdType) * vertexCount,
               partition_size = sizeof(int) * vertexCount;
        if (vertexCount > used_vertexCount) {
            /*
            std::cout << "[CPU " << thread_id
                      << " ] (Re)alloc vvi, partitions for " << vertexCount
                      << " vertices" << std::endl;
            */
            if (host_vvi != nullptr) {
                delete host_vvi;
                CHECK(cudaFree(devi_vvi));
            }
            host_vvi = new vtkIdType[vertexCount];
            CUDA_ASSERT(cudaMalloc((void**)&devi_vvi, vvi_size));
            if (host_partition != nullptr) {
                delete host_partition;
                CHECK(cudaFree(devi_partition));
            }
            host_partition = new int[vertexCount]();
            CUDA_ASSERT(cudaMalloc((void**)&devi_partition, partition_size));
            // Denote new size to prevent unnecessary reallocations
            used_vertexCount = vertexCount;
        }
        else {
            // Zero out partition always
            bzero(host_partition, partition_size);
        }
        // Always have to reset ivvi, valences, and CPC
        bzero(host_ivvi, ivvi_size);
        bzero(host_CPC, classes_size);
        CHECK(cudaMemsetAsync(device_valences, 0, valences_size, thread_stream));
        CHECK(cudaMemcpyAsync(device_CPC, host_CPC, classes_size, cudaMemcpyHostToDevice, thread_stream));

        int nth_point = 0,
        #if CONSTRAIN_BLOCK
            search_point = 2692,
            force_point = -1,
        #endif
            gale_n_in_partition = 0;
        // Have to maintain this for SFCP, but may be able to steal from GALE internals to save processing
        //std::cout << "Set VVI/IVVI/Partition for " << FORCE_PARTITION << std::endl;
        for (SimplexId &vid : clusterVertices[FORCE_PARTITION]) {
            host_vvi[nth_point] = vid;
            host_ivvi[vid] = nth_point;
            host_partition[nth_point] = (hIndexArray[vid] == FORCE_PARTITION);
            gale_n_in_partition += (hIndexArray[vid] == FORCE_PARTITION);
            #if CONSTRAIN_BLOCK
            if (vid == search_point) {
                force_point = nth_point;
                std::cout << "[CPU " << thread_id
                          << "] Targeting global id " << search_point
                          << " as block " << nth_point << std::endl;
            }
            #endif
            nth_point++;
        }
        #if CONSTRAIN_BLOCK
        if (force_point < 0) {
            std::cout << "[CPU " << thread_id
                      << "] Requested global id " << search_point
                      << " not found in partition, using point 0 for safety"
                      << std::endl;
            force_point = 0;
        }
        #endif
        //std::cout << "Memory copies to GPU" << std::endl;
        CUDA_WARN(cudaMemcpyAsync(devi_vvi, host_vvi, vvi_size, cudaMemcpyHostToDevice, thread_stream));
        CUDA_WARN(cudaMemcpyAsync(devi_ivvi, host_ivvi, ivvi_size, cudaMemcpyHostToDevice, thread_stream));
        CUDA_WARN(cudaMemcpyAsync(devi_partition, host_partition, partition_size, cudaMemcpyHostToDevice, thread_stream));
        CUDA_WARN(cudaMemcpyAsync(device_scalar_values, vertexAttributes, scalars_size, cudaMemcpyHostToDevice, thread_stream));


        std::cout << "[TIMER] GALE SFCP preparation [CPU "
                  << thread_id << "]: "
                  << compat_timer.getElapsedTime() << std::endl;
        compat_timer.reStart();

        // SFCP kernel
        dim3 grid_size = vertexCount,
             thread_block_size = args->SFCP_ColumnNum;
        #if CONSTRAIN_BLOCK
        grid_size.x = 1;
        #endif
        // Need one entry per block (VV width) + 2 entries for tracking edits to # upper and #lower components
        const int shared_mem_size = (sizeof(unsigned int) * 2) + (sizeof(vtkIdType) * args->SFCP_ColumnNum);

        //dsv_check<<<1,1>>>(device_scalar_values);
        critPoints<<<grid_size, thread_block_size, shared_mem_size, thread_stream>>>(
                dDataArray, dSizeArray, dVertexMap, device_valences, nPoints,
                args->SFCP_ColumnNum, device_scalar_values, devi_partition,
                devi_vvi, devi_ivvi,
                #if CONSTRAIN_BLOCK
                force_point,
                #endif
                device_CPC);
        //dsv_check<<<1,1>>>(device_scalar_values);
        CUDA_WARN(cudaStreamSynchronize(thread_stream)); // Make algorithm timing accurate
        // Results are retrieved from GPU
        std::cout << "[ScalarFieldCriticalPoints] Time usage for computation [CPU "
                  << thread_id << "]: "
                  << compat_timer.getElapsedTime() << " s." << std::endl;
        /*
        MemoryUsage m;
        std::cout << "[ScalarFieldCriticalPoints] Memory usage: "
                  << m.getValue_in_MB(false) << " MB." << std::endl;
        */
        compat_timer.reStart();
        CUDA_WARN(cudaMemcpyAsync(host_CPC, device_CPC, classes_size, cudaMemcpyDeviceToHost, thread_stream));
        std::cout << "[TIMER] Retrieve results from GPU [CPU "
                  << thread_id << "]: "
                  << compat_timer.getElapsedTime() << std::endl;
        compat_timer.reStart();

        for (int i = 0; i < clusterVertices[FORCE_PARTITION].size(); i++) {
            int global_idx = clusterVertices[FORCE_PARTITION][i],
                idx = (i*3)+2;
            if (host_CPC[idx] == MINIMUM_CLASS) {
                partitionCount->numMinimum++;
            } else if (host_CPC[idx] == MAXIMUM_CLASS) {
                partitionCount->numMaximum++;
            } else if (host_CPC[idx] == SADDLE_CLASS) {
                partitionCount->numSaddles++;
            } else if (host_CPC[idx] == REGULAR_CLASS) {
                partitionCount->numRegular++;
            }
            else {
                partitionCount->numVoid++; // You don't want this to ever trip!
            }
        }
        std::cout << "[ScalarFieldCriticalPoints] [CPU " << thread_id
                  << "] Partition " << FORCE_PARTITION
                  << " Number of minimum points: "
                  << partitionCount->numMinimum << std::endl
                  << "[ScalarFieldCriticalPoints] [CPU " << thread_id
                  << "] Partition " << FORCE_PARTITION
                  << " Number of maximum points: "
                  << partitionCount->numMaximum << std::endl
                  << "[ScalarFieldCriticalPoints] [CPU " << thread_id
                  << "] Partition " << FORCE_PARTITION
                  << " Number of saddle points: "
                  << partitionCount->numSaddles << std::endl
                  << "[ScalarFieldCriticalPoints] [CPU " << thread_id
                  << "] Partition " << FORCE_PARTITION
                  << " Number of regular points: "
                  << partitionCount->numRegular << std::endl
                  << "[ScalarFieldCriticalPoints] [CPU " << thread_id
                  << "] Partition " << FORCE_PARTITION
                  << " Number of void points: "
                  << partitionCount->numVoid << std::endl
                  << "[TIMER] Tabulate partition results [CPU "
                  << thread_id << "]: "
                  << compat_timer.getElapsedTime() << std::endl;
        compat_timer.reStart();

        if (args->specific_partition >= 0) break;
        // Reset CPC?
    }
}

int main(int argc, char *argv[])
{
    // Initialize program: Parse args and handle non-TTK/non-GALE libraries
    ttk::Timer compat_timer;
    cli_args args;
    int parse_return = parse(&args, argc, argv);
    if(parse_return != 0) return parse_return;
    std::cout << "[TIMER] Argument parsing: " << compat_timer.getElapsedTime()
              << std::endl;
    compat_timer.reStart();
    // Creating the CUDA Context helps to stablize timings
    (void)cudaFree(0);
    std::cout << "[TIMER] Create CUDA Context: " << compat_timer.getElapsedTime()
              << std::endl;
    compat_timer.reStart();



    // Initialize TTK and GALE
    // TTK
    vtkSmartPointer<vtkXMLGenericDataObjectReader> reader =
        vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(args.fileName.c_str());
    reader->Update();
    vtkDataObject *inputDataObject = reader->GetOutput();
    if(!inputDataObject) {
        std::cerr << "Unable to read input file `" << args.fileName << "'"
                  << std::endl;
        return 1;
    }
    vtkDataSet *inputDataSet = vtkDataSet::SafeDownCast(inputDataObject);
    int ScalarFieldIndex = 0;
    vtkDataArray *inputArray = GetInputArray(args.inputArrayName, inputDataSet, ScalarFieldIndex);

    // GALE
    ttk::Gale gale(args.threadNumber, args.numThreadsPerCluster, args.SFCP_ColumnNum);
    // reorder=True is BUGGED; use FALSE (our dataset preprocessing should be compatible with this always)
    gale.initialize(inputDataSet, args.blockNumber, args.blockSize, false);
    const int nCells = static_cast<int>(gale.getNumberOfCells()),
              nPoints = static_cast<int>(gale.getNumberOfVertices()),
              nPartitions = gale.getNumberOfClusters();
    std::cout << "Gale loaded " << nCells << " cells and " << nPoints
              << " points across " << nPartitions << " partitions"
              << std::endl;
    std::cout << "[TIMER] Initialize VTK: " << compat_timer.getElapsedTime() << std::endl;
    compat_timer.reStart();



    sfcp_classification_counter MeshClassifications;
    int clusterNum = gale.getNumberOfClusters(),
        n_parallel = args.SFCP_threads,
        pids[args.SFCP_threads];
    // Set up global memory for threads to utilize
    global_gale = &gale;
    global_inputArray = inputArray;
    global_args = &args;
    global_classifications = new sfcp_classification_counter[args.SFCP_threads]();

    pthread_t threads[args.SFCP_threads];
    for (int i = 1; i < n_parallel+1; i++) {
        pids[i-1] = i;
        pthread_create(&threads[i-1], NULL, classify_partition, (void*)&pids[i-1]);
    }
    for (int i = 1; i < n_parallel+1; i++) {
        pthread_join(threads[i-1], NULL);
        sfcp_classification_counter *pc_pointer = &global_classifications[i-1];
        MeshClassifications.numMinimum += pc_pointer->numMinimum;
        MeshClassifications.numMaximum += pc_pointer->numMaximum;
        MeshClassifications.numSaddles += pc_pointer->numSaddles;
        MeshClassifications.numRegular += pc_pointer->numRegular;
        MeshClassifications.numVoid    += pc_pointer->numVoid;
    }

    std::cout << "[ScalarFieldCriticalPoints] Number of minimum points: " << MeshClassifications.numMinimum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of maximum points: " << MeshClassifications.numMaximum << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of saddle points:  " << MeshClassifications.numSaddles << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of regular points: " << MeshClassifications.numRegular << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Number of void points:    " << MeshClassifications.numVoid << std::endl;
    return 0;
}

