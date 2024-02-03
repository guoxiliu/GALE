#include "ttk/Relations.cuh"
#include <cstddef>
#include <iostream>
#include <set>
#include <stdlib.h>
#include <vector>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        if(abort) exit(error);                                                 \
    }                                                                          \
}


int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string fileName = argv[2];
    // std::cout << "Parsing vtu file: " << fileName << std::endl;

    // read all the data from the file
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
            vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();

    int tetraNumber = 0;
    int vertexNumber = 0;
    // int edgeNumber = 0;

    // create the triangulation
    vtkUnstructuredGrid *ugrid = reader->GetOutput();
    vtkPoints *points = ugrid->GetPoints();
    if (!points) {
        std::cerr << "DataSet has uninitialized `vtkPoints`." << std::endl;
    }

    vtkCellArray *cells = static_cast<vtkUnstructuredGrid *>(ugrid)->GetCells();
    if (!cells) {
        std::cerr << "DataSet has uninitialized `vtkCellArray`." << std::endl;
    }

    // Points
    {
        auto pointDataType = points->GetDataType();
        if (pointDataType != VTK_FLOAT && pointDataType != VTK_DOUBLE) {
            std::cerr << "Unable to initialize 'ttk::Triangulation' for point "
                                     "precision other than 'float' or 'double'."
                                << std::endl;
        }

        void *pointDataArray = points->GetData()->GetVoidPointer(0);
        vertexNumber = points->GetNumberOfPoints();
    }

    // Cells
    vtkIdType *connectivity, *offsets;
    vtkIdType *dConnectitity, *dOffsets;
    tetraNumber = cells->GetNumberOfCells();
    if (tetraNumber > 0) {
        if (!cells->IsStorage64Bit()) {
            if (cells->CanConvertTo64BitStorage()) {
                std::cout << "Converting the cell array to 64-bit storage" << std::endl;
                bool success = cells->ConvertTo64BitStorage();
                if (!success) {
                    std::cerr
                            << "Error converting the provided cell array to 64-bit storage"
                            << std::endl;
                }
            } else {
                std::cerr << "Cannot convert the provided cell array to 64-bit storage"
                                    << std::endl;
            }
        }
        connectivity = static_cast<vtkIdType *>(
                cells->GetConnectivityArray()->GetVoidPointer(0));
        offsets =
                static_cast<vtkIdType *>(cells->GetOffsetsArray()->GetVoidPointer(0));
    }
    int verticesPerCell = offsets[1] - offsets[0];

    cudaMallocAsync(&dConnectitity, sizeof(vtkIdType) * (tetraNumber * verticesPerCell), 0);
    cudaMallocAsync(&dOffsets, sizeof(vtkIdType) * tetraNumber, 0);
    cudaMemcpyAsync(dConnectitity, connectivity,
                                    sizeof(vtkIdType) * (tetraNumber * verticesPerCell),
                                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dOffsets, offsets, sizeof(vtkIdType) * tetraNumber,
                                    cudaMemcpyHostToDevice);

    // index array
    int *indexArray = NULL, *dIndexArray;
    {
        int hasIndexArray = ugrid->GetPointData()->HasArray("_index");
        if (hasIndexArray) {
            indexArray = (int *)ugrid->GetPointData()
                                        ->GetAbstractArray("_index")
                                        ->GetVoidPointer(0);
            for (int vid = 0; vid < vertexNumber; ++vid) indexArray[vid] += 1;
            cudaMallocAsync(&dIndexArray, sizeof(int) * vertexNumber, 0);
            cudaMemcpyAsync(dIndexArray, indexArray, sizeof(int) * vertexNumber,
                                            cudaMemcpyHostToDevice);
        } else {
            std::cerr << "Cannot detect index array!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    int nodeNumber = indexArray[vertexNumber - 1] + 1;
    // int edgeCount[nodeNumber+1], *dEdgeCount;
    printf("Node number: %d, cell number: %d, vertex number: %d\n", nodeNumber,
                 tetraNumber, vertexNumber);

    // Host matrix definition

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // compute vertex intervals 
    std::vector<int> hVertexIntervals(nodeNumber, 0);
    int *dVertexIntervals;
    cudaMalloc(&dVertexIntervals, sizeof(int) * nodeNumber);
    for (int vid = 0; vid < vertexNumber; ++vid) {
        hVertexIntervals[indexArray[vid]]++;
    }
    for (int i = 1; i < nodeNumber; ++i) {
        hVertexIntervals[i] += hVertexIntervals[i-1];
    }

    cudaMemcpy(dVertexIntervals, hVertexIntervals.data(), sizeof(int) * nodeNumber, cudaMemcpyHostToDevice);


    // compute cell intervals
    std::vector<std::vector<int>> externalCells(nodeNumber);
    std::vector<int> hExternalCellOffsets(nodeNumber, 0);
    int *hCellIntervals = (int *) calloc(nodeNumber, sizeof(int));

    int *dCellIntervals, *dExternalCells, *dExternalCellOffsets;
    CHECK(cudaMalloc(&dCellIntervals, sizeof(int) * nodeNumber));
    CHECK(cudaMalloc(&dExternalCellOffsets, sizeof(int) * nodeNumber));

    for (int cid = 0; cid < tetraNumber; ++cid) {
        int clusterId = indexArray[connectivity[offsets[cid]]];
        for (int j = 1; j < verticesPerCell; ++j) {
            int tmpCluster = indexArray[connectivity[offsets[cid] + j]];
            if (tmpCluster > clusterId) {
                if (externalCells[tmpCluster].empty() || externalCells[tmpCluster].back() != cid) {
                    externalCells[tmpCluster].push_back(cid);
                }
            }
        }
        hCellIntervals[clusterId]++;
    }

    for (int i = 1; i < nodeNumber; ++i) {
        hCellIntervals[i] += hCellIntervals[i-1];
        hExternalCellOffsets[i] = hExternalCellOffsets[i-1] + externalCells[i].size();
    }
    CHECK(cudaMemcpy(dCellIntervals, hCellIntervals, sizeof(int) * nodeNumber, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dExternalCellOffsets, hExternalCellOffsets.data(), sizeof(int) * nodeNumber, cudaMemcpyHostToDevice));


    // Convert the 2D vector to 1D vector
    int *hExternalCells = new int[hExternalCellOffsets[nodeNumber-1]];
    int *tmp = hExternalCells;
    CHECK(cudaMalloc((void **) &dExternalCells, sizeof(int) * hExternalCellOffsets[nodeNumber-1]));
    for (int i = 1; i < externalCells.size(); ++i) {
        std::copy(externalCells[i].begin(), externalCells[i].end(), tmp);
        tmp += externalCells[i].size();
    }
    CHECK(cudaMemcpy(dExternalCells, hExternalCells, sizeof(int) * hExternalCellOffsets[nodeNumber-1], cudaMemcpyHostToDevice));
    // delete[] hExternalCells;

    // compute edges and edge intervals
    // int edgeNumber = 0;
    int *hEdgeIntervals = (int *) calloc(nodeNumber, sizeof(int));
    int *dEdgeIntervals;
    cudaMallocAsync(&dEdgeIntervals, sizeof(int) * nodeNumber, 0);
    std::vector<std::vector<std::array<int, 2>>> internalEdgeList(nodeNumber);
    for (int clusterId = 0; clusterId < nodeNumber; ++clusterId) {
        std::set<std::array<int, 2>> localEdgeSet;
        int startCellId = clusterId == 0 ? 0 : hCellIntervals[clusterId-1];
        for (int cid = startCellId; cid < hCellIntervals[clusterId]; ++cid) {
            std::array<int, 2> edgeIds;
            for (int j = 0; j < verticesPerCell - 1; ++j) {
                edgeIds[0] = connectivity[offsets[cid] + j];
                if (edgeIds[0] >= hVertexIntervals[clusterId]) break;
                for (int k = j + 1; k < verticesPerCell; ++k) {
                    edgeIds[1] = connectivity[offsets[cid] + k];

                    if (localEdgeSet.find(edgeIds) == localEdgeSet.end()) {
                        hEdgeIntervals[clusterId]++;
                        localEdgeSet.insert(edgeIds);
                    }
                }
            }
        }

        // loop through the external cell list
        for (int &cid : externalCells[clusterId]) {
            std::array<int, 2> edgeIds;
            for (int j = 0; j < verticesPerCell - 1; ++j) {
                for (int k = j + 1; k < verticesPerCell; ++k) {
                    edgeIds[0] = connectivity[offsets[cid] + j];
                    edgeIds[1] = connectivity[offsets[cid] + k];
                    if (indexArray[edgeIds[0]] == clusterId && localEdgeSet.find(edgeIds) == localEdgeSet.end()) {
                        hEdgeIntervals[clusterId]++;
                        localEdgeSet.insert(edgeIds);
                    }
                }
            }
        }
        internalEdgeList[clusterId] = {localEdgeSet.begin(), localEdgeSet.end()};
    } 

    for (int i = 1; i < nodeNumber; ++i) {
        hEdgeIntervals[i] += hEdgeIntervals[i-1];
    }
    CHECK(cudaMemcpy(dEdgeIntervals, hEdgeIntervals, sizeof(int) * nodeNumber, cudaMemcpyHostToDevice));

    // TODO: compute triangles and triangle intervals

    
    /**
     * Testing output goes here
     */

    // for (int i = 1; i < nodeNumber; ++i) {
    //     std::cout << "cluster " << i << ": " << hExternalCellOffsets[i] << std::endl;
    // }

    // int maxVertexId = 0;
    // int startCellId = hCellIntervals[1];
    // std::cout << "cellInterval: " << hCellIntervals[2] << ", vertexInterval: " << hVertexIntervals[2] << std::endl;
    // for (int cid = hCellIntervals[2]; cid >= startCellId; --cid) {
    //     int lastVertexId = connectivity[offsets[cid] + 3];
    //     std::cout << "cid: " << cid << ", " << "last vertex: " << lastVertexId << " ";
    //     if (lastVertexId > maxVertexId) maxVertexId = lastVertexId;
    //     if (lastVertexId < hVertexIntervals[2]) break;     // early stop 
    // }
    // std::cout << "max vertex: " << maxVertexId << std::endl;


    // test the vertex edge relations 


    int blockSize = 32;
    int numBlock = 120;
    int baseCluster = 1;
    int numThreads = numBlock * blockSize;
    // int numThreads = numBlock;
    int endCluster = min(numThreads, nodeNumber - 1);
    const int localVertexNum = hVertexIntervals[endCluster] - hVertexIntervals[baseCluster-1];


    // int *dVertexEdgeMatrix;
    // int *dVertexEdgeSize;
    // std::vector<int> hVertexEdgeSize(localVertexNum);
    // std::vector<int> hVertexEdgeMatrix(localVertexNum * matrixColumnSmall);

    // // clock_t t = clock();
    // cudaEventRecord(start, 0);
    // CHECK(cudaMalloc((void **)&dVertexEdgeMatrix, sizeof(int) * localVertexNum * matrixColumnSmall));
    // CHECK(cudaMalloc((void **) &dVertexEdgeSize, sizeof(int) * localVertexNum));

    // precomputeVertexEdges<<<numBlock, blockSize>>>(dVertexEdgeMatrix, dVertexEdgeSize, baseCluster, dConnectitity, dOffsets, dIndexArray, 
    //                             dVertexIntervals, dCellIntervals, dExternalCells, dExternalCellOffsets, 3);

    // // cudaDeviceSynchronize();
    // CHECK(cudaMemcpy(hVertexEdgeSize.data(), dVertexEdgeSize, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(hVertexEdgeMatrix.data(), dVertexEdgeMatrix, sizeof(int) * localVertexNum * matrixColumnSmall, cudaMemcpyDeviceToHost));
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // // t = clock() - t;
    // // printf("Total time measured for computation: %.5f s\n", ((double) t / CLOCKS_PER_SEC));


    // float computeTime;
    // cudaEventElapsedTime(&computeTime, start, stop);
    // printf("Time measured for computation: %f s\n", computeTime / 1000);
    
    // cudaError_t status = cudaGetLastError();
    // if (status != cudaSuccess) {
    //     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //         __FILE__, __LINE__); if (abort) exit(status);
    // }

    // // check the number of vertex edges for the first cluster
    // for (int i = 0; i < localVertexNum; ++i) {
    //     std::cout << "vertex " << i + hVertexIntervals[baseCluster-1] << ": " << hVertexEdgeSize[i] << std::endl;
    // }

    // compute vertex star matrix
    int *dVertexStarMatrix;
    int *dVertexStarSize;
    std::vector<int> hVertexStarSize(localVertexNum);
    std::vector<int> hVertexStarMatrix(localVertexNum * matrixColumnSmall);

    // clock_t t = clock();
    cudaEventRecord(start, 0);
    CHECK(cudaMalloc((void **)&dVertexStarMatrix, sizeof(int) * localVertexNum * matrixColumnSmall));
    CHECK(cudaMalloc((void **) &dVertexStarSize, sizeof(int) * localVertexNum));
    CHECK(cudaMemset(dVertexStarSize, 0, sizeof(int) * localVertexNum));

    precomputeVertexStars<<<numBlock, blockSize>>>(dVertexStarMatrix, 
            dVertexStarSize, baseCluster, nodeNumber, 
            dConnectitity, dOffsets, dVertexIntervals, 
            dCellIntervals, dExternalCells, dExternalCellOffsets, 3);

    // cudaDeviceSynchronize();
    CHECK(cudaMemcpy(hVertexStarSize.data(), dVertexStarSize, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hVertexStarMatrix.data(), dVertexStarMatrix, sizeof(int) * localVertexNum * matrixColumnSmall, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // t = clock() - t;
    // printf("Total time measured for computation: %.5f s\n", ((double) t / CLOCKS_PER_SEC));

    {
        float computeTime;
        cudaEventElapsedTime(&computeTime, start, stop);
        printf("Time measured for vertex star computation: %f s\n", computeTime / 1000);
    }
    
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
            __FILE__, __LINE__); if (abort) exit(status);
    }

    // // check the number of vertex star for the first cluster
    // for (int i = 0; i < localVertexNum; ++i) {
    //     std::cout << "vertex " << i + hVertexIntervals[baseCluster-1] << ": " << hVertexStarSize[i] << std::endl;
    // }

    // compute vertex neighbor matrix
    int *dVertexNeighborMatrix;
    int *dVertexNeighborSize;
    std::vector<int> hVertexNeighborSize(localVertexNum);
    std::vector<int> hVertexNeighborMatrix(localVertexNum * matrixColumnSmall);

    // clock_t t = clock();
    cudaEventRecord(start, 0);
    CHECK(cudaMalloc((void **)&dVertexNeighborMatrix, sizeof(int) * localVertexNum * matrixColumnSmall));
    CHECK(cudaMalloc((void **) &dVertexNeighborSize, sizeof(int) * localVertexNum));
    CHECK(cudaMemset(dVertexNeighborSize, 0, sizeof(int) * localVertexNum));

    precomputeVertexNeighbors<<<numBlock, blockSize>>>(
                    dVertexNeighborMatrix, dVertexNeighborSize, 
                    baseCluster, nodeNumber, dConnectitity, dOffsets, 
                    dVertexIntervals, dCellIntervals, 
                    dExternalCells, dExternalCellOffsets, 3);

    // cudaDeviceSynchronize();
    CHECK(cudaMemcpy(hVertexNeighborSize.data(), dVertexNeighborSize, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hVertexNeighborMatrix.data(), dVertexNeighborMatrix, sizeof(int) * localVertexNum * matrixColumnSmall, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // t = clock() - t;
    // printf("Total time measured for computation: %.5f s\n", ((double) t / CLOCKS_PER_SEC));

    {
        float computeTime;
        cudaEventElapsedTime(&computeTime, start, stop);
        printf("Time measured for vertex neighbor computation: %f s\n", computeTime / 1000);
    }
    
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
            __FILE__, __LINE__); if (abort) exit(status);
    }

    // // check the output of vertex neighbor list
    // for (int i = 0; i < localVertexNum; ++i) {
    //     std::cout << "vertex " << i + hVertexIntervals[baseCluster-1] << ": "  << hVertexNeighborSize[i] << " [";
    //     for (int j = 0; j < hVertexNeighborSize[i]; ++j) {
    //         std::cout << hVertexNeighborMatrix[i * matrixColumnSmall + j] << " ";
    //     }
    //     std::cout << "]" << std::endl;
    // }

    // free host memory
    free(hCellIntervals);
    free(hEdgeIntervals);
    delete[] hExternalCells;

    // free device memory
    cudaFree(dVertexIntervals);
    cudaFree(dCellIntervals);
    cudaFree(dEdgeIntervals);
    cudaFree(dExternalCells);
    cudaFree(dExternalCellOffsets);
    // cudaFree(dVertexEdgeSize);
    // cudaFree(dVertexEdgeMatrix);
    cudaFree(dVertexStarSize);
    cudaFree(dVertexStarMatrix);
    cudaFree(dVertexNeighborSize);
    cudaFree(dVertexNeighborMatrix);

    return 0;
}