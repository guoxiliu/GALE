#include "ttk/Relations.cuh"
#include "ttk/Timer.h"
#include <cstddef>
#include <cusparse.h>
#include <iostream>
#include <set>
#include <stdint.h>
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
    vtkIdType *hConnectivity, *hOffsets;
    vtkIdType *dConnectivity, *dOffsets;
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
        hConnectivity = static_cast<vtkIdType *>(
                cells->GetConnectivityArray()->GetVoidPointer(0));
        hOffsets =
                static_cast<vtkIdType *>(cells->GetOffsetsArray()->GetVoidPointer(0));
    }
    int verticesPerCell = hOffsets[1] - hOffsets[0];

    cudaMallocAsync(&dConnectivity, sizeof(vtkIdType) * (tetraNumber * verticesPerCell), 0);
    cudaMallocAsync(&dOffsets, sizeof(vtkIdType) * tetraNumber, 0);
    cudaMemcpyAsync(dConnectivity, hConnectivity,
                                    sizeof(vtkIdType) * (tetraNumber * verticesPerCell),
                                    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dOffsets, hOffsets, sizeof(vtkIdType) * tetraNumber,
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
        int clusterId = indexArray[hConnectivity[hOffsets[cid]]];
        for (int j = 1; j < verticesPerCell; ++j) {
            int tmpCluster = indexArray[hConnectivity[hOffsets[cid] + j]];
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
                edgeIds[0] = hConnectivity[hOffsets[cid] + j];
                if (edgeIds[0] >= hVertexIntervals[clusterId]) break;
                for (int k = j + 1; k < verticesPerCell; ++k) {
                    edgeIds[1] = hConnectivity[hOffsets[cid] + k];

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
                    edgeIds[0] = hConnectivity[hOffsets[cid] + j];
                    edgeIds[1] = hConnectivity[hOffsets[cid] + k];
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

    // Create sparse matrix using CSR format
    // csrVal: all 1s, the number = tetraNumber * 4
    // csrRowPtr: arithmetic sequence with 4 being common difference (= offsets array) 
    // csrColInd: the vertex indices for each tetra (= connectivity array)
    ttk::Timer t;
    int nnz = 4 * tetraNumber;
    int *hCsrRowPtr = new int[tetraNumber + 1];
    int *hCsrColInd = new int[nnz];
    for (int i = 0; i < tetraNumber; ++i) {
        hCsrRowPtr[i] = hOffsets[i];
        for (int j = 0; j < 4; ++j) {
            hCsrColInd[hOffsets[i] + j] = hConnectivity[hOffsets[i] + j];
        }
    }
    hCsrRowPtr[tetraNumber] = hOffsets[tetraNumber];
    // float *dCsrValues; 
    float *dCsrValues; 
    int *dCsrRowPtr, *dCsrColInd;
    
    cudaEventRecord(start);
    CHECK(cudaMalloc(&dCsrValues, sizeof(float) * nnz));
    CHECK(cudaMalloc(&dCsrRowPtr, sizeof(int) * (tetraNumber + 1)));
    CHECK(cudaMalloc(&dCsrColInd, sizeof(int) * nnz));

    CHECK(cudaMemset(dCsrValues, 1.0f, sizeof(float) * nnz));
    CHECK(cudaMemcpy(dCsrRowPtr, hCsrRowPtr, sizeof(int) * (tetraNumber + 1), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dCsrColInd, hCsrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice));


    // transpose the TV matrix
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    float *dTransValues;
    int *dTransColPtr, *dTransRowInd;
    CHECK(cudaMalloc((void **) &dTransValues, nnz * sizeof(float)));
    CHECK(cudaMalloc((void **) &dTransColPtr, (vertexNumber + 1) * sizeof(int)));
    CHECK(cudaMalloc((void **) &dTransRowInd, nnz * sizeof(int)));

    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( 
        cusparseCsr2cscEx2_bufferSize(handle, tetraNumber, vertexNumber, nnz,
                                      dCsrValues, dCsrRowPtr, dCsrColInd,
                                      dTransValues, dTransColPtr, dTransRowInd,
                                      CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
                                      CUSPARSE_CSR2CSC_ALG2, &bufferSize))
    CHECK(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUSPARSE(cusparseCsr2cscEx2(handle, tetraNumber, vertexNumber, nnz,
                                        dCsrValues, dCsrRowPtr, dCsrColInd,
                                        dTransValues, dTransColPtr, dTransRowInd,
                                        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
                                        CUSPARSE_CSR2CSC_ALG2, dBuffer))

    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                __FILE__, __LINE__);
        if (abort)
            exit(status);
    }

    int *hTransColPtr = new int[vertexNumber + 1];
    int *hTransRowInd = new int[nnz];
    CHECK(cudaMemcpy(hTransColPtr, dTransColPtr, sizeof(int) * (vertexNumber + 1), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hTransRowInd, dTransRowInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);

    std::vector<std::vector<int>> VTRelation(vertexNumber);
    for (int vid = 0; vid < vertexNumber; ++vid) {
        int vtSize = hTransColPtr[vid+1] - hTransColPtr[vid];
        VTRelation[vid] = std::vector<int>(vtSize);
        for (int j = 0; j < vtSize; ++j) {
            VTRelation[vid][j] = hTransRowInd[hTransColPtr[vid] + j];
        }
    }

    float computeTime = 0;
    cudaEventElapsedTime(&computeTime, start, stop);
    printf("Time measured for computation: %.5f s\n", computeTime/1000);
    printf("Total time measured for computation: %.5f s\n", t.getElapsedTime());



    // CHECK_CUSPARSE( cusparseCreateCsr(&matBTrans, vertexNumber, tetraNumber, nnz,
    //                                   dTransColPtr, dTransRowInd, transValues,
    //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I) )

    CHECK_CUSPARSE(cusparseDestroy(handle))
    CHECK(cudaFree(dTransRowInd))
    CHECK(cudaFree(dTransColPtr))
    CHECK(cudaFree(dTransValues))
    CHECK(cudaFree(dCsrColInd))
    CHECK(cudaFree(dCsrRowPtr))
    CHECK(cudaFree(dCsrValues))
    CHECK(cudaFree(dConnectivity))
    CHECK(cudaFree(dOffsets))
    CHECK(cudaFree(dIndexArray))

    delete[] hCsrRowPtr;
    delete[] hCsrColInd;

    return 0;
}