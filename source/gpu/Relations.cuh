#pragma once

#include <vtkType.h>
#include <cuda.h>
#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void multiply(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

__global__ void test2DArray(int **arr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 10; ++i) 
        arr[tid][i] *= 2;
}

__global__ void printArray(const int *arr, int start, int end) {
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = start; i < end; ++i) {
        printf("%d: %d\n", i, arr[i]);
    }
}

__device__ int binarySearch(const int arr[], const int length, const int target) {
    int left = 0, right = length-1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    // no exact match found, return negative
    return -(left+1);
}

__device__ void insertElement(int arr[], const int length, const int pos, const int target) {
    for (int i = length+1; i > pos; --i) {
        arr[i] = arr[i-1];
    }
    arr[pos] = target;
}

__global__ void countCells(int *localCellCounts, const vtkIdType *connectivity, const vtkIdType *offset, const int *vertexMap, const int cellNumber, const int dimension) {
    /* Divide the job by cells */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cellNumber) return;
    int cellId = offset[tid];
    int clusterId = vertexMap[connectivity[cellId]];
    // not needed if the cell array is already sorted
    // for (int j = 1; j <= dimension; ++j) {
    //     int tmp = vertexMap[connectivity[cellId + j]];
    //     if (tmp < clusterId) clusterId = tmp;
    // }
    atomicAdd(&localCellCounts[clusterId], 1);


    /* Divide the job by clusters */
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // for (int cid = 0; cid < cellNumber; ++cid) {
    //     int cellId = offset[cid];
    //     int clusterId = vertexMap[connectivity[cellId]];
    //     for (int j = 1; j <= dimension; ++j) {
    //         int tmp = vertexMap[connectivity[cellId + j]];
    //         if (tmp < clusterId) clusterId = tmp;
    //     }
    //     if (tid == clusterId) localCellCounts[tid]++;
    // }
}

__global__ void countExternalCells(int *externalCellOffsets, const vtkIdType *connectivity, const vtkIdType *offset, const int *vertexMap, const int cellNumber, const int dimension) {
    /* Divide the job by cells */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cellNumber) return;
    int cellId = offset[tid];
    int lastCluster = vertexMap[connectivity[cellId]];
    for (int j = 1; j <= dimension; ++j) {
        int clusterId = vertexMap[connectivity[cellId + j]];
        if (clusterId != lastCluster) {
            atomicAdd(&externalCellOffsets[clusterId], 1);
            lastCluster = clusterId;
        }
    }
}


__global__ void countExternalCells(int *externalCellOffsets, const vtkIdType *connectivity, const vtkIdType *offset, const int *cellIntervals, const int *vertexMap, const int clusterNumber, const int dimension) {
    /* Divide the job by clusters */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0 && tid < clusterNumber) {
        // externalCellOffsets[tid] = 0;
        int len = 0;
        // int *externalCellArray, len = 0;
        // cudaMalloc(&externalCellArray, dimension * clusterNumber);
        for (int i = 0; i < cellIntervals[tid-1]; ++i) {
            int cellId = offset[i];
            for (int j = 1; j <= dimension; ++j) {
                if (vertexMap[connectivity[cellId + j]] == tid) {
                    len++;
                    // externalCellArray[len++] = i;
                    break;
                }
            }
        }
        externalCellOffsets[tid] = len;
    }
}

__global__ void countVertices(int *localVertexCounts, const int *vertexMap, const int vertexNumber) {
    /* Divide the job by vertices */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vertexNumber) return;
    int clusterId = vertexMap[tid];
    atomicAdd(&localVertexCounts[clusterId], 1);

    // /* Divide the job by clusters */
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // for (int vid = 0; vid < vertexNumber; ++vid) {
    //     if (vertexMap[vid] == tid) localVertexCounts[tid]++;
    // }
}

__global__ void countEdges(int *edgeCounts, const int *vertexMap,
                            const vtkIdType *connectivity, const vtkIdType *offset,
                            const int cellNumber, const int dimension) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cellNumber) return;

    int cellId = offset[tid];
    for (int i = 0; i < dimension; ++i) {
        int clusterId = vertexMap[connectivity[cellId + i]];
        atomicAdd(&edgeCounts[clusterId], dimension - i);
    }

    // // loop through the internal cell list
    // for(int cid = cellIntervals[tid - 1] + 1;
    //     cid <= cellIntervals[tid]; cid++) {
    //     int edgeIds[2];

    //     // loop through each edge of the cell
    //     for(int j = 0; j < dimension; j++) {
    //         edgeIds[0] = connectivity[offset[cid] + j];
    //         // the edge does not belong to the current node
    //         if(edgeIds[0] > vertexIntervals[tid]) {
    //             break;
    //         }
    //         for(int k = j + 1; k <= dimension; k++) {
    //             edgeIds[1] = connectivity[offset[cid] + k];

    //             // // not found in the edge map - assign new edge id
    //             // if(localEdgeSet.find(edgeIds) == localEdgeSet.end()) {
    //             //     edgeCount++;
    //             //     localEdgeSet.insert(edgeIds);
    //             // }
    //             localEdgeCounts[tid]++;
    //         }
    //     }
    // }

    // // loop through the external cell list
    // for(const int &cid : externalCells_[tid]) {
    //     std::array<int, 2> edgeIds;

    //     // loop through each edge of the cell
    //     for(int j = 0; j < dimension; j++) {
    //         for(int k = j + 1; k <= dimension; k++) {
    //             edgeIds[0] = connectivity[offset[cid] + j];
    //             edgeIds[1] = connectivity[offset[cid] + k];

    //             // the edge is in the current node
    //             if(edgeIds[0] > vertexIntervals[tid - 1]
    //                 && edgeIds[0] <= vertexIntervals[tid]) {
    //                 if(localEdgeSet.find(edgeIds) == localEdgeSet.end()) {
    //                     edgeCount++;
    //                     localEdgeSet.insert(edgeIds);
    //                 }
    //             }
    //         }
    //     }
    // }
}

__global__ void countTriangles(int *triangleCounts, const int *vertexMap,
                        const vtkIdType *connectivity, const vtkIdType *offset,
                        const int cellNumber, const int dimension) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* divide the job by cells */
    if (tid >= cellNumber) return;

    int cellId = offset[tid];
    for (int i = 0; i < dimension - 1; ++i) {
        int clusterId = vertexMap[connectivity[cellId + i]];
        for (int j = i + 1; j < dimension; ++j) {
            int combination = dimension - j;
            atomicAdd(&triangleCounts[clusterId], combination);
        }
    }
}

__global__ void computeVertexCells(int *vertexCellSize, const vtkIdType *connectivity, const vtkIdType *offset, const int cellNumber, const int dimension) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cellNumber) return;

    int cellId = offset[tid];
    for (int i = 0; i <= dimension; ++i) {
        int vertexId = connectivity[cellId + i];
        atomicAdd(&vertexCellSize[vertexId], 1);
    }
}


__global__ void getVertexEdgesSize(int *vertexEdgeSizes, const int baseCluster,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *indexArray, const int *vertexIntervals, const int * cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    
    // each block thread computes one cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;

    /*
     * divide the job by clusters
     */
    // determine the range of vertices
    const int startVertexId = vertexIntervals[tid-1];
    const int startCellId = cellIntervals[tid-1];
    const int localVertexNum = vertexIntervals[tid]-startVertexId;
    const int columns = 30;
    // printf("localVertexNum: %d\n", localVertexNum);
    // printf("vertexEdgeSizes[0]: %p\n", vertexEdgeSizes[0]);
    
    int *vertexEdgeMatrix = new int[localVertexNum * columns];
    if (vertexEdgeMatrix == NULL) {
        printf("Cannot allocate memory!\n");
        return;
    }

    // // allocate 2D matrix
    // int **vertexEdgeMatrix = new int*[localVertexNum];
    // for (int i = 0; i < localVertexNum; ++i) {
    //     vertexEdgeMatrix[i] = new int[columns];
    //     if (vertexEdgeMatrix[i] == NULL) {
    //         printf("Cannot allocate memory!");
    //         return;
    //     }
    // }
    // for (int i = 0; i < matrixSize; ++i) vertexEdgeMatrix[i] = false;
    for (int i = 0; i < localVertexNum; ++i) vertexEdgeSizes[vertexIntervals[tid-1] - vertexIntervals[baseCluster-1] + i] = 0;

    // loop through the internal cell list
    for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[offsets[cid] + vj];
                int localVjId = vjId - vertexIntervals[tid-1];
                if (indexArray[viId] == tid) {
                    int relativeViId = viId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexEdgeMatrix + columns * localViId, vertexEdgeSizes[relativeViId], vjId);
                    if (pos < 0) {
                        insertElement(vertexEdgeMatrix + columns * localViId, vertexEdgeSizes[relativeViId], -pos-1, vjId);
                        atomicAdd(&vertexEdgeSizes[relativeViId], 1);
                    }
                } 
                if (indexArray[vjId] == tid) {
                    int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexEdgeMatrix + columns * localVjId, vertexEdgeSizes[relativeVjId], viId);
                    if (pos < 0) {
                        insertElement(vertexEdgeMatrix + columns * localVjId, vertexEdgeSizes[relativeVjId], -pos-1, viId);
                        atomicAdd(&vertexEdgeSizes[relativeVjId], 1);
                    }
                }
            }
        }
    }

    // loop through the external cell list
    for (int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                int localVjId = vjId - vertexIntervals[tid-1];
                if (indexArray[viId] == tid) {
                    int relativeViId = viId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexEdgeMatrix + columns * localViId, vertexEdgeSizes[relativeViId], vjId);
                    if (pos < 0) {
                        insertElement(vertexEdgeMatrix + columns * localViId, vertexEdgeSizes[relativeViId], -pos-1, vjId);
                        atomicAdd(&vertexEdgeSizes[relativeViId], 1);
                    }
                } 
                if (indexArray[vjId] == tid) {
                    int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexEdgeMatrix + columns * localVjId, vertexEdgeSizes[relativeVjId], viId);
                    if (pos < 0) {
                        insertElement(vertexEdgeMatrix + columns * localVjId, vertexEdgeSizes[relativeVjId], -pos-1, viId);
                        atomicAdd(&vertexEdgeSizes[relativeVjId], 1);
                    }
                }
            }
        }
    }

    // free up the memory usage
    delete[] vertexEdgeMatrix;

    // int testVertexId = 3873;
    // if (indexArray[testVertexId] == tid) {
    //     printf("vertex %d is in cluster id %d\n", testVertexId, tid);
    //     for (int i = 0; i < vertexEdgeSizes[testVertexId - vertexIntervals[baseCluster-1]]; ++i) {
    //         printf("%d ", vertexEdgeMatrix[testVertexId - vertexIntervals[tid-1]][i]);
    //     }
    // }

    // for (int i = 0; i < 5; ++i) {
    //     printf("vertexEdgeSizes %d: %d\n", i, vertexEdgeSizes[i]);
    // }

    // // fill up the vertex edge array
    // for (int vid = startVertexId; vid < vertexIntervals[tid]; ++vid) {
    //     int localVid = vid - startVertexId;
    //     localVertexEdges[tid][localVid] = new int[vertexEdgeSizes[localVid]];
    //     int k = 0;
    //     for (int i = 0; i < vertexRange; ++i) {
    //         if (vertexEdgeMatrix[localVid][i])
    //             localVertexEdges[tid][localVid][k++] = (i + minVertexId);
    //     }
    // }

    /**
     * divide the job by vertices
    int vertexId = startVertexId + tid;
    int minVertexId = connectivity[offsets[externalCells[0]]];
    int maxVertexId = 0; 
    vertexEdgeSizes[tid] = 0;
    // get the maximum vertex id
    int startCellId = (tid == 0 ? 0 : cellIntervals[tid-1]);
    for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        int lastVertexId = connectivity[offsets[cid] + dimension];
        if (lastVertexId > maxVertexId) maxVertexId = lastVertexId;
    }

    // loop through the internal cell list
    int vertexVec[maxVertexId - minimumVid + 1];
    for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        int v0 = connectivity[offsets[cid]];
        if (v0 == )
    }
    */

}
