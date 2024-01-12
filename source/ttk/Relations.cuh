#pragma once

#include "Utils.cuh"
#include <vtkType.h>

const int matrixColumns = 30;


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


__global__ void precomputeVertexEdges(int *vertexEdgeMatrix, int *vertexEdgeSizes, const int baseCluster,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *indexArray, const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    
    // const int method = 1;

    // if (method == 1) 
    {
        // each block thread computes one cluster
        int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;

        /*
        * divide the job by clusters
        */
        // determine the range of vertices
        const int startVertexId = vertexIntervals[tid-1];
        const int startCellId = cellIntervals[tid-1];
        const int localVertexNum = vertexIntervals[tid]-startVertexId;
        // printf("localVertexNum: %d\n", localVertexNum);
        // printf("vertexEdgeSizes[0]: %p\n", vertexEdgeSizes[0]);
        
        for (int i = 0; i < localVertexNum; ++i) 
            vertexEdgeSizes[vertexIntervals[tid-1] - vertexIntervals[baseCluster-1] + i] = 0;

        // loop through the internal cell list
        for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
            for (int vi = 0; vi < dimension; ++vi) {
                int viId = connectivity[offsets[cid] + vi];
                // int localViId = viId - vertexIntervals[tid-1];
                for (int vj = vi+1; vj <= dimension; ++vj) {
                    int vjId = connectivity[offsets[cid] + vj];
                    // int localVjId = vjId - vertexIntervals[tid-1];
                    if (indexArray[viId] == tid) {
                        int relativeViId = viId - vertexIntervals[baseCluster-1];
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], vjId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], -pos-1, vjId);
                            atomicAdd(&vertexEdgeSizes[relativeViId], 1);
                        }
                    } 
                    if (indexArray[vjId] == tid) {
                        int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], viId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], -pos-1, viId);
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
                // int localViId = viId - vertexIntervals[tid-1];
                for (int vj = vi + 1; vj <= dimension; ++vj) {
                    int vjId = connectivity[cellId + vj];
                    // int localVjId = vjId - vertexIntervals[tid-1];
                    if (indexArray[viId] == tid) {
                        int relativeViId = viId - vertexIntervals[baseCluster-1];
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], vjId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], -pos-1, vjId);
                            atomicAdd(&vertexEdgeSizes[relativeViId], 1);
                        }
                    } 
                    if (indexArray[vjId] == tid) {
                        int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], viId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], -pos-1, viId);
                            atomicAdd(&vertexEdgeSizes[relativeVjId], 1);
                        }
                    }
                }
            }
        }
    }

    /*
    else if (method == 2) {
        // all threads in the block computes one cluster

        __shared__ int cellNum;
        __shared__ int externalCellNum;
        __shared__ int cellDivisionSize;
        __shared__ int externalCellDivisionSize;
        __shared__ int vertexDivisionSize;

        const int clusterId = blockIdx.x + baseCluster;
        const int vertexNum = vertexIntervals[clusterId] - vertexIntervals[clusterId - 1];
        
        // the first thread initialize the variables 
        if (threadIdx.x == 0) {
            cellNum = cellIntervals[clusterId] - cellIntervals[clusterId-1];
            externalCellNum = externalCellOffsets[clusterId] - externalCellOffsets[clusterId - 1];
            cellDivisionSize = cellNum / blockDim.x;
            externalCellDivisionSize = externalCellNum / blockDim.x;
            vertexDivisionSize = vertexNum / blockDim.x;
        }

        for (int i = threadIdx.x; i < vertexNum; i += blockDim.x) 
            vertexEdgeSizes[vertexIntervals[clusterId-1] - vertexIntervals[baseCluster-1] + i] = 0;

        __syncthreads();

        const int startCellId = cellIntervals[clusterId-1] + cellDivisionSize * threadIdx.x;
        const int startExternalIndex = externalCellOffsets[clusterId-1] + externalCellDivisionSize * threadIdx.x;
        const int endCellId = (threadIdx.x == blockDim.x - 1 ? cellIntervals[clusterId] : startCellId + cellDivisionSize);
        const int endExternalIndex = (threadIdx.x == blockDim.x - 1 ? externalCellOffsets[clusterId] : startExternalIndex + externalCellDivisionSize);
        // const int startVertexId = vertexIntervals[clusterId-1] + vertexDivisionSize * threadIdx.x;
        // const int endVertexId = (threadIdx.x == blockDim.x - 1 ? vertexIntervals[clusterId] : startVertexId + vertexDivisionSize);

        printf("threadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
        
        // printf("localVertexNum: %d\n", localVertexNum);
        // printf("vertexEdgeSizes[0]: %p\n", vertexEdgeSizes[0]);

        // loop through the internal cell list
        for (int cid = startCellId; cid < endCellId; ++cid) {
            for (int vi = 0; vi < dimension; ++vi) {
                int viId = connectivity[offsets[cid] + vi];
                // int localViId = viId - vertexIntervals[clusterId-1];
                for (int vj = vi+1; vj <= dimension; ++vj) {
                    int vjId = connectivity[offsets[cid] + vj];
                    // int localVjId = vjId - vertexIntervals[clusterId-1];
                    if (indexArray[viId] == clusterId) {
                        int relativeViId = viId - vertexIntervals[baseCluster-1];
                        __syncthreads();
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], vjId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], -pos-1, vjId);
                            atomicAdd(&vertexEdgeSizes[relativeViId], 1);
                        }
                    }
                    if (indexArray[vjId] == clusterId) {
                        int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                        __syncthreads();
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], viId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], -pos-1, viId);
                            atomicAdd(&vertexEdgeSizes[relativeVjId], 1);
                        }
                    }
                }
            }
        }

        // loop through the external cell list
        for (int i = startExternalIndex; i < endExternalIndex; ++i) {
            int cellId = offsets[externalCells[i]];
            for (int vi = 0; vi < dimension; ++vi) {
                int viId = connectivity[cellId + vi];
                // int localViId = viId - vertexIntervals[clusterId-1];
                for (int vj = vi + 1; vj <= dimension; ++vj) {
                    int vjId = connectivity[cellId + vj];
                    // int localVjId = vjId - vertexIntervals[clusterId-1];
                    if (indexArray[viId] == clusterId) {
                        int relativeViId = viId - vertexIntervals[baseCluster-1];
                        __syncthreads();
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], vjId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], -pos-1, vjId);
                            atomicAdd(&vertexEdgeSizes[relativeViId], 1);
                        }
                    }
                    if (indexArray[vjId] == clusterId) {
                        int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                        __syncthreads();
                        int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], viId);
                        if (pos < 0) {
                            insertElement(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], -pos-1, viId);
                            atomicAdd(&vertexEdgeSizes[relativeVjId], 1);
                        }
                    }
                }
            }
        }
    }

    // int testVertexId = 3873;
    // if (indexArray[testVertexId] == tid) {
    //     printf("vertex %d is in cluster id %d\n", testVertexId, tid);
    //     for (int i = 0; i < vertexEdgeSizes[testVertexId - vertexIntervals[baseCluster-1]]; ++i) {
    //         printf("%d ", vertexEdgeMatrix[testVertexId - vertexIntervals[tid-1]][i]);
    //     }
    // }
    */

}

__global__ void precomputeVertexNeighbors(int *vertexNeighborMatrix, int *vertexNeighborSizes, 
            const int baseCluster, const int clusterNumber,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    // each block thread computes one cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    // determine the range of vertices
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startCellId = cellIntervals[tid-1];
    const int localVertexNum = vertexIntervals[tid]-startVertexId;
    
    for (int i = 0; i < localVertexNum; ++i) 
        vertexNeighborSizes[vertexIntervals[tid-1] - vertexIntervals[baseCluster-1] + i] = 0;

    // loop through the internal cell list
    for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[offsets[cid] + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], vjId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], -pos-1, vjId);
                        atomicAdd(&vertexNeighborSizes[relativeViId], 1);
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], viId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], -pos-1, viId);
                        atomicAdd(&vertexNeighborSizes[relativeVjId], 1);
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
            // int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                // int localVjId = vjId - vertexIntervals[tid-1];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], vjId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], -pos-1, vjId);
                        atomicAdd(&vertexNeighborSizes[relativeViId], 1);
                    }
                } 
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - vertexIntervals[baseCluster-1];
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], viId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], -pos-1, viId);
                        atomicAdd(&vertexNeighborSizes[relativeVjId], 1);
                    }
                }
            }
        }
    }
}


__global__ void precomputeVertexStars(int *vertexStarMatrix, int *vertexStarSizes, 
            const int baseCluster, const int clusterNumber,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *indexArray, const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    // each block thread computes one cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    // determine the range of vertices
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startCellId = cellIntervals[tid-1];
    const int localVertexNum = vertexIntervals[tid]-startVertexId;
    
    for (int i = 0; i < localVertexNum; ++i) 
        vertexStarSizes[vertexIntervals[tid-1] - vertexIntervals[baseCluster-1] + i] = 0;

    // loop through the internal cell list
    for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        // the first vertex belongs to the cluster
        for (int vi = 0; vi <= dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            if (viId < endVertexId) {
                int relativeViId = viId - vertexIntervals[baseCluster-1];
                vertexStarMatrix[matrixColumns * relativeViId + vertexStarSizes[relativeViId]] = cid;
                atomicAdd(&vertexStarSizes[relativeViId], 1);
            }
        }
    }

    // loop through the external cell list
    for (int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 1; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
                // int localVjId = vjId - vertexIntervals[tid-1];
            if (viId >= startVertexId && viId < endVertexId) {
                int relativeViId = viId - vertexIntervals[baseCluster-1];
                vertexStarMatrix[matrixColumns * relativeViId + vertexStarSizes[relativeViId]] = externalCells[i];
                atomicAdd(&vertexStarSizes[relativeViId], 1);
            }
        }
    }
}


__global__ void precomputeMultiVertexNeighbors(int *vertexNeighborMatrix, int *vertexNeighborSizes, 
            const int clusterNumber, const int *baseClusters, const int *vertexOffsets,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    // each block thread computes one cluster
    // (blockIdx.x, blockIdx.y) -> dim3 gpuBlock(threadNum, numBlocks/threadNum)
    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = blockIdx.y * blockDim.x + threadIdx.x + baseCluster;
    // printf("tid: %d, baseCluster: %d, vertexOffset: %d\n", tid, baseCluster, vertexOffset);
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    // determine the range of vertices
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startCellId = cellIntervals[tid-1];
    const int localVertexNum = vertexIntervals[tid]-startVertexId;
    
    for (int i = 0; i < localVertexNum; ++i) 
        vertexNeighborSizes[vertexIntervals[tid-1] - vertexIntervals[baseCluster-1] + i + vertexOffset] = 0;

    // loop through the internal cell list
    for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[offsets[cid] + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - vertexIntervals[baseCluster-1] + vertexOffset;
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], vjId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], -pos-1, vjId);
                        atomicAdd(&vertexNeighborSizes[relativeViId], 1);
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - vertexIntervals[baseCluster-1] + vertexOffset;
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], viId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], -pos-1, viId);
                        atomicAdd(&vertexNeighborSizes[relativeVjId], 1);
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
            // int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                // int localVjId = vjId - vertexIntervals[tid-1];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - vertexIntervals[baseCluster-1] + vertexOffset;
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], vjId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeViId, vertexNeighborSizes[relativeViId], -pos-1, vjId);
                        atomicAdd(&vertexNeighborSizes[relativeViId], 1);
                    }
                } 
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - vertexIntervals[baseCluster-1] + vertexOffset;
                    int pos = binarySearch(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], viId);
                    if (pos < 0) {
                        insertElement(vertexNeighborMatrix + matrixColumns * relativeVjId, vertexNeighborSizes[relativeVjId], -pos-1, viId);
                        atomicAdd(&vertexNeighborSizes[relativeVjId], 1);
                    }
                }
            }
        }
    }
}


__global__ void precomputeMultiVertexStars(int *vertexStarMatrix, int *vertexStarSizes, 
            const int clusterNumber, const int *baseClusters, const int *vertexOffsets,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    // each block thread computes one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = blockIdx.y * blockDim.x + threadIdx.x + baseCluster;
    // int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    // determine the range of vertices
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startCellId = cellIntervals[tid-1];
    const int localVertexNum = vertexIntervals[tid]-startVertexId;
    
    for (int i = 0; i < localVertexNum; ++i) 
        vertexStarSizes[vertexIntervals[tid-1] - vertexIntervals[baseCluster-1] + i + vertexOffset] = 0;

    // loop through the internal cell list
    for (int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        // the first vertex belongs to the cluster
        for (int vi = 0; vi <= dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            if (viId < endVertexId) {
                int relativeViId = viId - vertexIntervals[baseCluster-1] + vertexOffset;
                vertexStarMatrix[matrixColumns * relativeViId + vertexStarSizes[relativeViId]] = cid;
                atomicAdd(&vertexStarSizes[relativeViId], 1);
            }
        }
    }

    // loop through the external cell list
    for (int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 1; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
                // int localVjId = vjId - vertexIntervals[tid-1];
            if (viId >= startVertexId && viId < endVertexId) {
                int relativeViId = viId - vertexIntervals[baseCluster-1] + vertexOffset;
                vertexStarMatrix[matrixColumns * relativeViId + vertexStarSizes[relativeViId]] = externalCells[i];
                atomicAdd(&vertexStarSizes[relativeViId], 1);
            }
        }
    }
}
