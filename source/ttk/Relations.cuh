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


/* NOT USED: Build the edge list */
__global__ void buildEdges(int *edgeList, int *edgeCounts,
                            const int *vertexMap, const int *cellIntervals, 
                            const int *vertexIntervals, const int *edgeIntervals, 
                            const int *externalCells, const int *externalCellOffsets, 
                            const vtkIdType *connectivity, const vtkIdType *offsets, 
                            const int clusterNumber, const int cellNumber, const int dimension) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (tid >= clusterNumber) return;

    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startCellId = cellIntervals[tid-1];
    const int startEdgeIdx = edgeIntervals[tid-1];

    // printf("tid: %d, startEdgeIdx: %d, startCellId: %d\n", tid, startEdgeIdx, startCellId);

    // loop through the internal cell list
    for(int cid = startCellId; cid < cellIntervals[tid]; ++cid) {
        int edgeIds[2];

        // loop through each edge of the cell
        for(int j = 0; j < dimension; ++j) {
            edgeIds[0] = connectivity[offsets[cid] + j];
            // the edge does not belong to the current node
            if(edgeIds[0] < endVertexId) {
                for(int k = j + 1; k <= dimension; ++k) {
                    edgeIds[1] = connectivity[offsets[cid] + k];

                    // not found in the cluster edge list - increase the edge count
                    int pos = binarySearch(edgeList, startEdgeIdx, edgeCounts[tid], edgeIds);
                    // printf("tid: %d, edgeId: (%d, %d), pos:%d\n", tid, edgeIds[0], edgeIds[1], pos);
                    if(pos < 0) {
                        insertElement(edgeList, startEdgeIdx, edgeCounts[tid], -pos-1, edgeIds);
                        edgeCounts[tid] += 1;
                    }
                }
            }
        }
    }

    // loop through the external cell list
    for(int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        int edgeIds[2];

        // loop through each edge of the cell
        for(int j = 0; j < dimension; j++) {
            for(int k = j + 1; k <= dimension; k++) {
                edgeIds[0] = connectivity[cellId + j];
                edgeIds[1] = connectivity[cellId + k];

                // the edge is in the current cluster
                if(edgeIds[0] >= startVertexId && edgeIds[0] < endVertexId) {
                    int pos = binarySearch(edgeList, startEdgeIdx, edgeCounts[tid], edgeIds);
                    if (pos < 0) {
                        insertElement(edgeList, startEdgeIdx, edgeCounts[tid], -pos-1, edgeIds);
                        edgeCounts[tid] += 1;
                    }
                }
            }
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


__global__ void precomputeVertexEdges(int *edgeList, int *vertexEdgeMatrix, int *vertexEdgeSizes, 
            const int baseCluster, const int clusterNumber,
            const vtkIdType *connectivity, const vtkIdType *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *edgeIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, const int dimension) {

    // each block thread computes one cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster - 1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];
    // const int localVertexNum = vertexIntervals[tid]-startVertexId;
    
    // for (int i = 0; i < localVertexNum; ++i) 
    //     vertexEdgeSizes[vertexIntervals[tid-1] - vertexIntervals[baseCluster-1] + i] = 0;

    // internel edges can be handled from the edge list
    for (int edgeId = startEdgeId; edgeId < endEdgeId; ++edgeId) {
        int relativeViId = edgeList[edgeId * 2] - baseVertexId;
        // no need to check duplication as the edge id is unique
        insertElement(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], vertexEdgeSizes[relativeViId], edgeId);
        vertexEdgeSizes[relativeViId] += 1;
        if (edgeList[edgeId * 2 + 1] < endVertexId) {
            int relativeVjId = edgeList[edgeId * 2 + 1] - baseVertexId;
            insertElement(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], vertexEdgeSizes[relativeVjId], edgeId);
            vertexEdgeSizes[relativeVjId] += 1;
        }
    }

    // only external edges need to be checked
    for (int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            // int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                // // already checked from the internal edge list above
                // if (viId >= startVertexId && viId < endVertexId) {
                //     int relativeViId = viId - vertexIntervals[baseCluster-1];
                //     int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], edgeId);
                //     if (pos < 0) {
                //         insertElement(vertexEdgeMatrix + matrixColumns * relativeViId, vertexEdgeSizes[relativeViId], -pos-1, edgeId);
                //         vertexEdgeSizes[relativeViId] += 1;
                //     }
                // } 
                if (viId < startVertexId && vjId >= startVertexId && vjId < endVertexId) {
                    int exCluster = indexArray[viId];
                    int newEdge[2] = {viId, vjId};
                    int relativeVjId = vjId - baseVertexId;
                    int edgeId = binarySearch(edgeList, edgeIntervals[exCluster-1], edgeIntervals[exCluster]-edgeIntervals[exCluster-1], newEdge);
                    int pos = binarySearch(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], edgeId);
                    if (pos < 0) {
                        insertElement(vertexEdgeMatrix + matrixColumns * relativeVjId, vertexEdgeSizes[relativeVjId], -pos-1, edgeId);
                        vertexEdgeSizes[relativeVjId] += 1;
                    }
                }
            }
        }
    }
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
