#pragma once

#include "Utils.cuh"
#include <vtkType.h>

const int matrixColumnSmall = 30;
const int matrixColumnLarge = 80;


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
        vtkIdType edgeIds[2];

        // loop through each edge of the cell
        for(int j = 0; j < dimension; ++j) {
            edgeIds[0] = connectivity[offsets[cid] + j];
            // the edge does not belong to the current node
            if(edgeIds[0] < endVertexId) {
                for(int k = j + 1; k <= dimension; ++k) {
                    edgeIds[1] = connectivity[offsets[cid] + k];

                    // not found in the cluster edge list - increase the edge count
                    int pos = binarySearch<2>(edgeList, startEdgeIdx, startEdgeIdx + edgeCounts[tid], edgeIds);
                    // printf("tid: %d, edgeId: (%d, %d), pos:%d\n", tid, edgeIds[0], edgeIds[1], pos);
                    if(pos < 0) {
                        insertElement<2>(edgeList, startEdgeIdx, edgeCounts[tid], -pos-1, edgeIds);
                        edgeCounts[tid] += 1;
                    }
                }
            }
        }
    }

    // loop through the external cell list
    for(int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        vtkIdType edgeIds[2];

        // loop through each edge of the cell
        for(int j = 0; j < dimension; j++) {
            for(int k = j + 1; k <= dimension; k++) {
                edgeIds[0] = connectivity[cellId + j];
                edgeIds[1] = connectivity[cellId + k];

                // the edge is in the current cluster
                if(edgeIds[0] >= startVertexId && edgeIds[0] < endVertexId) {
                    int pos = binarySearch<2>(edgeList, startEdgeIdx, startEdgeIdx + edgeCounts[tid], edgeIds);
                    if (pos < 0) {
                        insertElement<2>(edgeList, startEdgeIdx, edgeCounts[tid], -pos-1, edgeIds);
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
    
    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster - 1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];

    // only external edges need to be checked for duplication
    for (int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension; ++vi) {
            vtkIdType newEdge[2] = {connectivity[cellId + vi]};
            // int localViId = viId - vertexIntervals[tid-1];
            if (newEdge[0] < startVertexId) {
                int exCluster = indexArray[newEdge[0]];
                for (int vj = vi + 1; vj <= dimension; ++vj) {
                    newEdge[1] = connectivity[cellId + vj];
                    // // already included in the internal edge list
                    // if (viId >= startVertexId && viId < endVertexId) {
                    //     int relativeViId = viId - vertexIntervals[baseCluster-1];
                    //     int pos = binarySearch(vertexEdgeMatrix + matrixColumnSmall * relativeViId, vertexEdgeSizes[relativeViId], edgeId);
                    //     if (pos < 0) {
                    //         insertElement(vertexEdgeMatrix + matrixColumnSmall * relativeViId, vertexEdgeSizes[relativeViId], -pos-1, edgeId);
                    //         vertexEdgeSizes[relativeViId] += 1;
                    //     }
                    // } 
                    if (newEdge[1] >= startVertexId && newEdge[1] < endVertexId) {
                        int relativeVertexId = newEdge[1] - baseVertexId;
                        int edgeId = binarySearch<2>(edgeList, edgeIntervals[exCluster-1], edgeIntervals[exCluster], newEdge);
                        int pos = binarySearch1D(vertexEdgeMatrix + matrixColumnSmall * relativeVertexId, vertexEdgeSizes[relativeVertexId], edgeId);
                        if (pos < 0) {
                            insertElement1D(vertexEdgeMatrix + matrixColumnSmall * relativeVertexId, vertexEdgeSizes[relativeVertexId], -pos-1, edgeId);
                            vertexEdgeSizes[relativeVertexId] += 1;
                        }
                    }
                }
            }
        }
    }
    
    // internel edges can be handled from the edge list
    for (int edgeId = startEdgeId; edgeId < endEdgeId; ++edgeId) {
        int relativeViId = edgeList[edgeId * 2] - baseVertexId;
        // no need to check duplication as the edge id is unique
        insertElement1D(vertexEdgeMatrix + matrixColumnSmall * relativeViId, vertexEdgeSizes[relativeViId], vertexEdgeSizes[relativeViId], edgeId);
        vertexEdgeSizes[relativeViId] += 1;
        if (edgeList[edgeId * 2 + 1] < endVertexId) {
            int relativeVjId = edgeList[edgeId * 2 + 1] - baseVertexId;
            insertElement1D(vertexEdgeMatrix + matrixColumnSmall * relativeVjId, vertexEdgeSizes[relativeVjId], vertexEdgeSizes[relativeVjId], edgeId);
            vertexEdgeSizes[relativeVjId] += 1;
        }
    }
}


// __global__ void precomputeVertexNeighbors(int *vertexNeighborMatrix, int *vertexNeighborSizes, 
//             const int baseCluster, const int clusterNumber,
//             const vtkIdType *connectivity, const vtkIdType *offsets, 
//             const int *vertexIntervals, const int *cellIntervals, 
//             const int *externalCells, const int *externalCellOffsets, const int dimension) {
//     // each block thread computes one cluster
//     int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
//     if (tid >= clusterNumber) return;

//     /*
//     * divide the job by clusters
//     */
//     // determine the range of vertices
//     const int baseVertexId = vertexIntervals[baseCluster-1];
//     const int startVertexId = vertexIntervals[tid-1];
//     const int endVertexId = vertexIntervals[tid];
//     const int startCellId = cellIntervals[tid-1];
//     const int endCellId = cellIntervals[tid];
//     const int startExCellId = externalCellOffsets[tid-1];
//     const int endExCellId = externalCellOffsets[tid];

//     // loop through the external cell list
//     for (int i = startExCellId; i < endExCellId; ++i) {
//         int cellId = offsets[externalCells[i]];
//         for (int vi = 0; vi < dimension; ++vi) {
//             int viId = connectivity[cellId + vi];
//             // int localViId = viId - vertexIntervals[tid-1];
//             for (int vj = vi + 1; vj <= dimension; ++vj) {
//                 int vjId = connectivity[cellId + vj];
//                 // int localVjId = vjId - vertexIntervals[tid-1];
//                 if (viId >= startVertexId && viId < endVertexId) {
//                     int relativeViId = viId - baseVertexId;

//                     // // hash function version to keep the list unique (not sorted)
//                     // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeViId;
//                     // int pos = getHashPosition(list, matrixColumnSmall, vjId);
//                     // if (list[pos] == -1) {
//                     //     list[pos] = vjId;
//                     //     vertexNeighborSizes[relativeViId] += 1;
//                     // }

//                     // binary search version to keep the list sorted and unique
//                     int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, vertexNeighborSizes[relativeViId], vjId);
//                     if (pos < 0) {
//                         insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, vertexNeighborSizes[relativeViId], -pos-1, vjId);
//                         vertexNeighborSizes[relativeViId] += 1;
//                     }
//                 }
//                 if (vjId >= startVertexId && vjId < endVertexId) {
//                     int relativeVjId = vjId - baseVertexId;

//                     // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeVjId;
//                     // int pos = getHashPosition(list, matrixColumnSmall, viId);
//                     // if (list[pos] == -1) {
//                     //     list[pos] = viId;
//                     //     vertexNeighborSizes[relativeVjId] += 1;
//                     // }

//                     int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, vertexNeighborSizes[relativeVjId], viId);
//                     if (pos < 0) {
//                         insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, vertexNeighborSizes[relativeVjId], -pos-1, viId);
//                         vertexNeighborSizes[relativeVjId] += 1;
//                     }
//                 }
//             }
//         }
//     }

//     // loop through the internal cell list
//     for (int cid = startCellId; cid < endCellId; ++cid) {
//         int offset = offsets[cid];
//         for (int vi = 0; vi < dimension; ++vi) {
//             int viId = connectivity[offset + vi];
//             for (int vj = vi+1; vj <= dimension; ++vj) {
//                 int vjId = connectivity[offset + vj];
//                 if (viId >= startVertexId && viId < endVertexId) {
//                     int relativeViId = viId - baseVertexId;

//                     // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeViId;
//                     // int pos = getHashPosition(list, matrixColumnSmall, vjId);
//                     // if (list[pos] == -1) {
//                     //     list[pos] = vjId;
//                     //     vertexNeighborSizes[relativeViId] += 1;
//                     // }

//                     int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, vertexNeighborSizes[relativeViId], vjId);
//                     if (pos < 0) {
//                         insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, vertexNeighborSizes[relativeViId], -pos-1, vjId);
//                         vertexNeighborSizes[relativeViId] += 1;
//                     }
//                 }
//                 if (vjId >= startVertexId && vjId < endVertexId) {
//                     int relativeVjId = vjId - baseVertexId;

//                     // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeVjId;
//                     // int pos = getHashPosition(list, matrixColumnSmall, viId);
//                     // if (list[pos] == -1) {
//                     //     list[pos] = viId;
//                     //     vertexNeighborSizes[relativeVjId] += 1;
//                     // }

//                     int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, vertexNeighborSizes[relativeVjId], viId);
//                     if (pos < 0) {
//                         insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, vertexNeighborSizes[relativeVjId], -pos-1, viId);
//                         vertexNeighborSizes[relativeVjId] += 1;
//                     }
//                 }
//             }
//         }
//     }
// }

__global__ void precomputeVertexNeighbors(int *vertexNeighborMatrix, int *vertexNeighborSizes, 
            const int baseCluster, const int clusterNumber,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    
    // use `short` to avoid the memory overflow
    extern __shared__ short sVertexNeighborSizes[];
    
    // each block thread computes one cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster-1];
    const int blockVertexId = vertexIntervals[blockIdx.x * blockDim.x + baseCluster - 1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];

    // printf("tid: %d, base vertex id: %d, block vertex id: %d, start vertex: %d\n", tid, baseVertexId, blockVertexId, startVertexId);

    // initialize shared array to 0 (a thread block shares the space)
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        sVertexNeighborSizes[vid-blockVertexId] = 0;
    }
    // __syncthreads();


    const int startCellId = cellIntervals[tid-1];
    const int endCellId = cellIntervals[tid];
    const int startExCellId = externalCellOffsets[tid-1];
    const int endExCellId = externalCellOffsets[tid];

    // loop through the external cell list
    for (int i = startExCellId; i < endExCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - baseVertexId;
                    // // hash function version to keep the list unique (not sorted)
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeViId;
                    // int pos = getHashPosition(list, matrixColumnSmall, vjId);
                    // if (list[pos] == -1) {
                    //     list[pos] = vjId;
                    //     sVertexNeighborSizes[viId-blockVertexId] += 1;
                    // }

                    // binary search version to keep the list sorted and unique
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId-blockVertexId], vjId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId-blockVertexId], -pos-1, vjId);
                        sVertexNeighborSizes[viId-blockVertexId] += 1;
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - baseVertexId;
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeVjId;
                    // int pos = getHashPosition(list, matrixColumnSmall, viId);
                    // if (list[pos] == -1) {
                    //     list[pos] = viId;
                    //     sVertexNeighborSizes[relativeVjId] += 1;
                    // }
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId-blockVertexId], viId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId-blockVertexId], -pos-1, viId);
                        sVertexNeighborSizes[vjId-blockVertexId] += 1;
                    }
                }
            }
        }
    }

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int offset = offsets[cid];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[offset + vi];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[offset + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - baseVertexId;
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeViId;
                    // int pos = getHashPosition(list, matrixColumnSmall, vjId);
                    // if (list[pos] == -1) {
                    //     list[pos] = vjId;
                    //     sVertexNeighborSizes[relativeViId] += 1;
                    // }
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId-blockVertexId], vjId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId-blockVertexId], -pos-1, vjId);
                        sVertexNeighborSizes[viId-blockVertexId] += 1;
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - baseVertexId;
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeVjId;
                    // int pos = getHashPosition(list, matrixColumnSmall, viId);
                    // if (list[pos] == -1) {
                    //     list[pos] = viId;
                    //     sVertexNeighborSizes[relativeVjId] += 1;
                    // }
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId-blockVertexId], viId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId-blockVertexId], -pos-1, viId);
                        sVertexNeighborSizes[vjId-blockVertexId] += 1;
                    }
                }
            }
        }
    }

    // copy the shared array to the global memory 
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        vertexNeighborSizes[vid - baseVertexId] = sVertexNeighborSizes[vid - blockVertexId];
    }

}


__global__ void precomputeVertexStars(int *vertexStarMatrix, int *vertexStarSizes, 
            const int baseCluster, const int clusterNumber,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    
    extern __shared__ short sVertexStarSizes[];
    // each block thread computes one cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster-1];
    const int blockVertexId = vertexIntervals[blockIdx.x * blockDim.x + baseCluster - 1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];

    // initialize the shared array
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        sVertexStarSizes[vid - blockVertexId] = 0;
    }

    const int startExCellId = externalCellOffsets[tid-1];
    const int endExCellId = externalCellOffsets[tid];
    // loop through the external cell list
    for (int i = startExCellId; i < endExCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 1; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
                // int localVjId = vjId - vertexIntervals[tid-1];
            if (viId >= startVertexId && viId < endVertexId) {
                int relativeViId = viId - baseVertexId;
                vertexStarMatrix[matrixColumnSmall * relativeViId + sVertexStarSizes[viId - blockVertexId]] = externalCells[i];
                sVertexStarSizes[viId - blockVertexId] += 1;
            }
        }
    }

    const int startCellId = cellIntervals[tid-1];
    const int endCellId = cellIntervals[tid];

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        // the first vertex belongs to the cluster
        for (int vi = 0; vi <= dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            if (viId < endVertexId) {
                int relativeViId = viId - baseVertexId;
                vertexStarMatrix[matrixColumnSmall * relativeViId + sVertexStarSizes[viId - blockVertexId]] = cid;
                sVertexStarSizes[viId - blockVertexId] += 1;
            }
        }
    }
    
    // copy the shared array to the global memory 
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        vertexStarSizes[vid - baseVertexId] = sVertexStarSizes[vid - blockVertexId];
    }
}


__global__ void precomputeVertexTriangles(int *triangleList, int *vertexTriangleMatrix, int *vertexTriangleSizes, 
            const int baseCluster, const int clusterNumber,
            const vtkIdType *connectivity, const vtkIdType *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, const int dimension) {

    // each block thread computes one cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;
    
    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster - 1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int startTriangleId = triangleIntervals[tid-1];
    const int endTriangleId = triangleIntervals[tid];

    // only external triangles need to be checked for duplication
    for (int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            vtkIdType newTriangle[3] = {connectivity[cellId + vi]};
            // int localViId = viId - vertexIntervals[tid-1];
            if (newTriangle[0] < startVertexId) {
                int exCluster = indexArray[newTriangle[0]];
                for (int vj = vi + 1; vj < dimension; ++vj) {
                    newTriangle[1] = connectivity[cellId + vj];
                    for (int vk = vj + 1; vk <= dimension; ++vk) {
                        newTriangle[2] = connectivity[cellId + vk];
                        // the second vertex belongs to the cluster
                        if (newTriangle[1] >= startVertexId && newTriangle[1] < endVertexId) {
                            int relativeVertexId = newTriangle[1] - baseVertexId;
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);
                            int pos = binarySearch1D(vertexTriangleMatrix + matrixColumnLarge * relativeVertexId, vertexTriangleSizes[relativeVertexId], triangleId);
                            if (pos < 0) {
                                insertElement1D(vertexTriangleMatrix + matrixColumnLarge * relativeVertexId, vertexTriangleSizes[relativeVertexId], -pos-1, triangleId);
                                vertexTriangleSizes[relativeVertexId] += 1;
                            }
                        }
                        // the last vertex belongs to the cluster
                        else if (newTriangle[2] >= startVertexId && newTriangle[2] < endVertexId) {
                            int relativeVertexId = newTriangle[2] - baseVertexId;
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);
                            int pos = binarySearch1D(vertexTriangleMatrix + matrixColumnLarge * relativeVertexId, vertexTriangleSizes[relativeVertexId], triangleId);
                            if (pos < 0) {
                                insertElement1D(vertexTriangleMatrix + matrixColumnLarge * relativeVertexId, vertexTriangleSizes[relativeVertexId], -pos-1, triangleId);
                                vertexTriangleSizes[relativeVertexId] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // internel triangles can be handled from the edge list
    for (int triangleId = startTriangleId; triangleId < endTriangleId; ++triangleId) {
        int relativeViId = triangleList[triangleId * 3] - baseVertexId;
        // no need to check duplication as the edge id is unique
        insertElement1D(vertexTriangleMatrix + matrixColumnLarge * relativeViId, vertexTriangleSizes[relativeViId], vertexTriangleSizes[relativeViId], triangleId);
        vertexTriangleSizes[relativeViId] += 1;
        for (int i = 1; i < dimension; ++i) {
            if (triangleList[triangleId * 3 + i] < endVertexId) {
                int relativeVjId = triangleList[triangleId * 3 + i] - baseVertexId;
                insertElement1D(vertexTriangleMatrix + matrixColumnLarge * relativeVjId, vertexTriangleSizes[relativeVjId], vertexTriangleSizes[relativeVjId], triangleId);
                vertexTriangleSizes[relativeVjId] += 1;
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
    
    extern __shared__ short sVertexNeighborSizes[];

    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = blockIdx.y * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;
    // printf("blockIdx.x: %d, blockIdx.y: %d, tid: %d, baseCluster: %d, vertexOffset: %d\n", blockIdx.x, blockIdx.y, tid, baseCluster, vertexOffset);

    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster-1];
    const int blockVertexId = vertexIntervals[blockIdx.y * blockDim.x + baseCluster - 1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];

    // initialize the shared array
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        sVertexNeighborSizes[vid - blockVertexId] = 0;
    }

    const int startExCellId = externalCellOffsets[tid-1];
    const int endExCellId = externalCellOffsets[tid];

    // loop through the external cell list
    for (int i = startExCellId; i < endExCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            // int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                // int localVjId = vjId - vertexIntervals[tid-1];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - baseVertexId + vertexOffset;
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeViId;
                    // int pos = getHashPosition(list, matrixColumnSmall, vjId);
                    // if (list[pos] == -1) {
                    //     list[pos] = vjId;
                    //     sVertexNeighborSizes[viId - blockVertexId] += 1;
                    // }
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId - blockVertexId], vjId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId - blockVertexId], -pos-1, vjId);
                        sVertexNeighborSizes[viId - blockVertexId] += 1;
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - baseVertexId + vertexOffset;
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeVjId;
                    // int pos = getHashPosition(list, matrixColumnSmall, viId);
                    // if (list[pos] == -1) {
                    //     list[pos] = viId;
                    //     sVertexNeighborSizes[vjId - blockVertexId] += 1;
                    // }
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId - blockVertexId], viId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId - blockVertexId], -pos-1, viId);
                        sVertexNeighborSizes[vjId - blockVertexId] += 1;
                    }
                }
            }
        }
    }

    const int startCellId = cellIntervals[tid-1];
    const int endCellId = cellIntervals[tid];

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[offsets[cid] + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - baseVertexId + vertexOffset;
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeViId;
                    // int pos = getHashPosition(list, matrixColumnSmall, vjId);
                    // if (list[pos] == -1) {
                    //     list[pos] = vjId;
                    //     sVertexNeighborSizes[viId - blockVertexId] += 1;
                    // }
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId - blockVertexId], vjId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeViId, sVertexNeighborSizes[viId - blockVertexId], -pos-1, vjId);
                        sVertexNeighborSizes[viId - blockVertexId] += 1;
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVjId = vjId - baseVertexId + vertexOffset;
                    // int *list = vertexNeighborMatrix + matrixColumnSmall * relativeVjId;
                    // int pos = getHashPosition(list, matrixColumnSmall, viId);
                    // if (list[pos] == -1) {
                    //     list[pos] = viId;
                    //     sVertexNeighborSizes[vjId - blockVertexId] += 1;
                    // }
                    int pos = binarySearch1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId - blockVertexId], viId);
                    if (pos < 0) {
                        insertElement1D(vertexNeighborMatrix + matrixColumnSmall * relativeVjId, sVertexNeighborSizes[vjId - blockVertexId], -pos-1, viId);
                        sVertexNeighborSizes[vjId - blockVertexId] += 1;
                    }
                }
            }
        }
    }
    
    // copy the shared array to the global memory 
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        vertexNeighborSizes[vid - baseVertexId + vertexOffset] = sVertexNeighborSizes[vid - blockVertexId];
    }
}


__global__ void precomputeMultiVertexStars(int *vertexStarMatrix, int *vertexStarSizes, 
            const int clusterNumber, const int *baseClusters, const int *vertexOffsets,
            const vtkIdType *connectivity, const vtkIdType *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, const int dimension) {
    // each block thread computes one cluster

    extern __shared__ short sVertexStarSizes[];

    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = blockIdx.y * blockDim.x + threadIdx.x + baseCluster;
    // int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster-1];
    const int blockVertexId = vertexIntervals[blockIdx.y * blockDim.x + baseCluster - 1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];

    // initialize the shared array
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        sVertexStarSizes[vid - blockVertexId] = 0;
    }

    const int startCellId = cellIntervals[tid-1];
    const int endCellId = cellIntervals[tid];
    
    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        // the first vertex belongs to the cluster
        for (int vi = 0; vi <= dimension; ++vi) {
            int viId = connectivity[offsets[cid] + vi];
            if (viId < endVertexId) {
                int relativeViId = viId - baseVertexId + vertexOffset;
                vertexStarMatrix[matrixColumnSmall * relativeViId + sVertexStarSizes[viId - blockVertexId]] = cid;
                sVertexStarSizes[viId - blockVertexId] += 1;
            }
        }
    }

    const int startExCellId = externalCellOffsets[tid-1];
    const int endExCellId = externalCellOffsets[tid];

    // loop through the external cell list
    for (int i = startExCellId; i < endExCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 1; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
                // int localVjId = vjId - vertexIntervals[tid-1];
            if (viId >= startVertexId && viId < endVertexId) {
                int relativeViId = viId - baseVertexId + vertexOffset;
                vertexStarMatrix[matrixColumnSmall * relativeViId + sVertexStarSizes[viId - blockVertexId]] = externalCells[i];
                sVertexStarSizes[viId - blockVertexId] += 1;
            }
        }
    }

    // copy the shared array to the global memory 
    for (int vid = startVertexId; vid < endVertexId; ++vid) {
        vertexStarSizes[vid - baseVertexId + vertexOffset] = sVertexStarSizes[vid - blockVertexId];
    }
}
