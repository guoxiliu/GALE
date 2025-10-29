#pragma once

#include <Utils.cuh>

const int efColumnNum = 20;
const int etColumnNum = 20;
const int ftColumnNum = 5;
const int vvColumnNum = 30;
const int vtColumnNum = 60;
const int veColumnNum = 30;
const int vfColumnNum = 80;

/**
 * Check if a value exists in an array. If not found, insert it atomically.
 * Returns the position of the value in the array.
 * 
 * @param arr - pointer to the array
 * @param size - pointer to the current size of the array (will be atomically updated if insertion occurs)
 * @param value - the value to search for or insert
 * @return the position of the value in the array (either existing position or newly inserted position)
 */
__device__ inline int findOrInsert(int *arr, int *size, int value) {
    // Linear search to check if value already exists
    int localSize = *size;
    for (int k = 0; k < localSize; ++k) {
        if (arr[k] == value) {
            return k;  // Found at position k
        }
    }
    
    // Not found, need to insert the element
    int insertPos = localSize;
    int oldVal = -1;
    
    // Handle the case where multiple threads work with the same location
    // One array index can only be updated by one thread
    do {
        oldVal = atomicCAS(&arr[insertPos], -1, value);
        if (oldVal == -1) {
            atomicAdd(size, 1);
            return insertPos;
        } else if (oldVal == value) {
            return insertPos;
        }
        insertPos++;
    } while (oldVal != -1 && oldVal != value);

    return -1;
}


__global__ void countCells(int *localCellCounts, const int *connectivity, const int *offset, const int *vertexMap, const int cellNumber, const int dimension) {
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


__global__ void countExternalCells(int *externalCellOffsets, const int *connectivity, const int *offset, const int *vertexMap, const int cellNumber, const int dimension) {
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


__global__ void countExternalCells(int *externalCellOffsets, const int *connectivity, const int *offset, const int *cellIntervals, const int *vertexMap, const int clusterNumber, const int dimension) {
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


/**
 * Build the edge list. 
 * NOTE: NOT USED due to poor performance.
*/
__global__ void buildEdges(int *edgeList, int *edgeCounts,
                            const int *vertexMap, const int *cellIntervals, 
                            const int *vertexIntervals, const int *edgeIntervals, 
                            const int *externalCells, const int *externalCellOffsets, 
                            const int *connectivity, const int *offsets, 
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
        int selectedEdge[2];

        // loop through each edge of the cell
        for(int j = 0; j < dimension; ++j) {
            selectedEdge[0] = connectivity[offsets[cid] + j];
            // the edge does not belong to the current node
            if(selectedEdge[0] < endVertexId) {
                for(int k = j + 1; k <= dimension; ++k) {
                    selectedEdge[1] = connectivity[offsets[cid] + k];

                    // not found in the cluster edge list - increase the edge count
                    int pos = binarySearch<2>(edgeList, startEdgeIdx, startEdgeIdx + edgeCounts[tid], selectedEdge);
                    // printf("tid: %d, edgeId: (%d, %d), pos:%d\n", tid, selectedEdge[0], selectedEdge[1], pos);
                    if(pos < 0) {
                        insertElement<2>(edgeList, startEdgeIdx, edgeCounts[tid], -pos-1, selectedEdge);
                        edgeCounts[tid] += 1;
                    }
                }
            }
        }
    }

    // loop through the external cell list
    for(int i = externalCellOffsets[tid-1]; i < externalCellOffsets[tid]; ++i) {
        int cellId = offsets[externalCells[i]];
        int selectedEdge[2];

        // loop through each edge of the cell
        for(int j = 0; j < dimension; j++) {
            for(int k = j + 1; k <= dimension; k++) {
                selectedEdge[0] = connectivity[cellId + j];
                selectedEdge[1] = connectivity[cellId + k];

                // the edge is in the current cluster
                if(selectedEdge[0] >= startVertexId && selectedEdge[0] < endVertexId) {
                    int pos = binarySearch<2>(edgeList, startEdgeIdx, startEdgeIdx + edgeCounts[tid], selectedEdge);
                    if (pos < 0) {
                        insertElement<2>(edgeList, startEdgeIdx, edgeCounts[tid], -pos-1, selectedEdge);
                        edgeCounts[tid] += 1;
                    }
                }
            }
        }
    }
}


__global__ void computeVertexCells(int *vertexCellSize, const int *connectivity, const int *offset, const int cellNumber, const int dimension) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cellNumber) return;

    int cellId = offset[tid];
    for (int i = 0; i <= dimension; ++i) {
        int vertexId = connectivity[cellId + i];
        atomicAdd(&vertexCellSize[vertexId], 1);
    }
}

/* The following kernels compute relations for a single consumer thread */

/**
 * Compute the boundary edge list
*/
__global__ void computeBoundaryEdges(int *edgeList, int *triangleList,
            bool *boundaryEdges, bool *boundaryTriangles, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *edgeIntervals, const int *triangleIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseEdgeId = edgeIntervals[baseCluster - 1];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startEdgeId: %d, endEdgeId: %d\n", threadIdx.x, tid, startEdgeId, endEdgeId);

    // loop through the external cell list
    int clusterStartCellId = externalCellOffsets[tid - 1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] < startVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    if (selectedTriangle[1] >= startVertexId && selectedTriangle[1] < endVertexId) {
                        for (int vk = vj+1; vk <= dimension; ++vk) {
                            selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                            int exCluster = indexArray[selectedTriangle[0]];
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], selectedTriangle);
                            if (boundaryTriangles[triangleId]) {
                                int selectedEdge[2] = {selectedTriangle[1], selectedTriangle[2]};
                                int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId;
                                boundaryEdges[relativeEid] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // determine the range of internal triangles
    clusterStartCellId = triangleIntervals[tid - 1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        if (boundaryTriangles[triangleId]) {
            int cellId = triangleId * 3;
            for (int vi = 0; vi < 2; ++vi) {
                int selectedEdge[2] = {triangleList[cellId + vi]};
                if (selectedEdge[0] < endVertexId) {
                    for (int vj = vi+1; vj < 3; ++vj) {
                        selectedEdge[1] = triangleList[cellId + vj];
                        int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId;
                        boundaryEdges[relativeEid] = true;
                    }
                }
            }
        }
    }
}

/**
 * Compute the boundary triangle list.
*/
__global__ void computeBoundaryTriangles(int *triangleList, int *triangleStarMatrix, 
            int *triangleStarSizes, bool *boundaryTriangles,
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseTriangleId = triangleIntervals[baseCluster - 1];
    const int startTriangleId = triangleIntervals[tid-1] - baseTriangleId;
    const int endTriangleId = triangleIntervals[tid] - baseTriangleId;

    // printf("threadIdx.x: %d, tid: %d, startTriangleId: %d, endTriangleId: %d\n", threadIdx.x, tid, startTriangleId, endTriangleId);

    // determine the range of cells
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] < endVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    for (int vk = vj+1; vk <= dimension; ++vk) {
                        selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                        selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int relativeFid = binarySearch<3>(triangleList, startTriangleId, endTriangleId, selectedTriangle);
                        int *arr = triangleStarMatrix + ftColumnNum * relativeFid;
                        int localSize = triangleStarSizes[relativeFid];
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, cid);
                        } while (oldVal != -1);
                        atomicAdd(&triangleStarSizes[relativeFid], 1);
                    }
                }
            }
        }
    }

    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 1; vi < dimension; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] >= startVertexId && selectedTriangle[0] < endVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    for (int vk = vj+1; vk <= dimension; ++vk) {
                        selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                        selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int relativeFid = binarySearch<3>(triangleList, startTriangleId, endTriangleId, selectedTriangle);
                        int *arr = triangleStarMatrix + ftColumnNum * relativeFid;
                        int localSize = triangleStarSizes[relativeFid];
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, cid);
                        } while (oldVal != -1);
                        atomicAdd(&triangleStarSizes[relativeFid], 1);
                    }
                }
            }
        }
    }

    __syncthreads();
    //  fill up the boundary triangle list
    // reuse the variables before
    cellsPerThread = (endTriangleId - startTriangleId) / numThreadsPerCluster;
    startCellId = startTriangleId + baseTriangleId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? (endTriangleId + baseTriangleId) : (startCellId + cellsPerThread);
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        if (triangleStarSizes[triangleId] == 1) {
            boundaryTriangles[triangleId] = true;
        }
    }
}

/**
 * Compute the boundary vertex list.
*/
__global__ void computeBoundaryVertices(int *triangleList, bool *boundaryTriangles, bool *boundaryVertices,
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple block threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;
    
    // determine the range of vertices
    int baseVertexId = vertexIntervals[baseCluster - 1];
    int startVertexId = vertexIntervals[tid-1];
    int endVertexId = vertexIntervals[tid];
    
    int clusterStartCellId = externalCellOffsets[tid-1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // only external triangles need to be checked for duplication
    for (int i = startCellId; i < endCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int newTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            // int localViId = viId - vertexIntervals[tid-1];
            if (newTriangle[0] < startVertexId) {
                int exCluster = indexArray[newTriangle[0]];
                for (int vj = vi + 1; vj < dimension; ++vj) {
                    newTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    for (int vk = vj + 1; vk <= dimension; ++vk) {
                        newTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);
                        if (boundaryTriangles[triangleId]) {
                            if (newTriangle[1] >= startVertexId && newTriangle[1] < endVertexId) {
                                int relativeVid = newTriangle[1] - baseVertexId;
                                boundaryVertices[relativeVid] = true;
                            }
                            if (newTriangle[2] >= startVertexId && newTriangle[2] < endVertexId) {
                                int relativeVid= newTriangle[2] - baseVertexId;
                                boundaryVertices[relativeVid] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // reuse the variables (cell refers to triangle here)
    clusterStartCellId = triangleIntervals[tid-1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // internel triangles can be handled from the triangle list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        // check if the triangle is on the boundary
        if (boundaryTriangles[triangleId]) {
            int relativeVid = triangleList[triangleId * 3] - baseVertexId;
            boundaryVertices[relativeVid] = true;
            for (int i = 1; i < dimension; ++i) {
                if (triangleList[triangleId * 3 + i] >= endVertexId) break;
                relativeVid = triangleList[triangleId * 3 + i] - baseVertexId;
                boundaryVertices[relativeVid] = true;
            }
        }
    }
}

/**
 * Compute the edge triangle relation.
*/
__global__ void computeEdgeTriangles(int *edgeList, int *triangleList,
            int *edgeTriangleMatrix, int *edgeTriangleSizes, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *edgeIntervals, const int *triangleIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseEdgeId = edgeIntervals[baseCluster - 1];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startEdgeId: %d, endEdgeId: %d\n", threadIdx.x, tid, startEdgeId, endEdgeId);

    // loop through the external cell list
    int clusterStartCellId = externalCellOffsets[tid - 1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] < startVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    if (selectedTriangle[1] >= startVertexId && selectedTriangle[1] < endVertexId) {
                        for (int vk = vj+1; vk <= dimension; ++vk) {
                            selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                            int selectedEdge[2] = {selectedTriangle[1], selectedTriangle[2]};
                            int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId;
                            int exCluster = indexArray[selectedTriangle[0]];
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], selectedTriangle);
                            int *arr = edgeTriangleMatrix + efColumnNum * relativeEid;
                            int localSize = edgeTriangleSizes[relativeEid];
                            // check the duplication first
                            bool insertion = true;
                            for (int k = 0; k < localSize; ++k) {
                                if (arr[k] == triangleId) {
                                    insertion = false;
                                    break;
                                }
                            }
                            if (insertion) {
                                int oldVal = -1;
                                do {
                                    oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                                } while (oldVal != -1 && oldVal != triangleId);
                                if (oldVal == -1) {
                                    atomicAdd(&edgeTriangleSizes[relativeEid], 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // determine the range of internal triangles
    clusterStartCellId = triangleIntervals[tid - 1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        int cellId = triangleId * 3;
        for (int vi = 0; vi < 2; ++vi) {
            int selectedEdge[2] = {triangleList[cellId + vi]};
            if (selectedEdge[0] < endVertexId) {
                for (int vj = vi+1; vj < 3; ++vj) {
                    selectedEdge[1] = triangleList[cellId + vj];
                    int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId;
                    int *arr = edgeTriangleMatrix + efColumnNum * relativeEid;
                    int localSize = edgeTriangleSizes[relativeEid];
                    int oldVal = -1;
                    do {
                        oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                    } while (oldVal != -1);
                    atomicAdd(&edgeTriangleSizes[relativeEid], 1);
                }
            }
        }
    }
}

/**
 * Compute the edge tetrahedron relation.
*/
__global__ void computeEdgeStars(int *edgeList, int *edgeStarMatrix, int *edgeStarSizes, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *edgeIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseEdgeId = edgeIntervals[baseCluster - 1];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startEdgeId: %d, endEdgeId: %d\n", threadIdx.x, tid, startEdgeId, endEdgeId);

    // determine the range of cells
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension; ++vi) {
            int selectedEdge[2] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedEdge[0] < endVertexId) {
                for (int vj = vi+1; vj <= dimension; ++vj) {
                    selectedEdge[1] = static_cast<int>(connectivity[cellId + vj]);
                    int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId;
                    int *arr = edgeStarMatrix + etColumnNum * relativeEid;
                    int localSize = edgeStarSizes[relativeEid];
                    int oldVal = -1;
                    do {
                        oldVal = atomicCAS(&arr[localSize++], -1, cid);
                    } while (oldVal != -1);
                    atomicAdd(&edgeStarSizes[relativeEid], 1);
                }
            }
        }
    }

    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 1; vi < dimension; ++vi) {
            int selectedEdge[2] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedEdge[0] >= startVertexId && selectedEdge[0] < endVertexId) {
                for (int vj = vi+1; vj <= dimension; ++vj) {
                    selectedEdge[1] = static_cast<int>(connectivity[cellId + vj]);
                    int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId;
                    int *arr = edgeStarMatrix + etColumnNum * relativeEid;
                    int localSize = edgeStarSizes[relativeEid];
                    int oldVal = -1;
                    do {
                        oldVal = atomicCAS(&arr[localSize++], -1, cid);
                    } while (oldVal != -1);
                    atomicAdd(&edgeStarSizes[relativeEid], 1);
                }
            }
        }
    }
}

/**
 * Compute the triangle tetrahedron relation.
*/
__global__ void computeTriangleStars(int *triangleList, int *triangleStarMatrix, int *triangleStarSizes, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseTriangleId = triangleIntervals[baseCluster - 1];
    const int startTriangleId = triangleIntervals[tid-1];
    const int endTriangleId = triangleIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startTriangleId: %d, endTriangleId: %d\n", threadIdx.x, tid, startTriangleId, endTriangleId);

    // determine the range of cells
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] < endVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    for (int vk = vj+1; vk <= dimension; ++vk) {
                        selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                        selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int relativeFid = binarySearch<3>(triangleList, startTriangleId, endTriangleId, selectedTriangle) - baseTriangleId;
                        int *arr = triangleStarMatrix + ftColumnNum * relativeFid;
                        int localSize = triangleStarSizes[relativeFid];
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, cid);
                        } while (oldVal != -1);
                        atomicAdd(&triangleStarSizes[relativeFid], 1);
                    }
                }
            }
        }
    }

    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 1; vi < dimension; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] >= startVertexId && selectedTriangle[0] < endVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    for (int vk = vj+1; vk <= dimension; ++vk) {
                        selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                        selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int relativeFid = binarySearch<3>(triangleList, startTriangleId, endTriangleId, selectedTriangle) - baseTriangleId;
                        int *arr = triangleStarMatrix + ftColumnNum * relativeFid;
                        int localSize = triangleStarSizes[relativeFid];
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, cid);
                        } while (oldVal != -1);
                        atomicAdd(&triangleStarSizes[relativeFid], 1);
                    }
                }
            }
        }
    }
}

/**
 * Compute the vertex edge relation.
*/
__global__ void computeVertexEdges(int *edgeList, int *vertexEdgeMatrix, int *vertexEdgeSizes, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *edgeIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // each block thread computes one cluster
    // int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;

    // multiple block threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;
    
    // determine the range of vertices
    int baseVertexId = vertexIntervals[baseCluster - 1];
    int startVertexId = vertexIntervals[tid-1];
    int endVertexId = vertexIntervals[tid];

    int clusterStartCellId = externalCellOffsets[tid-1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // only external edges need to be checked for duplication
    for (int i = startCellId; i < endCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension; ++vi) {
            int selectedEdge[2] = {static_cast<int>(connectivity[cellId + vi])};
            // int localViId = viId - vertexIntervals[tid-1];
            if (selectedEdge[0] < startVertexId) {
                int exCluster = indexArray[selectedEdge[0]];
                for (int vj = vi + 1; vj <= dimension; ++vj) {
                    selectedEdge[1] = static_cast<int>(connectivity[cellId + vj]);
                    // // already included in the internal edge list
                    // if (viId >= startVertexId && viId < endVertexId) {
                    //     int relativeViId = viId - vertexIntervals[baseCluster-1];
                    //     int pos = binarySearch(vertexEdgeMatrix + veColumnNum * relativeViId, vertexEdgeSizes[relativeViId], edgeId);
                    //     if (pos < 0) {
                    //         insertElement(vertexEdgeMatrix + veColumnNum * relativeViId, vertexEdgeSizes[relativeViId], -pos-1, edgeId);
                    //         vertexEdgeSizes[relativeViId] += 1;
                    //     }
                    // }
                    if (selectedEdge[1] >= startVertexId && selectedEdge[1] < endVertexId) {
                        int relativeVid = selectedEdge[1] - baseVertexId;
                        int edgeId = binarySearch<2>(edgeList, edgeIntervals[exCluster-1], edgeIntervals[exCluster], selectedEdge);
                        bool insert = true;
                        int *arr = vertexEdgeMatrix + veColumnNum * relativeVid;
                        int localSize = vertexEdgeSizes[relativeVid];
                        // linear search to check the duplication
                        for (int k = 0; k < localSize; ++k) {
                            if (arr[k] == edgeId) {
                                insert = false;
                                break;
                            }
                        }
                        // not found, insert the new edge id
                        if (insert) {
                            int oldVal = -1;
                            do {
                                oldVal = atomicCAS(&arr[localSize++], -1, edgeId);
                            } while (oldVal != -1 && oldVal != edgeId);
                            if (oldVal == -1) {
                                atomicAdd(&vertexEdgeSizes[relativeVid], 1);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // reuse the variables (cell refers to edge here)
    clusterStartCellId = edgeIntervals[tid-1];
    clusterEndCellId = edgeIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // internel edges can be handled from the edge list
    for (int edgeId = startCellId; edgeId < endCellId; ++edgeId) {
        int relativeVid = edgeList[edgeId * 2] - baseVertexId;
        // no need to check duplication as the edge id is unique
        int *arr = vertexEdgeMatrix + veColumnNum * relativeVid;
        int oldVal = -1;
        int localSize = vertexEdgeSizes[relativeVid];
        do {
            oldVal = atomicCAS(&arr[localSize++], -1, edgeId);
        } while (oldVal != -1);
        atomicAdd(&vertexEdgeSizes[relativeVid], 1);
        if (edgeList[edgeId * 2 + 1] < endVertexId) {
            relativeVid = edgeList[edgeId * 2 + 1] - baseVertexId;
            arr = vertexEdgeMatrix + veColumnNum * relativeVid;
            oldVal = -1;
            localSize = vertexEdgeSizes[relativeVid];
            do {
                oldVal = atomicCAS(&arr[localSize++], -1, edgeId);
            } while (oldVal != -1);
            atomicAdd(&vertexEdgeSizes[relativeVid], 1);
        }
    }
}

/**
 * Compute the vertex neighbor relations for ONLY ONE CLUSTER including the vertices in the external cells.
 * The relation array will store the local vertex ids. 
 * vertexMap: map the local vertex id to the global vertex id (need to be initialized to -1)
 * totalVertexNum: the number of vertices in the vertex neighbor relations (need to be initialized to internal vertex number)
 *
*/
__global__ void computeVertexNeighborsWithExternals(
            int *vertexNeighborMatrix, int *vertexNeighborSizes, 
            int *vertexMap, int *totalVertexNum,
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets,
            const int dimension, const int sfcp_ColumnNum) {
    
    // redudant check if necessary to make sure only one cluster is processed
    // int clustersToCompute = (gridDim.x * blockDim.x) / numThreadsPerCluster;
    // multiple threads compute one cluster
    int numThreadsPerCluster = gridDim.x * blockDim.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Thread %d, Block %d: tid=%d, clusterNumber=%d\n", threadIdx.x, blockIdx.x, tid, clusterNumber);
    if (baseCluster >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId = vertexIntervals[baseCluster-1];
    const int endVertexId = vertexIntervals[baseCluster];
    
    // parallel writes to the vertex map for the internal vertices 
    // (might be optional as not too much speedup expected)
    int verticesPerThread = (endVertexId - startVertexId) / numThreadsPerCluster;
    int startVid = startVertexId + threadId * verticesPerThread;
    int endVid = (threadId == (numThreadsPerCluster - 1)) ? endVertexId : (startVid + verticesPerThread);
    for (int vid = startVid; vid < endVid; ++vid) {
        vertexMap[vid - startVertexId] = vid;
    }

    // printf("threadIdx.x: %d, tid: %d, startVertexId: %d, endVertexId: %d\n", threadIdx.x, tid, startVertexId, endVertexId);

    // determine the range of vertices
    int clusterStartCellId = cellIntervals[baseCluster - 1];
    int clusterEndCellId = cellIntervals[baseCluster];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + threadId * cellsPerThread;
    int endCellId = (threadId == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    
    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int localViId = viId - startVertexId;
                    int localVjId = findOrInsert(vertexMap, totalVertexNum, vjId);
                    
                    int *arr = vertexNeighborMatrix + sfcp_ColumnNum * localViId;
                    findOrInsert(arr, &vertexNeighborSizes[localViId], localVjId);
                    arr = vertexNeighborMatrix + sfcp_ColumnNum * localVjId;
                    findOrInsert(arr, &vertexNeighborSizes[localVjId], localViId);
                }
            }
        }
    }
    
    // loop through the external cell list
    // reuse previous variables
    clusterStartCellId = externalCellOffsets[baseCluster - 1];
    clusterEndCellId = externalCellOffsets[baseCluster];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + threadId * cellsPerThread;
    endCellId = (threadId == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // printf("threadIdx.x: %d, tid: %d, startVertexId: %d, endVertexId: %d, startExCellIdx: %d, endExCellIdx: %d\n", threadIdx.x, tid, startVertexId, endVertexId, startCellId, endCellId);

    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[externalCells[cid]];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int localVjId = vjId - startVertexId;
                    int localViId = findOrInsert(vertexMap, totalVertexNum, viId);
                    int *arr = vertexNeighborMatrix + sfcp_ColumnNum * localVjId;
                    findOrInsert(arr, &vertexNeighborSizes[localVjId], localViId);
                    arr = vertexNeighborMatrix + sfcp_ColumnNum * localViId;
                    findOrInsert(arr, &vertexNeighborSizes[localViId], localVjId);
                }
            }
        }
    }
}

/**
 * Compute the vertex neighbor relation.
*/
__global__ void computeVertexNeighbors(int *vertexNeighborMatrix, int *vertexNeighborSizes, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets,
            const int dimension, const int numThreadsPerCluster,
            const int sfcp_ColumnNum) {
    
    // multiple threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    // printf("Thread %d, Block %d: tid=%d, clusterNumber=%d\n", threadIdx.x, blockIdx.x, tid, clusterNumber);
    // int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseVertexId = vertexIntervals[baseCluster - 1];

    // printf("threadIdx.x: %d, tid: %d, startVertexId: %d, endVertexId: %d\n", threadIdx.x, tid, startVertexId, endVertexId);

    // determine the range of vertices
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    
    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeVid = viId - baseVertexId;
                    // linear search to check the duplication
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + sfcp_ColumnNum * relativeVid;
                    int localSize = vertexNeighborSizes[relativeVid];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == vjId) {
                            insertion = false;
                            break;
                        }
                    }
                    // not found, need to insert the element
                    if (insertion) {
                        int oldVal = -1;
                        // handle the case where multiple threads work with the same location
                        // one array index can only be updated by one thread
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, vjId);
                        } while (oldVal != -1 && oldVal != vjId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeVid], 1);
                        }
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVid = vjId - baseVertexId;
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + sfcp_ColumnNum * relativeVid;
                    int localSize = vertexNeighborSizes[relativeVid];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == viId) {
                            insertion = false;
                            break;
                        }
                    }
                    if (insertion) {
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, viId);
                        } while (oldVal != -1 && oldVal != viId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeVid], 1);
                        }
                    }
                }
            }
        }
    }
    
    // loop through the external cell list
    // reuse previous variables 
    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // printf("threadIdx.x: %d, tid: %d, startVertexId: %d, endVertexId: %d, startExCellIdx: %d, endExCellIdx: %d\n", threadIdx.x, tid, startVertexId, endVertexId, startCellId, endCellId);

    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[externalCells[cid]];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            // int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                // int localVjId = vjId - vertexIntervals[tid-1];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeVid = viId - baseVertexId;
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + sfcp_ColumnNum * relativeVid;
                    int localSize = vertexNeighborSizes[relativeVid];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == vjId) {
                            insertion = false;
                            break;
                        }
                    }
                    if (insertion) {
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, vjId);
                        } while (oldVal != -1 && oldVal != vjId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeVid], 1);
                        }
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVid = vjId - baseVertexId;
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + sfcp_ColumnNum * relativeVid;
                    int localSize = vertexNeighborSizes[relativeVid];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == viId) {
                            insertion = false;
                            break;
                        }
                    }
                    if (insertion) {
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, viId);
                        } while (oldVal != -1 && oldVal != viId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeVid], 1);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Compute the vertex tetrahedron relation.
*/
__global__ void computeVertexStars(int *vertexStarMatrix, int *vertexStarSizes, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseVertexId = vertexIntervals[baseCluster - 1];

    // determine the range of cells
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            if (viId < endVertexId) {
                int relativeVid = viId - baseVertexId;
                int *arr = vertexStarMatrix + vtColumnNum * relativeVid;
                int localSize = vertexStarSizes[relativeVid];
                int oldVal = -1;
                do {
                    oldVal = atomicCAS(&arr[localSize++], -1, cid);
                } while (oldVal != -1);
                atomicAdd(&vertexStarSizes[relativeVid], 1);
                // vertexStarMatrix[vtColumnNum * relativeViId + vertexStarSizes[relativeViId]] = cid;
                // vertexStarSizes[relativeViId] += 1;
            }
        }
    }

    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 1; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
                // int localVjId = vjId - vertexIntervals[tid-1];
            if (viId >= startVertexId && viId < endVertexId) {
                int relativeVid = viId - baseVertexId;
                int *arr = vertexStarMatrix + vtColumnNum * relativeVid;
                int localSize = vertexStarSizes[relativeVid];
                int oldVal = -1;
                do {
                    oldVal = atomicCAS(&arr[localSize++], -1, cid);
                } while (oldVal != -1);
                atomicAdd(&vertexStarSizes[relativeVid], 1);
                // vertexStarMatrix[vtColumnNum * relativeViId + vertexStarSizes[relativeViId]] = cid;
                // vertexStarSizes[relativeViId] += 1;
            }
        }
    }
}

/**
 * Compute the vertex triangle relation.
*/
__global__ void computeVertexTriangles(int *triangleList, int *vertexTriangleMatrix, int *vertexTriangleSizes, 
            const int baseCluster, const int clusterNumber,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // each block thread computes one cluster
    // int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;

    // multiple block threads compute one cluster
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;
    
    // determine the range of vertices
    int baseVertexId = vertexIntervals[baseCluster - 1];
    int startVertexId = vertexIntervals[tid-1];
    int endVertexId = vertexIntervals[tid];
    
    int clusterStartCellId = externalCellOffsets[tid-1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // only external triangles need to be checked for duplication
    for (int i = startCellId; i < endCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int newTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            // int localViId = viId - vertexIntervals[tid-1];
            if (newTriangle[0] < startVertexId) {
                int exCluster = indexArray[newTriangle[0]];
                for (int vj = vi + 1; vj < dimension; ++vj) {
                    newTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    for (int vk = vj + 1; vk <= dimension; ++vk) {
                        newTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        // the second vertex belongs to the cluster
                        if (newTriangle[1] >= startVertexId && newTriangle[1] < endVertexId) {
                            int relativeVid = newTriangle[1] - baseVertexId;
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);

                            bool insert = true;
                            int *arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
                            int localSize = vertexTriangleSizes[relativeVid];
                            // linear search to check the duplication
                            for (int k = 0; k < localSize; ++k) {
                                if (arr[k] == triangleId) {
                                    insert = false;
                                    break;
                                }
                            }
                            // not found, insert the new edge id
                            if (insert) {
                                int oldVal = -1;
                                do {
                                    oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                                } while (oldVal != -1 && oldVal != triangleId);
                                if (oldVal == -1) {
                                    atomicAdd(&vertexTriangleSizes[relativeVid], 1);
                                }
                            }
                            // int pos = binarySearch1D(, vertexTriangleSizes[relativeVid], triangleId);
                            // if (pos < 0) {
                            //     insertElement1D(vertexTriangleMatrix + vfColumnNum * relativeVid, vertexTriangleSizes[relativeVid], -pos-1, triangleId);
                            //     vertexTriangleSizes[relativeVid] += 1;
                            // }
                        }
                        // the last vertex belongs to the cluster
                        if (newTriangle[2] >= startVertexId && newTriangle[2] < endVertexId) {
                            int relativeVid= newTriangle[2] - baseVertexId;
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);

                            bool insert = true;
                            int *arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
                            int localSize = vertexTriangleSizes[relativeVid];
                            // linear search to check the duplication
                            for (int k = 0; k < localSize; ++k) {
                                if (arr[k] == triangleId) {
                                    insert = false;
                                    break;
                                }
                            }
                            // not found, insert the new edge id
                            if (insert) {
                                int oldVal = -1;
                                do {
                                    oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                                } while (oldVal != -1 && oldVal != triangleId);
                                if (oldVal == -1) {
                                    atomicAdd(&vertexTriangleSizes[relativeVid], 1);
                                }
                            }
                            // int pos = binarySearch1D(vertexTriangleMatrix + vfColumnNum * relativeVid, vertexTriangleSizes[relativeVid], triangleId);
                            // if (pos < 0) {
                            //     insertElement1D(vertexTriangleMatrix + vfColumnNum * relativeVid, vertexTriangleSizes[relativeVid], -pos-1, triangleId);
                            //     vertexTriangleSizes[relativeVid] += 1;
                            // }
                        }
                    }
                }
            }
        }
    }

    // reuse the variables (cell refers to triangle here)
    clusterStartCellId = triangleIntervals[tid-1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // internel triangles can be handled from the triangle list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        int relativeVid = triangleList[triangleId * 3] - baseVertexId;
        int *arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
        int oldVal = -1;
        int localSize = vertexTriangleSizes[relativeVid];
        do {
            oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
        } while (oldVal != -1);
        atomicAdd(&vertexTriangleSizes[relativeVid], 1);
        // insertElement1D(vertexTriangleMatrix + vfColumnNum * relativeVid, vertexTriangleSizes[relativeVid], vertexTriangleSizes[relativeVid], triangleId);
        // vertexTriangleSizes[relativeVid] += 1;
        for (int i = 1; i < dimension; ++i) {
            if (triangleList[triangleId * 3 + i] >= endVertexId) break;
            relativeVid = triangleList[triangleId * 3 + i] - baseVertexId;
            arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
            oldVal = -1;
            localSize = vertexTriangleSizes[relativeVid];
            do {
                oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
            } while (oldVal != -1);
            atomicAdd(&vertexTriangleSizes[relativeVid], 1);
            // insertElement1D(vertexTriangleMatrix + vfColumnNum * relativeVid, vertexTriangleSizes[relativeVid], vertexTriangleSizes[relativeVid], triangleId);
            // vertexTriangleSizes[relativeVid] += 1;
        }
    }
}


/* The following kernels compute relations for multiple consumer threads */

/**
 * Compute boundary edge list.
*/
__global__ void computeMultiBoundaryEdges(int *edgeList, int *triangleList,
            bool *boundaryTriangles, int *boundaryEdges, 
            const int clusterNumber, const int *baseClusters, const int *edgeOffsets,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *edgeIntervals, const int *triangleIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int edgeOffset = edgeOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseEdgeId = edgeIntervals[baseCluster - 1];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startEdgeId: %d, endEdgeId: %d\n", threadIdx.x, tid, startEdgeId, endEdgeId);

    // loop through the external cell list
    int clusterStartCellId = externalCellOffsets[tid - 1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] < startVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    if (selectedTriangle[1] >= startVertexId && selectedTriangle[1] < endVertexId) {
                        for (int vk = vj+1; vk <= dimension; ++vk) {
                            selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                            int exCluster = indexArray[selectedTriangle[0]];
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], selectedTriangle);
                            if (boundaryTriangles[triangleId]) {
                                int selectedEdge[2] = {selectedTriangle[1], selectedTriangle[2]};
                                int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId + edgeOffset;
                                boundaryEdges[relativeEid] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // determine the range of internal triangles
    clusterStartCellId = triangleIntervals[tid - 1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        if (boundaryTriangles[triangleId]) {
            int cellId = triangleId * 3;
            for (int vi = 0; vi < 2; ++vi) {
                int selectedEdge[2] = {triangleList[cellId + vi]};
                if (selectedEdge[0] < endVertexId) {
                    for (int vj = vi+1; vj < 3; ++vj) {
                        selectedEdge[1] = triangleList[cellId + vj];
                        int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId + edgeOffset;
                        boundaryEdges[relativeEid] = 1;
                    }
                }
            }
        }
    }
}

/**
 * Compute the boundary vertex list.
*/
__global__ void computeMultiBoundaryVertices(int *triangleList, bool *boundaryTriangles, int *boundaryVertices,
            const int clusterNumber, const int *baseClusters, const int *vertexOffsets,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple block threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;
    
    // determine the range of vertices
    int baseVertexId = vertexIntervals[baseCluster - 1];
    int startVertexId = vertexIntervals[tid-1];
    int endVertexId = vertexIntervals[tid];
    
    int clusterStartCellId = externalCellOffsets[tid-1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // only external triangles need to be checked for duplication
    for (int i = startCellId; i < endCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int newTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            // int localViId = viId - vertexIntervals[tid-1];
            if (newTriangle[0] < startVertexId) {
                int exCluster = indexArray[newTriangle[0]];
                for (int vj = vi + 1; vj < dimension; ++vj) {
                    newTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    for (int vk = vj + 1; vk <= dimension; ++vk) {
                        newTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);
                        if (boundaryTriangles[triangleId]) {
                            if (newTriangle[1] >= startVertexId && newTriangle[1] < endVertexId) {
                                int relativeVid = newTriangle[1] - baseVertexId + vertexOffset;
                                boundaryVertices[relativeVid] = 1;
                            }
                            if (newTriangle[2] >= startVertexId && newTriangle[2] < endVertexId) {
                                int relativeVid= newTriangle[2] - baseVertexId + vertexOffset;
                                boundaryVertices[relativeVid] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // reuse the variables (cell refers to triangle here)
    clusterStartCellId = triangleIntervals[tid-1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // internel triangles can be handled from the triangle list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        // check if the triangle is on the boundary
        if (boundaryTriangles[triangleId]) {
            int relativeVid = triangleList[triangleId * 3] - baseVertexId + vertexOffset;
            boundaryVertices[relativeVid] = 1;
            for (int i = 1; i < dimension; ++i) {
                if (triangleList[triangleId * 3 + i] >= endVertexId) break;
                relativeVid = triangleList[triangleId * 3 + i] - baseVertexId + vertexOffset;
                boundaryVertices[relativeVid] = 1;
            }
        }
    }
}

/**
 * Compute the edge tetrahedron relation.
*/
__global__ void computeMultiEdgeStars(int *edgeList, int *edgeStarMatrix, int *edgeStarSizes, 
            const int clusterNumber, const int *baseClusters, const int *edgeOffsets,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *edgeIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int edgeOffset = edgeOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseEdgeId = edgeIntervals[baseCluster - 1];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startEdgeId: %d, endEdgeId: %d\n", threadIdx.x, tid, startEdgeId, endEdgeId);

    // determine the range of cells
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension; ++vi) {
            int selectedEdge[2] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedEdge[0] < endVertexId) {
                for (int vj = vi+1; vj <= dimension; ++vj) {
                    selectedEdge[1] = static_cast<int>(connectivity[cellId + vj]);
                    int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId + edgeOffset;
                    int *arr = edgeStarMatrix + etColumnNum * relativeEid;
                    int localSize = edgeStarSizes[relativeEid];
                    int oldVal = -1;
                    do {
                        oldVal = atomicCAS(&arr[localSize++], -1, cid);
                    } while (oldVal != -1);
                    atomicAdd(&edgeStarSizes[relativeEid], 1);
                }
            }
        }
    }

    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 1; vi < dimension; ++vi) {
            int selectedEdge[2] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedEdge[0] >= startVertexId && selectedEdge[0] < endVertexId) {
                for (int vj = vi+1; vj <= dimension; ++vj) {
                    selectedEdge[1] = static_cast<int>(connectivity[cellId + vj]);
                    int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId + edgeOffset;
                    int *arr = edgeStarMatrix + etColumnNum * relativeEid;
                    int localSize = edgeStarSizes[relativeEid];
                    int oldVal = -1;
                    do {
                        oldVal = atomicCAS(&arr[localSize++], -1, cid);
                    } while (oldVal != -1);
                    atomicAdd(&edgeStarSizes[relativeEid], 1);
                }
            }
        }
    }
}

/**
 * Compute the edge triangle relation.
*/
__global__ void computeMultiEdgeTriangles(int *edgeList, int *triangleList,
            int *edgeTriangleMatrix, int *edgeTriangleSizes, 
            const int clusterNumber, const int *baseClusters, const int *edgeOffsets,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *edgeIntervals, const int *triangleIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int edgeOffset = edgeOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseEdgeId = edgeIntervals[baseCluster - 1];
    const int startEdgeId = edgeIntervals[tid-1];
    const int endEdgeId = edgeIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startEdgeId: %d, endEdgeId: %d\n", threadIdx.x, tid, startEdgeId, endEdgeId);

    // loop through the external cell list
    int clusterStartCellId = externalCellOffsets[tid - 1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] < startVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    if (selectedTriangle[1] >= startVertexId && selectedTriangle[1] < endVertexId) {
                        for (int vk = vj+1; vk <= dimension; ++vk) {
                            selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                            int selectedEdge[2] = {selectedTriangle[1], selectedTriangle[2]};
                            int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId + edgeOffset;
                            int exCluster = indexArray[selectedTriangle[0]];
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], selectedTriangle);
                            int *arr = edgeTriangleMatrix + efColumnNum * relativeEid;
                            int localSize = edgeTriangleSizes[relativeEid];
                            // check the duplication first
                            bool insertion = true;
                            for (int k = 0; k < localSize; ++k) {
                                if (arr[k] == triangleId) {
                                    insertion = false;
                                    break;
                                }
                            }
                            if (insertion) {
                                int oldVal = -1;
                                do {
                                    oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                                } while (oldVal != -1 && oldVal != triangleId);
                                if (oldVal == -1) {
                                    atomicAdd(&edgeTriangleSizes[relativeEid], 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // determine the range of internal triangles
    clusterStartCellId = triangleIntervals[tid - 1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        int cellId = triangleId * 3;
        for (int vi = 0; vi < 2; ++vi) {
            int selectedEdge[2] = {triangleList[cellId + vi]};
            if (selectedEdge[0] < endVertexId) {
                for (int vj = vi+1; vj < 3; ++vj) {
                    selectedEdge[1] = triangleList[cellId + vj];
                    int relativeEid = binarySearch<2>(edgeList, startEdgeId, endEdgeId, selectedEdge) - baseEdgeId + edgeOffset;
                    int *arr = edgeTriangleMatrix + efColumnNum * relativeEid;
                    int localSize = edgeTriangleSizes[relativeEid];
                    int oldVal = -1;
                    do {
                        oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                    } while (oldVal != -1);
                    atomicAdd(&edgeTriangleSizes[relativeEid], 1);
                }
            }
        }
    }
}

/**
 * Compute the triangle tetrahedron relation.
*/
__global__ void computeMultiTriangleStars(int *triangleList, int *triangleStarMatrix, int *triangleStarSizes, 
            const int clusterNumber, const int *baseClusters, const int *triangleOffsets,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int triangleOffset = triangleOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    /*
    * divide the job by clusters
    */
    const int startVertexId= vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseTriangleId = triangleIntervals[baseCluster - 1];
    const int startTriangleId = triangleIntervals[tid-1];
    const int endTriangleId = triangleIntervals[tid];

    // printf("threadIdx.x: %d, tid: %d, startTriangleId: %d, endTriangleId: %d\n", threadIdx.x, tid, startTriangleId, endTriangleId);

    // determine the range of cells
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] < endVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    for (int vk = vj+1; vk <= dimension; ++vk) {
                        selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                        selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int relativeFid = binarySearch<3>(triangleList, startTriangleId, endTriangleId, selectedTriangle) - baseTriangleId + triangleOffset;
                        int *arr = triangleStarMatrix + ftColumnNum * relativeFid;
                        int localSize = triangleStarSizes[relativeFid];
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, cid);
                        } while (oldVal != -1);
                        atomicAdd(&triangleStarSizes[relativeFid], 1);
                    }
                }
            }
        }
    }

    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 1; vi < dimension; ++vi) {
            int selectedTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            if (selectedTriangle[0] >= startVertexId && selectedTriangle[0] < endVertexId) {
                for (int vj = vi+1; vj < dimension; ++vj) {
                    for (int vk = vj+1; vk <= dimension; ++vk) {
                        selectedTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                        selectedTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        int relativeFid = binarySearch<3>(triangleList, startTriangleId, endTriangleId, selectedTriangle) - baseTriangleId + triangleOffset;
                        int *arr = triangleStarMatrix + ftColumnNum * relativeFid;
                        int localSize = triangleStarSizes[relativeFid];
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, cid);
                        } while (oldVal != -1);
                        atomicAdd(&triangleStarSizes[relativeFid], 1);
                    }
                }
            }
        }
    }
}

/**
 * Compute the vertex edge relation.
*/
__global__ void computeMultiVertexEdges(int *edgeList, int *vertexEdgeMatrix, 
            int *vertexEdgeSizes, const int clusterNumber, 
            const int *baseClusters, const int *vertexOffsets,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *edgeIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple block threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;
    
    // determine the range of vertices
    int baseVertexId = vertexIntervals[baseCluster - 1];
    int startVertexId = vertexIntervals[tid-1];
    int endVertexId = vertexIntervals[tid];

    int clusterStartCellId = externalCellOffsets[tid-1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // only external edges need to be checked for duplication
    for (int i = startCellId; i < endCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension; ++vi) {
            int selectedEdge[2] = {static_cast<int>(connectivity[cellId + vi])};
            // int localViId = viId - vertexIntervals[tid-1];
            if (selectedEdge[0] < startVertexId) {
                int exCluster = indexArray[selectedEdge[0]];
                for (int vj = vi + 1; vj <= dimension; ++vj) {
                    selectedEdge[1] = static_cast<int>(connectivity[cellId + vj]);
                    if (selectedEdge[1] >= startVertexId && selectedEdge[1] < endVertexId) {
                        int relativeVid = selectedEdge[1] - baseVertexId + vertexOffset;
                        int edgeId = binarySearch<2>(edgeList, edgeIntervals[exCluster-1], edgeIntervals[exCluster], selectedEdge);
                        bool insert = true;
                        int *arr = vertexEdgeMatrix + veColumnNum * relativeVid;
                        int localSize = vertexEdgeSizes[relativeVid];
                        // linear search to check the duplication
                        for (int k = 0; k < localSize; ++k) {
                            if (arr[k] == edgeId) {
                                insert = false;
                                break;
                            }
                        }
                        // not found, insert the new edge id
                        if (insert) {
                            int oldVal = -1;
                            do {
                                oldVal = atomicCAS(&arr[localSize++], -1, edgeId);
                            } while (oldVal != -1 && oldVal != edgeId);
                            if (oldVal == -1) {
                                atomicAdd(&vertexEdgeSizes[relativeVid], 1);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // reuse the variables (cell refers to edge here)
    clusterStartCellId = edgeIntervals[tid-1];
    clusterEndCellId = edgeIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // internel edges can be handled from the edge list
    for (int edgeId = startCellId; edgeId < endCellId; ++edgeId) {
        int relativeVid = edgeList[edgeId * 2] - baseVertexId + vertexOffset;
        // no need to check duplication as the edge id is unique
        int *arr = vertexEdgeMatrix + veColumnNum * relativeVid;
        int oldVal = -1;
        int localSize = vertexEdgeSizes[relativeVid];
        do {
            oldVal = atomicCAS(&arr[localSize++], -1, edgeId);
        } while (oldVal != -1);
        atomicAdd(&vertexEdgeSizes[relativeVid], 1);
        if (edgeList[edgeId * 2 + 1] < endVertexId) {
            relativeVid = edgeList[edgeId * 2 + 1] - baseVertexId + vertexOffset;
            arr = vertexEdgeMatrix + veColumnNum * relativeVid;
            oldVal = -1;
            localSize = vertexEdgeSizes[relativeVid];
            do {
                oldVal = atomicCAS(&arr[localSize++], -1, edgeId);
            } while (oldVal != -1);
            atomicAdd(&vertexEdgeSizes[relativeVid], 1);
        }
    }
}

/**
 * Compute the vertex triangle relation.
*/
__global__ void computeMultiVertexTriangles(int *triangleList, int *vertexTriangleMatrix, 
            int *vertexTriangleSizes, const int clusterNumber, 
            const int *baseClusters, const int *vertexOffsets,
            const int *connectivity, const int *offsets, const int *indexArray, 
            const int *vertexIntervals, const int *triangleIntervals, const int *cellIntervals,
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {

    // multiple block threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;
    
    // determine the range of vertices
    int baseVertexId = vertexIntervals[baseCluster - 1];
    int startVertexId = vertexIntervals[tid-1];
    int endVertexId = vertexIntervals[tid];
    
    int clusterStartCellId = externalCellOffsets[tid-1];
    int clusterEndCellId = externalCellOffsets[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // only external triangles need to be checked for duplication
    for (int i = startCellId; i < endCellId; ++i) {
        int cellId = offsets[externalCells[i]];
        for (int vi = 0; vi < dimension - 1; ++vi) {
            int newTriangle[3] = {static_cast<int>(connectivity[cellId + vi])};
            // int localViId = viId - vertexIntervals[tid-1];
            if (newTriangle[0] < startVertexId) {
                int exCluster = indexArray[newTriangle[0]];
                for (int vj = vi + 1; vj < dimension; ++vj) {
                    newTriangle[1] = static_cast<int>(connectivity[cellId + vj]);
                    for (int vk = vj + 1; vk <= dimension; ++vk) {
                        newTriangle[2] = static_cast<int>(connectivity[cellId + vk]);
                        // the second vertex belongs to the cluster
                        if (newTriangle[1] >= startVertexId && newTriangle[1] < endVertexId) {
                            int relativeVid = newTriangle[1] - baseVertexId + vertexOffset;
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);

                            bool insert = true;
                            int *arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
                            int localSize = vertexTriangleSizes[relativeVid];
                            // linear search to check the duplication
                            for (int k = 0; k < localSize; ++k) {
                                if (arr[k] == triangleId) {
                                    insert = false;
                                    break;
                                }
                            }
                            // not found, insert the new edge id
                            if (insert) {
                                int oldVal = -1;
                                do {
                                    oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                                } while (oldVal != -1 && oldVal != triangleId);
                                if (oldVal == -1) {
                                    atomicAdd(&vertexTriangleSizes[relativeVid], 1);
                                }
                            }
                        }
                        // the last vertex belongs to the cluster
                        if (newTriangle[2] >= startVertexId && newTriangle[2] < endVertexId) {
                            int relativeVid= newTriangle[2] - baseVertexId + vertexOffset;
                            int triangleId = binarySearch<3>(triangleList, triangleIntervals[exCluster-1], triangleIntervals[exCluster], newTriangle);

                            bool insert = true;
                            int *arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
                            int localSize = vertexTriangleSizes[relativeVid];
                            // linear search to check the duplication
                            for (int k = 0; k < localSize; ++k) {
                                if (arr[k] == triangleId) {
                                    insert = false;
                                    break;
                                }
                            }
                            // not found, insert the new edge id
                            if (insert) {
                                int oldVal = -1;
                                do {
                                    oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
                                } while (oldVal != -1 && oldVal != triangleId);
                                if (oldVal == -1) {
                                    atomicAdd(&vertexTriangleSizes[relativeVid], 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // reuse the variables (cell refers to triangle here)
    clusterStartCellId = triangleIntervals[tid-1];
    clusterEndCellId = triangleIntervals[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);
    // internel triangles can be handled from the triangle list
    for (int triangleId = startCellId; triangleId < endCellId; ++triangleId) {
        int relativeVid = triangleList[triangleId * 3] - baseVertexId + vertexOffset;
        int *arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
        int oldVal = -1;
        int localSize = vertexTriangleSizes[relativeVid];
        do {
            oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
        } while (oldVal != -1);
        atomicAdd(&vertexTriangleSizes[relativeVid], 1);
        for (int i = 1; i < dimension; ++i) {
            if (triangleList[triangleId * 3 + i] >= endVertexId) break;
            relativeVid = triangleList[triangleId * 3 + i] - baseVertexId + vertexOffset;
            arr = vertexTriangleMatrix + vfColumnNum * relativeVid;
            oldVal = -1;
            localSize = vertexTriangleSizes[relativeVid];
            do {
                oldVal = atomicCAS(&arr[localSize++], -1, triangleId);
            } while (oldVal != -1);
            atomicAdd(&vertexTriangleSizes[relativeVid], 1);
        }
    }
}

/**
 * Compute the vertex neighbor relation.
*/
__global__ void computeMultiVertexNeighbors(int *vertexNeighborMatrix, int *vertexNeighborSizes, 
            const int clusterNumber, const int *baseClusters, const int *vertexOffsets,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {
    // multiple threads compute one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    if (tid >= clusterNumber) return;

    // determine the range of vertices
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];
    const int baseVertexId = vertexIntervals[baseCluster - 1];

    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;   // assume the blockDim.x is divisible by numThreadsPerCluster
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // printf("threadIdx.x: %d, tid: %d, startVertexId: %d, endVertexId: %d\n", threadIdx.x, tid, startVertexId, endVertexId);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            for (int vj = vi+1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeViId = viId - baseVertexId + vertexOffset;
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + vvColumnNum * relativeViId;
                    int localSize = vertexNeighborSizes[relativeViId];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == vjId) {
                            insertion = false;
                            break;
                        }
                    }
                    // not found, need to insert the element
                    if (insertion) {
                        int oldVal = -1;
                        // handle the case where multiple threads work with the same location
                        // one array index can only be updated by one thread
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, vjId);
                        } while (oldVal != -1 && oldVal != vjId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeViId], 1);
                        }
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVid = vjId - baseVertexId + vertexOffset;
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + vvColumnNum * relativeVid;
                    int localSize = vertexNeighborSizes[relativeVid];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == viId) {
                            insertion = false;
                            break;
                        }
                    }
                    if (insertion) {
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, viId);
                        } while (oldVal != -1 && oldVal != viId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeVid], 1);
                        }
                    }
                }
            }
        }
    }

    // reuse previous variables
    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[externalCells[cid]];
        for (int vi = 0; vi < dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            // int localViId = viId - vertexIntervals[tid-1];
            for (int vj = vi + 1; vj <= dimension; ++vj) {
                int vjId = connectivity[cellId + vj];
                // int localVjId = vjId - vertexIntervals[tid-1];
                if (viId >= startVertexId && viId < endVertexId) {
                    int relativeVid = viId - baseVertexId + vertexOffset;
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + vvColumnNum * relativeVid;
                    int localSize = vertexNeighborSizes[relativeVid];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == vjId) {
                            insertion = false;
                            break;
                        }
                    }
                    if (insertion) {
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, vjId);
                        } while (oldVal != -1 && oldVal != vjId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeVid], 1);
                        }
                    }
                }
                if (vjId >= startVertexId && vjId < endVertexId) {
                    int relativeVid = vjId - baseVertexId + vertexOffset;
                    bool insertion = true;
                    int *arr = vertexNeighborMatrix + vvColumnNum * relativeVid;
                    int localSize = vertexNeighborSizes[relativeVid];
                    for (int k = 0; k < localSize; ++k) {
                        if (arr[k] == viId) {
                            insertion = false;
                            break;
                        }
                    }
                    if (insertion) {
                        int oldVal = -1;
                        do {
                            oldVal = atomicCAS(&arr[localSize++], -1, viId);
                        } while (oldVal != -1 && oldVal != viId);
                        if (oldVal == -1) {
                            atomicAdd(&vertexNeighborSizes[relativeVid], 1);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Compute the vertex tetrahedron relation.
*/
__global__ void computeMultiVertexStars(int *vertexStarMatrix, int *vertexStarSizes, 
            const int clusterNumber, const int *baseClusters, const int *vertexOffsets,
            const int *connectivity, const int *offsets, 
            const int *vertexIntervals, const int *cellIntervals, 
            const int *externalCells, const int *externalCellOffsets, 
            const int dimension, const int numThreadsPerCluster) {
    // each block thread computes one cluster
    int baseCluster = baseClusters[blockIdx.x];
    int vertexOffset = vertexOffsets[blockIdx.x];
    int tid = (blockIdx.y * blockDim.x + threadIdx.x) / numThreadsPerCluster + baseCluster;
    // int tid = blockIdx.x * blockDim.x + threadIdx.x + baseCluster;
    if (tid >= clusterNumber) return;

    // printf("blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);

    // determine the range of vertices
    const int baseVertexId = vertexIntervals[baseCluster-1];
    const int startVertexId = vertexIntervals[tid-1];
    const int endVertexId = vertexIntervals[tid];

    // determine the range of cells
    int clusterStartCellId = cellIntervals[tid - 1];
    int clusterEndCellId = cellIntervals[tid];
    int cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    int startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    int endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the internal cell list
    for (int cid = startCellId; cid < endCellId; ++cid) {
        int cellId = offsets[cid];
        for (int vi = 0; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
            if (viId < endVertexId) {
                int relativeVid = viId - baseVertexId + vertexOffset;
                int *arr = vertexStarMatrix + vtColumnNum * relativeVid;
                int localSize = vertexStarSizes[relativeVid];
                int oldVal = -1;
                do {
                    oldVal = atomicCAS(&arr[localSize++], -1, cid);
                } while (oldVal != -1);
                atomicAdd(&vertexStarSizes[relativeVid], 1);
            }
        }
    }

    clusterStartCellId = externalCellOffsets[tid - 1];
    clusterEndCellId = externalCellOffsets[tid];
    cellsPerThread = (clusterEndCellId - clusterStartCellId) / numThreadsPerCluster;
    startCellId = clusterStartCellId + (threadIdx.x % numThreadsPerCluster) * cellsPerThread;
    endCellId = (threadIdx.x % numThreadsPerCluster == (numThreadsPerCluster - 1)) ? clusterEndCellId : (startCellId + cellsPerThread);

    // loop through the external cell list
    for (int i = startCellId; i < endCellId; ++i) {
        int cid = externalCells[i];
        int cellId = offsets[cid];
        for (int vi = 1; vi <= dimension; ++vi) {
            int viId = connectivity[cellId + vi];
                // int localVjId = vjId - vertexIntervals[tid-1];
            if (viId >= startVertexId && viId < endVertexId) {
                int relativeVid = viId - baseVertexId + vertexOffset;
                int *arr = vertexStarMatrix + vtColumnNum * relativeVid;
                int localSize = vertexStarSizes[relativeVid];
                int oldVal = -1;
                do {
                    oldVal = atomicCAS(&arr[localSize++], -1, cid);
                } while (oldVal != -1);
                atomicAdd(&vertexStarSizes[relativeVid], 1);
            }
        }
    }
}
