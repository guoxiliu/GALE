#include <Gale.h>
#include <Relations.cuh>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <algorithm>
#include <iostream>

using namespace ttk;


Gale::Gale(int tnum, int tnumPerCluster) {
    this->setDebugMsgPrefix("Gale");
    threadNumber_ = tnum;
    threadsPerCluster_ = tnumPerCluster;
    currClusters_ = std::vector<SimplexId>(threadNumber_, 0);
    currRelations_ = std::vector<RelationType>(threadNumber_, RelationType::BlankRelation);
    // waitings_ = std::vector<int>(threadNumber_, 0);
    timers_ = std::vector<Timer>(threadNumber_);
    waitingTimes_ = std::vector<double>(threadNumber_, 0);
    waitingCounts_ = std::vector<int>(threadNumber_, 0);
    relationSems_ = std::vector<sem_t>(threadNumber_);
    for (int i = 0; i < 25; ++i) {
        baseClusters_[i] = std::vector<int>(threadNumber_, 0);
    }

    // GPU parameters
    numBlocks_ = 50, blockSize_ = 32;
}


Gale::~Gale() {
    if (producer_.joinable()) {
        currClusters_[0] = -1;
        producer_.join();
    }

    for (int i = 0; i < threadNumber_; ++i)
        std::cout << "waiting count " << i << ": " << waitingCounts_[i] << ", waiting time: " << waitingTimes_[i] << std::endl;

    // host memory deallocation
    if (hVertexIntervals_ != nullptr) free(hVertexIntervals_);
    if (hCellIntervals_ != nullptr) free(hCellIntervals_);
    if (hEdgeIntervals_ != nullptr) delete[] hEdgeIntervals_;
    if (hTriangleIntervals_ != nullptr) delete[] hTriangleIntervals_;
    if (hExternalCells_ != nullptr) delete[] hExternalCells_;
    if (hExternalCellOffsets_ != nullptr) delete[] hExternalCellOffsets_;

    // relation matrices on host and device memory get freed during the computation

    // device memory deallocation
    if (dIndexArray_ != nullptr) cudaFree(dIndexArray_);
    if (dConnectivity_ != nullptr) cudaFree(dConnectivity_);
    if (dOffsets_ != nullptr) cudaFree(dOffsets_);
    if (dVertexIntervals_ != nullptr) cudaFree(dVertexIntervals_);
    if (dCellIntervals_ != nullptr) cudaFree(dCellIntervals_);
    if (dExternalCells_ != nullptr) cudaFree(dExternalCells_);
    if (dExternalCellOffsets_ != nullptr) cudaFree(dExternalCellOffsets_);
    if (dEdgeIntervals_ != nullptr) cudaFree(dEdgeIntervals_);
    if (dTriangleIntervals_ != nullptr) cudaFree(dTriangleIntervals_);
    if (dBoundaryTriangles_ != nullptr) cudaFree(dBoundaryTriangles_);
}


int Gale::buildInternalEdgeList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 2>>> &clusterEdges, bool buildBoundary) {

    if(clusterId <= 0 || clusterId > clusterNumber_)
        return -1;

    SimplexId edgeCount = 0;
    // if we need to build the boundary edge list
    if(buildBoundary) {
        std::map<std::array<SimplexId, 2>, int> localEdgeMap;

        // loop through the internal cell list
        for(SimplexId cid = hCellIntervals_[clusterId - 1];
            cid < hCellIntervals_[clusterId]; cid++) {
            std::array<SimplexId, 2> edgeIds;

            // loop through each edge of the cell
            for(SimplexId j = 0; j < cellDimension_; j++) {
                edgeIds[0] = hConnectivity_[hOffsets_[cid] + j];
                // the edge does not belong to the current node
                if(edgeIds[0] < hVertexIntervals_[clusterId]) {
                    for(SimplexId k = j + 1; k < cellDimension_ + 1; k++) {
                        edgeIds[1] = hConnectivity_[hOffsets_[cid] + k];
                        localEdgeMap[edgeIds]++;
                    }
                }
            }
        }

        // loop through the external cell list
        for(int i = hExternalCellOffsets_[clusterId-1]; i < hExternalCellOffsets_[clusterId]; ++i) {
            SimplexId cellId = hOffsets_[hExternalCells_[i]];

            std::array<SimplexId, 2> edgeIds;

            // loop through each edge of the cell from cell[1]
            for(SimplexId j = 1; j < cellDimension_; j++) {
                for(SimplexId k = j + 1; k < cellDimension_ + 1; k++) {
                    edgeIds[0] = hConnectivity_[cellId + j];
                    edgeIds[1] = hConnectivity_[cellId + k];

                    // the edge is in the current node
                    if(edgeIds[0] >= hVertexIntervals_[clusterId - 1]
                        && edgeIds[0] < hVertexIntervals_[clusterId]) {
                        localEdgeMap[edgeIds]++;
                    }
                }
            }
        }

        // compute boundary edges
        edgeCount = localEdgeMap.size();
        std::vector<bool> localBoundEdges(edgeCount, false);
        int k = 0;
        for(auto &m : localEdgeMap) {
            clusterEdges[clusterId].push_back(std::move(m.first));
            if(m.second == 1) {
                localBoundEdges[k] = true;
            }
            k++;
        }
        // allClusters_[clusterId].boundaryEdges_[0] = std::move(localBoundEdges);
    }
    // otherwise, we just build the edge list
    else {
        std::set<std::array<SimplexId, 2>> localEdgeSet;

        // loop through the internal cell list
        for(SimplexId cid = hCellIntervals_[clusterId - 1];
            cid < hCellIntervals_[clusterId]; cid++) {
            std::array<SimplexId, 2> edgeIds;

            // loop through each edge of the cell
            for(SimplexId j = 0; j < cellDimension_; j++) {
                edgeIds[0] = hConnectivity_[hOffsets_[cid] + j];
                // the edge does not belong to the current node
                if(edgeIds[0] < hVertexIntervals_[clusterId]) {
                    for(SimplexId k = j + 1; k < cellDimension_ + 1; k++) {
                        edgeIds[1] = hConnectivity_[hOffsets_[cid] + k];

                        // not found in the edge map - assign new edge id
                        if(localEdgeSet.find(edgeIds) == localEdgeSet.end()) {
                            edgeCount++;
                            localEdgeSet.insert(edgeIds);
                        }
                    }
                }
            }
        }

        // loop through the external cell list
        for(int i = hExternalCellOffsets_[clusterId - 1]; i < hExternalCellOffsets_[clusterId]; ++i) {
            SimplexId cellId = hOffsets_[hExternalCells_[i]];
            std::array<SimplexId, 2> edgeIds;

            // loop through each edge of the cell from cell[1]
            for(SimplexId j = 1; j < cellDimension_; j++) {
                for(SimplexId k = j + 1; k < cellDimension_ + 1; k++) {
                    edgeIds[0] = hConnectivity_[cellId + j];
                    edgeIds[1] = hConnectivity_[cellId + k];

                    // the edge is in the current node
                    if(edgeIds[0] >= hVertexIntervals_[clusterId - 1]
                        && edgeIds[0] < hVertexIntervals_[clusterId]) {
                        if(localEdgeSet.find(edgeIds) == localEdgeSet.end()) {
                            edgeCount++;
                            localEdgeSet.insert(edgeIds);
                        }
                    }
                }
            }
        }

        clusterEdges[clusterId] = {localEdgeSet.begin(), localEdgeSet.end()};
    }
    return edgeCount;
}

int Gale::buildInternalTriangleList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 3>>> &clusterTriangles, bool buildBoundary) {

    if(clusterId <= 0 || clusterId > clusterNumber_)
        return -1;

    SimplexId triangleCount = 0;

    // if we need to build the boundary triangle list
    if(buildBoundary) {
        std::map<std::array<SimplexId, 3>, int> localTriangleMap;

        // loop through the internal cell list
        for(SimplexId cid = hCellIntervals_[clusterId - 1];
            cid < hCellIntervals_[clusterId]; cid++) {
            std::array<SimplexId, 3> triangleIds;

            // loop through each triangle of the cell
            for(SimplexId j = 0; j < cellDimension_ - 1; j++) {
                triangleIds[0] = hConnectivity_[hOffsets_[cid] + j];
                // the triangle does not belong to the current node
                if(triangleIds[0] < hVertexIntervals_[clusterId]) {
                    for(SimplexId k = j + 1; k < cellDimension_; k++) {
                        for(SimplexId l = k + 1; l < cellDimension_ + 1; l++) {
                            triangleIds[1] = hConnectivity_[hOffsets_[cid] + k];
                            triangleIds[2] = hConnectivity_[hOffsets_[cid] + l];
                            localTriangleMap[triangleIds]++;
                        }
                    }
                }
            }
        }

        // loop through the external cell list
        for(int i = hExternalCellOffsets_[clusterId-1]; i < hExternalCellOffsets_[clusterId]; ++i) {
            SimplexId cellId = hOffsets_[hExternalCells_[i]];
            // only check the cell[1,2,3] since cell[0] belongs to the previous cluster?
            std::array<SimplexId, 3> triangleIds;
            triangleIds[0] = hConnectivity_[cellId + 1];
            if(triangleIds[0] >= hVertexIntervals_[clusterId - 1]
                && triangleIds[0] < hVertexIntervals_[clusterId]) {
                triangleIds[1] = hConnectivity_[cellId + 2];
                triangleIds[2] = hConnectivity_[cellId + 3];
                localTriangleMap[triangleIds]++;
            }
        }

        // compute boundary triangles
        triangleCount = localTriangleMap.size();
        std::vector<bool> localBoundTriangles(triangleCount, false);
        int k = 0;
        for(auto &m : localTriangleMap) {
            clusterTriangles[clusterId].push_back(std::move(m.first));
            if(m.second == 1) {
                localBoundTriangles[k] = true;
            }
            k++;
        }
        // allClusters_[clusterId].boundaryTriangles_[0] = std::move(localBoundTriangles);
    }
    // otherwise, we just build the triangle list
    else {
        std::set<std::array<SimplexId, 3>> localTriangleSet;

        // loop through the internal cell list
        for(SimplexId cid = hCellIntervals_[clusterId - 1];
            cid < hCellIntervals_[clusterId]; cid++) {
            std::array<SimplexId, 3> triangleIds;

            // loop through each triangle of the cell
            for(SimplexId j = 0; j < cellDimension_ - 1; j++) {
                triangleIds[0] = hConnectivity_[hOffsets_[cid] + j];
                // the triangle does not belong to the current node
                if(triangleIds[0] < hVertexIntervals_[clusterId]) {
                    for(SimplexId k = j + 1; k < cellDimension_; k++) {
                        for(SimplexId l = k + 1; l < cellDimension_ + 1; l++) {
                            triangleIds[1] = hConnectivity_[hOffsets_[cid] + k];
                            triangleIds[2] = hConnectivity_[hOffsets_[cid] + l];

                            if(localTriangleSet.find(triangleIds) == localTriangleSet.end()) {
                                triangleCount++;
                                localTriangleSet.insert(triangleIds);
                            }
                        }
                    }
                }
            }
        }

        // loop through the external cell list
        for(int i = hExternalCellOffsets_[clusterId-1]; i < hExternalCellOffsets_[clusterId]; ++i) {
            SimplexId cellId = hOffsets_[hExternalCells_[i]];
            // only check the cell[1,2,3] since cell[0] belongs to the previous cluster?
            std::array<SimplexId, 3> triangleIds;
            triangleIds[0] = hConnectivity_[cellId + 1];
            if(triangleIds[0] >= hVertexIntervals_[clusterId - 1]
                && triangleIds[0] < hVertexIntervals_[clusterId]) {
                triangleIds[1] = hConnectivity_[cellId + 2];
                triangleIds[2] = hConnectivity_[cellId + 3];
                if(localTriangleSet.find(triangleIds) == localTriangleSet.end()) {
                    triangleCount++;
                    localTriangleSet.insert(triangleIds);
                }
            }
        }
        clusterTriangles[clusterId] = {localTriangleSet.begin(), localTriangleSet.end()};
    }
    return triangleCount;
}


// Compute relations when there is only one single consumer thread
// WARNING: will be deprecated later
int Gale::computeRelation(const RelationType relation,
                            const SimplexId clusterId,
                            const int threadId = 0,
                            bool signal = false) {

    // std::cout << "computing cluster " << clusterId << " relation " << static_cast<int>(relation) << std::endl;
    int res = 0;
    // int relationId = static_cast<int>(relation);
    // int totalBlocks = numBlocks_ * blockSize_;

    // switch (relation) {
    //     case RelationType::BERelation: 
    //     {
    //         int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localEdgeNum = hEdgeIntervals_[endCluster] - edgeIdOffset;
    //         bool *dBoundaryEdgeList;
    //         int *dEdgeList, *dTriangleList; 
    //         CHECK(cudaMalloc((void **)&dEdgeList, sizeof(SimplexId) * localEdgeNum * 2));
    //         CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3));
    //         CHECK(cudaMalloc((void **)&dBoundaryEdgeList, sizeof(bool) * localEdgeNum));
    //         CHECK(cudaMemcpy(dEdgeList, edgeList_.data() + edgeIdOffset, sizeof(SimplexId) * localEdgeNum * 2, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dBoundaryEdgeList, false, sizeof(bool) * localEdgeNum));

    //         computeBoundaryEdges<<<numBlocks_, blockSize_>>>(
    //             dEdgeList, dTriangleList, 
    //             dBoundaryEdgeList, dBoundaryTriangles_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_, dIndexArray_,
    //             dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         bool *hBoundaryEdgeList = new bool[localEdgeNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         CHECK(cudaMemcpy(hBoundaryEdgeList, dBoundaryEdgeList, sizeof(bool) * localEdgeNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(BEMutex_);
    //         for (int i = clusterId; i <= endCluster; ++i) {
    //             if (clusterBoundaryEdges_[i].empty()) {
    //                 clusterBoundaryEdges_[i].resize(hEdgeIntervals_[i] - hEdgeIntervals_[i-1]);
    //                 int idOffset = hEdgeIntervals_[i-1] - edgeIdOffset;
    //                 for (int j = 0; j < clusterBoundaryEdges_[i].size(); ++j) {
    //                     clusterBoundaryEdges_[i][j] = hBoundaryEdgeList[j + idOffset];
    //                 }
    //             }
    //         }
    //         baseClusters_[relationId][threadId] = clusterId;

    //         lck.unlock();
    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dEdgeList));
    //         CHECK(cudaFree(dTriangleList));
    //         CHECK(cudaFree(dBoundaryEdgeList));

    //         delete[] hBoundaryEdgeList;
    //         break;
    //     }

    //     case RelationType::BVRelation: 
    //     {
    //         int vertexIdOffset = hVertexIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
    //         bool *dBoundaryVertexList;
    //         int *dTriangleList; 
    //         CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3));
    //         CHECK(cudaMalloc((void **)&dBoundaryVertexList, sizeof(bool) * localVertexNum));
    //         CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dBoundaryVertexList, false, sizeof(bool) * localVertexNum));

    //         computeBoundaryVertices<<<numBlocks_, blockSize_>>>(
    //             dTriangleList, dBoundaryTriangles_, dBoundaryVertexList,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_, dIndexArray_,
    //             dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         bool *hBoundaryVertexList = new bool[localVertexNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         CHECK(cudaMemcpy(hBoundaryVertexList, dBoundaryVertexList, sizeof(bool) * localVertexNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(BVMutex_);
    //         for (int i = clusterId; i <= endCluster; ++i) {
    //             if (clusterBoundaryVertices_[i].empty()) {
    //                 clusterBoundaryVertices_[i].resize(hVertexIntervals_[i] - hVertexIntervals_[i-1]);
    //                 int idOffset = hVertexIntervals_[i-1] - vertexIdOffset;
    //                 for (int j = 0; j < clusterBoundaryVertices_[i].size(); ++j) {
    //                     clusterBoundaryVertices_[i][j] = hBoundaryVertexList[j + idOffset];
    //                 }
    //             }
    //         }
    //         baseClusters_[relationId][threadId] = clusterId;

    //         lck.unlock();
    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dTriangleList));
    //         CHECK(cudaFree(dBoundaryVertexList));

    //         delete[] hBoundaryVertexList;
    //         break;
    //     }

    //     case RelationType::BFRelation: 
    //     {
    //         int numBlocks = clusterNumber_ / (blockSize_ / threadsPerCluster_) + 1;
    //         int localTriangleNum = triangleNumber_;
    //         int *dTriangleList; 
    //         CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * localTriangleNum * 3));
    //         CHECK(cudaMalloc((void **)&dBoundaryTriangles_, sizeof(bool) * localTriangleNum));
    //         CHECK(cudaMalloc((void **)&dTriangleStarSizes_, sizeof(int) * localTriangleNum));
    //         CHECK(cudaMalloc((void **)&dTriangleStarMatrix_, sizeof(int) * localTriangleNum * ftColumnNum));
    //         CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * localTriangleNum * 3, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dBoundaryTriangles_, false, sizeof(bool) * localTriangleNum));
    //         CHECK(cudaMemset(dTriangleStarSizes_, 0, sizeof(int) * localTriangleNum));
    //         CHECK(cudaMemset(dTriangleStarMatrix_, -1, sizeof(int) * localTriangleNum * ftColumnNum));

    //         computeBoundaryTriangles<<<numBlocks, blockSize_>>>(
    //             dTriangleList, dTriangleStarMatrix_, 
    //             dTriangleStarSizes_, dBoundaryTriangles_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_,
    //             dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         bool *hBoundaryTriangleList = new bool[localTriangleNum];
    //         CHECK(cudaMemcpy(hBoundaryTriangleList, dBoundaryTriangles_, sizeof(bool) * localTriangleNum, cudaMemcpyDeviceToHost));
    //         for (int fid = 0; fid < localTriangleNum; ++fid) {
    //             boundaryTriangles_[fid] = hBoundaryTriangleList[fid];
    //         }

    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dTriangleList));
    //         CHECK(cudaFree(dTriangleStarSizes_));
    //         dTriangleStarSizes_ = nullptr;
    //         CHECK(cudaFree(dTriangleStarMatrix_));
    //         dTriangleStarMatrix_ = nullptr;
    //         delete[] hBoundaryTriangleList;

    //         break;
    //     }

    //     case RelationType::EFRelation:
    //     {
    //         int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localEdgeNum = hEdgeIntervals_[endCluster] - edgeIdOffset;
    //         int *dEdgeList, *dTriangleList; 
    //         CHECK(cudaMalloc((void **)&dEdgeList, sizeof(SimplexId) * localEdgeNum * 2));
    //         CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3));
    //         CHECK(cudaMalloc((void **)&dEdgeTriangleSizes_, sizeof(int) * localEdgeNum));
    //         CHECK(cudaMalloc((void **)&dEdgeTriangleMatrix_, sizeof(int) * localEdgeNum * efColumnNum));
    //         CHECK(cudaMemcpy(dEdgeList, edgeList_.data() + edgeIdOffset, sizeof(SimplexId) * localEdgeNum * 2, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dEdgeTriangleSizes_, 0, sizeof(int) * localEdgeNum));
    //         CHECK(cudaMemset(dEdgeTriangleMatrix_, -1, sizeof(int) * localEdgeNum * efColumnNum));

    //         computeEdgeTriangles<<<numBlocks_, blockSize_>>>(
    //             dEdgeList, dTriangleList,
    //             dEdgeTriangleMatrix_, dEdgeTriangleSizes_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_, dIndexArray_,
    //             dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         hEdgeTriangleSizes_ = new int[localEdgeNum];
    //         hEdgeTriangleMatrix_ = new int[localEdgeNum * efColumnNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         CHECK(cudaMemcpy(hEdgeTriangleSizes_, dEdgeTriangleSizes_, sizeof(int) * localEdgeNum, cudaMemcpyDeviceToHost));
    //         CHECK(cudaMemcpy(hEdgeTriangleMatrix_, dEdgeTriangleMatrix_, sizeof(int) * localEdgeNum * efColumnNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(EFMutex_);
    //         for (int i = 0; i < localEdgeNum; ++i) {
    //             SimplexId eid = edgeIdOffset + i;
    //             if (!edgeTriangleList_[eid].empty()) continue;
    //             edgeTriangleList_[eid] = std::vector<int>(hEdgeTriangleMatrix_ + i * efColumnNum, hEdgeTriangleMatrix_ + i * efColumnNum + hEdgeTriangleSizes_[i]);
    //         }
    //         baseClusters_[relationId][threadId] = clusterId;

    //         lck.unlock();
    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dEdgeList));
    //         CHECK(cudaFree(dTriangleList));
    //         CHECK(cudaFree(dEdgeTriangleSizes_));
    //         dEdgeTriangleSizes_ = nullptr;
    //         CHECK(cudaFree(dEdgeTriangleMatrix_));
    //         dEdgeTriangleMatrix_ = nullptr;

    //         delete[] hEdgeTriangleMatrix_;
    //         delete[] hEdgeTriangleSizes_;
    //         break;
    //     }

    //     case RelationType::ETRelation:
    //     {
    //         int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localEdgeNum = hEdgeIntervals_[endCluster] - edgeIdOffset;
    //         int *dEdgeList; 
    //         CHECK(cudaMalloc((void **)&dEdgeList, sizeof(SimplexId) * localEdgeNum * 2));
    //         CHECK(cudaMalloc((void **)&dEdgeStarSizes_, sizeof(int) * localEdgeNum));
    //         CHECK(cudaMalloc((void **)&dEdgeStarMatrix_, sizeof(int) * localEdgeNum * etColumnNum));
    //         CHECK(cudaMemcpy(dEdgeList, edgeList_.data() + edgeIdOffset, sizeof(SimplexId) * localEdgeNum * 2, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dEdgeStarSizes_, 0, sizeof(int) * localEdgeNum));
    //         CHECK(cudaMemset(dEdgeStarMatrix_, -1, sizeof(int) * localEdgeNum * etColumnNum));

    //         computeEdgeStars<<<numBlocks_, blockSize_>>>(
    //             dEdgeList, dEdgeStarMatrix_, dEdgeStarSizes_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_,
    //             dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         hEdgeStarSizes_ = new int[localEdgeNum];
    //         hEdgeStarMatrix_ = new int[localEdgeNum * etColumnNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         CHECK(cudaMemcpy(hEdgeStarSizes_, dEdgeStarSizes_, sizeof(int) * localEdgeNum, cudaMemcpyDeviceToHost));
    //         CHECK(cudaMemcpy(hEdgeStarMatrix_, dEdgeStarMatrix_, sizeof(int) * localEdgeNum * etColumnNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(ETMutex_);
    //         for (int i = 0; i < localEdgeNum; ++i) {
    //             SimplexId eid = edgeIdOffset + i;
    //             if (!edgeStarList_[eid].empty()) continue;
    //             edgeStarList_[eid] = std::vector<int>(hEdgeStarMatrix_ + i * etColumnNum, hEdgeStarMatrix_ + i * etColumnNum + hEdgeStarSizes_[i]);
    //         }
    //         baseClusters_[relationId][threadId] = clusterId;

    //         lck.unlock();
    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dEdgeList));
    //         CHECK(cudaFree(dEdgeStarSizes_));
    //         dEdgeStarSizes_ = nullptr;
    //         CHECK(cudaFree(dEdgeStarMatrix_));
    //         dEdgeStarMatrix_ = nullptr;

    //         delete[] hEdgeStarMatrix_;
    //         delete[] hEdgeStarSizes_;
    //         break;
    //     }

    //     case RelationType::FTRelation:
    //     {
    //         int triangleIdOffset = hTriangleIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localTriangleNum = hTriangleIntervals_[endCluster] - triangleIdOffset;
    //         int *dTriangleList; 
    //         CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * localTriangleNum * 3));
    //         CHECK(cudaMalloc((void **)&dTriangleStarSizes_, sizeof(int) * localTriangleNum));
    //         CHECK(cudaMalloc((void **)&dTriangleStarMatrix_, sizeof(int) * localTriangleNum * ftColumnNum));
    //         CHECK(cudaMemcpy(dTriangleList, triangleList_.data() + triangleIdOffset, sizeof(SimplexId) * localTriangleNum * 3, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dTriangleStarSizes_, 0, sizeof(int) * localTriangleNum));
    //         CHECK(cudaMemset(dTriangleStarMatrix_, -1, sizeof(int) * localTriangleNum * ftColumnNum));

    //         computeTriangleStars<<<numBlocks_, blockSize_>>>(
    //             dTriangleList, dTriangleStarMatrix_, dTriangleStarSizes_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_,
    //             dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         hTriangleStarSizes_ = new int[localTriangleNum];
    //         hTriangleStarMatrix_ = new int[localTriangleNum * ftColumnNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         CHECK(cudaMemcpy(hTriangleStarSizes_, dTriangleStarSizes_, sizeof(int) * localTriangleNum, cudaMemcpyDeviceToHost));
    //         CHECK(cudaMemcpy(hTriangleStarMatrix_, dTriangleStarMatrix_, sizeof(int) * localTriangleNum * ftColumnNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(FTMutex_);
    //         for (int i = 0; i < localTriangleNum; ++i) {
    //             SimplexId fid = triangleIdOffset + i;
    //             if (!triangleStarList_[fid].empty()) continue;
    //             triangleStarList_[fid] = std::vector<int>(hTriangleStarMatrix_ + i * ftColumnNum, hTriangleStarMatrix_ + i * ftColumnNum + hTriangleStarSizes_[i]);
    //         }
    //         baseClusters_[relationId][threadId] = clusterId;

    //         lck.unlock();
    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dTriangleList));
    //         CHECK(cudaFree(dTriangleStarSizes_));
    //         dTriangleStarSizes_ = nullptr;
    //         CHECK(cudaFree(dTriangleStarMatrix_));
    //         dTriangleStarMatrix_ = nullptr;

    //         delete[] hTriangleStarMatrix_;
    //         delete[] hTriangleStarSizes_;
    //         break;
    //     }

    //     case RelationType::VERelation:
    //     {
    //         int vertexIdOffset = hVertexIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[clusterId - 1];
    //         int *dEdgeList; 
    //         CHECK(cudaMalloc((void **)&dEdgeList, sizeof(SimplexId) * hEdgeIntervals_[endCluster] * 2));
    //         CHECK(cudaMalloc((void **)&dVertexEdgeSizes_, sizeof(int) * localVertexNum));
    //         CHECK(cudaMalloc((void **)&dVertexEdgeMatrix_, sizeof(int) * localVertexNum * veColumnNum));
    //         CHECK(cudaMemcpy(dEdgeList, edgeList_.data(), sizeof(SimplexId) * hEdgeIntervals_[endCluster] * 2, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dVertexEdgeSizes_, 0, sizeof(int) * localVertexNum));
    //         CHECK(cudaMemset(dVertexEdgeMatrix_, -1, sizeof(int) * localVertexNum * veColumnNum));

    //         computeVertexEdges<<<numBlocks_, blockSize_>>>(
    //             dEdgeList, dVertexEdgeMatrix_, dVertexEdgeSizes_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_, dIndexArray_,
    //             dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         hVertexEdgeSizes_ = new int[localVertexNum];
    //         hVertexEdgeMatrix_ = new int[localVertexNum * veColumnNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         CHECK(cudaMemcpy(hVertexEdgeSizes_, dVertexEdgeSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
    //         CHECK(cudaMemcpy(hVertexEdgeMatrix_, dVertexEdgeMatrix_, sizeof(int) * localVertexNum * veColumnNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(VEMutex_);
    //         for (int i = 0; i < localVertexNum; ++i) {
    //             SimplexId vid = vertexIdOffset + i;
    //             if (!vertexEdgeList_[vid].empty()) continue;
    //             vertexEdgeList_[vid] = std::vector<int>(hVertexEdgeMatrix_ + i * veColumnNum, hVertexEdgeMatrix_ + i * veColumnNum + hVertexEdgeSizes_[i]);
    //         }
    //         baseClusters_[relationId][threadId] = clusterId;

    //         lck.unlock();
    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dEdgeList));
    //         CHECK(cudaFree(dVertexEdgeSizes_));
    //         dVertexEdgeSizes_ = nullptr;
    //         CHECK(cudaFree(dVertexEdgeMatrix_));
    //         dVertexEdgeMatrix_ = nullptr;

    //         delete[] hVertexEdgeMatrix_;
    //         delete[] hVertexEdgeSizes_;
    //         break;
    //     }

    //     case RelationType::VFRelation:
    //     {
    //         int vertexIdOffset = hVertexIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[clusterId - 1];
    //         int *dTriangleList; 
    //         CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3));
    //         CHECK(cudaMalloc((void **)&dVertexTriangleSizes_, sizeof(int) * localVertexNum));
    //         CHECK(cudaMalloc((void **)&dVertexTriangleMatrix_, sizeof(int) * localVertexNum * vfColumnNum));
    //         CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3, cudaMemcpyHostToDevice));
    //         CHECK(cudaMemset(dVertexTriangleSizes_, 0, sizeof(int) * localVertexNum));
    //         CHECK(cudaMemset(dVertexTriangleMatrix_, -1, sizeof(int) * localVertexNum * vfColumnNum));

    //         computeVertexTriangles<<<numBlocks_, blockSize_>>>(
    //             dTriangleList, dVertexTriangleMatrix_, dVertexTriangleSizes_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_, dIndexArray_,
    //             dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         hVertexTriangleSizes_ = new int[localVertexNum];
    //         hVertexTriangleMatrix_ = new int[localVertexNum * vfColumnNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__);
    //             if (abort)
    //                 exit(status);
    //         }

    //         CHECK(cudaMemcpy(hVertexTriangleSizes_, dVertexTriangleSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
    //         CHECK(cudaMemcpy(hVertexTriangleMatrix_, dVertexTriangleMatrix_, sizeof(int) * localVertexNum * vfColumnNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(VEMutex_);
    //         for (int i = 0; i < localVertexNum; ++i) {
    //             SimplexId vid = vertexIdOffset + i;
    //             if (!vertexTriangleList_[vid].empty()) continue;
    //             vertexTriangleList_[vid] = std::vector<int>(hVertexTriangleMatrix_ + i * vfColumnNum, hVertexTriangleMatrix_ + i * vfColumnNum + hVertexTriangleSizes_[i]);
    //         }
    //         baseClusters_[relationId][threadId] = clusterId;

    //         lck.unlock();
    //         if (signal) {
    //             // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //             // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dTriangleList));
    //         CHECK(cudaFree(dVertexTriangleSizes_));
    //         dVertexTriangleSizes_ = nullptr;
    //         CHECK(cudaFree(dVertexTriangleMatrix_));
    //         dVertexTriangleMatrix_ = nullptr;

    //         delete[] hVertexTriangleMatrix_;
    //         delete[] hVertexTriangleSizes_;
    //         break;
    //     }

    //     case RelationType::VVRelation:
    //     {
    //         int vertexIdOffset = hVertexIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[clusterId - 1];
    //         // std::cout << "endVertex: " << hVertexIntervals_[endCluster] << ", startVertex: " << hVertexIntervals_[clusterId - 1] << ", localVertexNum: " << localVertexNum << std::endl;
    //         CHECK(cudaMalloc((void **)&dVertexNeighborSizes_, sizeof(int) * localVertexNum));
    //         CHECK(cudaMalloc((void **)&dVertexNeighborMatrix_, sizeof(int) * localVertexNum * vvColumnNum));
    //         CHECK(cudaMemset(dVertexNeighborSizes_, 0, sizeof(int) * localVertexNum));
    //         CHECK(cudaMemset(dVertexNeighborMatrix_, -1, sizeof(int) * localVertexNum * vvColumnNum));
            
    //         computeVertexNeighbors<<<numBlocks_, blockSize_>>>(
    //             dVertexNeighborMatrix_, dVertexNeighborSizes_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_,
    //             dVertexIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         hVertexNeighborSizes_ = new int[localVertexNum];
    //         hVertexNeighborMatrix_ = new int[localVertexNum * vvColumnNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //             if (status != cudaSuccess) {
    //                 fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                     __FILE__, __LINE__); if (abort) exit(status);
    //         }

    //         CHECK(cudaMemcpy(hVertexNeighborSizes_, dVertexNeighborSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
    //         CHECK(cudaMemcpy(hVertexNeighborMatrix_, dVertexNeighborMatrix_, sizeof(int) * localVertexNum * vvColumnNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(VVMutex_);
    //         for (int i = 0; i < localVertexNum; ++i) {
    //             SimplexId vid = vertexIdOffset + i;
    //             if (!vertexNeighborList_[vid].empty()) continue;
    //             vertexNeighborList_[vid] = std::vector<int>(hVertexNeighborMatrix_ + i * vvColumnNum, hVertexNeighborMatrix_ + i * vvColumnNum + hVertexNeighborSizes_[i]);
    //         }
    //         lck.unlock();
    //         baseClusters_[relationId][threadId] = clusterId;

    //         if (signal) {
    //         // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
    //         // else res = 1;
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dVertexNeighborSizes_)); dVertexNeighborSizes_ = nullptr;
    //         CHECK(cudaFree(dVertexNeighborMatrix_)); dVertexNeighborMatrix_ = nullptr;

    //         delete[] hVertexNeighborMatrix_;
    //         delete[] hVertexNeighborSizes_;
    //         break;
    //     }

    //     case RelationType::VTRelation:
    //     {
    //         int vertexIdOffset = hVertexIntervals_[clusterId - 1];
    //         int endCluster = std::min(clusterId + totalBlocks / threadsPerCluster_, clusterNumber_) - 1;
    //         int localVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[clusterId - 1];
    //         baseClusters_[static_cast<int>(RelationType::VTRelation)][threadId] = clusterId;
    //         CHECK(cudaMalloc((void **)&dVertexStarSizes_, sizeof(int) * localVertexNum));
    //         CHECK(cudaMalloc((void **)&dVertexStarMatrix_, sizeof(int) * localVertexNum * vtColumnNum));
    //         CHECK(cudaMemset(dVertexStarSizes_, 0, sizeof(int) * localVertexNum));
    //         CHECK(cudaMemset(dVertexStarMatrix_, -1, sizeof(int) * localVertexNum * vtColumnNum));

    //         computeVertexStars<<<numBlocks_, blockSize_>>>(
    //             dVertexStarMatrix_, dVertexStarSizes_,
    //             clusterId, clusterNumber_,
    //             dConnectivity_, dOffsets_,
    //             dVertexIntervals_, dCellIntervals_,
    //             dExternalCells_, dExternalCellOffsets_, 
    //             cellDimension_, threadsPerCluster_);

    //         hVertexStarSizes_ = new int[localVertexNum];
    //         hVertexStarMatrix_ = new int[localVertexNum * vtColumnNum];

    //         cudaError_t status = cudaDeviceSynchronize();
    //         if (status != cudaSuccess) {
    //             fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //                 __FILE__, __LINE__); if (abort) exit(status);
    //         }

    //         CHECK(cudaMemcpy(hVertexStarSizes_, dVertexStarSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
    //         CHECK(cudaMemcpy(hVertexStarMatrix_, dVertexStarMatrix_, sizeof(int) * localVertexNum * vtColumnNum, cudaMemcpyDeviceToHost));

    //         std::unique_lock<std::mutex> lck(VTMutex_);
    //         for (int i = 0; i < localVertexNum; ++i) {
    //             SimplexId vid = vertexIdOffset + i;
    //             if (!vertexStarList_[vid].empty()) continue;
    //             vertexStarList_[vid] = std::vector<int>(hVertexStarMatrix_ + i * vtColumnNum, hVertexStarMatrix_ + i * vtColumnNum + hVertexStarSizes_[i]);
    //         }
    //         lck.unlock();
    //         baseClusters_[relationId][threadId] = clusterId;
    //         if (signal) {
    //             sem_post(&relationSems_[threadId]);
    //         }

    //         CHECK(cudaFree(dVertexStarSizes_)); dVertexStarSizes_ = nullptr;
    //         CHECK(cudaFree(dVertexStarMatrix_)); dVertexStarMatrix_ = nullptr;

    //         delete[] hVertexStarMatrix_;
    //         delete[] hVertexStarSizes_;
    //         break;
    //     }

    //     default:
    //         res = -1;
    // }

    return res;
}


int Gale::computeCombinedRelation(std::map<RelationType, 
            std::vector<std::pair<int,int>>> &requestMap, 
            std::vector<RelationType> &relationOrders) {

    // std::cout << "request size: " << requestMap.size() << ": [";
    // for (auto &e : requestMap) {
    //     std::cout << static_cast<int>(e.first) << "(" << jobs.size() << ") ";
    // }
    // std::cout << "]" << std::endl;

    int res = 0;
    int numRelations = relationOrders.size();
    std::vector<cudaStream_t> threadStreams(numRelations);
    std::vector<std::vector<SimplexId>> hEndClusters(numRelations);
    std::vector<std::vector<int>> hSimplexOffsets(numRelations);
    std::vector<int *> dEdgeLists(numRelations);
    std::vector<int *> dTriangleLists(numRelations);
    std::vector<int *> dClustersIds(numRelations);
    std::vector<int *> dSimplexOffsets(numRelations);

    for (int curStream = 0; curStream < numRelations; ++curStream) {
        // (relation, [(tid, clusterId), ...])
        RelationType &relation = relationOrders[curStream];
        CHECK(cudaStreamCreate(&threadStreams[curStream]));
        int relationId = static_cast<int>(relation);
        std::vector<std::pair<int,int>> &jobs = requestMap[relation];

        int threadNum = jobs.size();
        int gpuBlocks = numBlocks_ / threadNum;
        int gpuThreads = blockSize_ / threadsPerCluster_ * gpuBlocks;

        // launches different kernels on different streams
        switch (relation) {
            case RelationType::BERelation:
            {
                int localEdgeSum = 0;
                int hEdgeOffsets[threadNum];
                int hClusterIds[threadNum];
                int hEndClusters[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hEdgeOffsets[i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                    int localEdgeNum = hEdgeIntervals_[hEndClusters[i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNum;
                }
                bool *dBoundaryEdgeList;
                int *dEdgeList, *dTriangleList, *dClusterIds, *dEdgeOffsets; 
                CHECK(cudaMalloc((void **)&dEdgeList, sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2));
                CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3));
                CHECK(cudaMalloc((void **)&dClusterIds, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dEdgeOffsets, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dBoundaryEdgeList, sizeof(bool) * localEdgeSum));
                CHECK(cudaMemcpy(dEdgeList, edgeList_.data(), sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dEdgeOffsets, hEdgeOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemset(dBoundaryEdgeList, false, sizeof(bool) * localEdgeSum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiBoundaryEdges<<<dimBlock, blockSize_>>>(
                    dEdgeList, dTriangleList,
                    dBoundaryEdgeList, dBoundaryTriangles_,
                    clusterNumber_, dClusterIds, dEdgeOffsets,
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                bool *hBoundaryEdgeList = new bool[localEdgeSum];

                cudaError_t status = cudaDeviceSynchronize();
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

                CHECK(cudaMemcpy(hBoundaryEdgeList, dBoundaryEdgeList, sizeof(bool) * localEdgeSum, cudaMemcpyDeviceToHost));

                for (int i = 0; i < threadNum; ++i) {
                    for (int j = hClusterIds[i]; j <= hEndClusters[i]; ++j) {
                        if (!allClusters_[j].boundaryEdges_[jobs[i].first].empty()) continue;
                        int localEdgeNum = hEdgeIntervals_[j] - hEdgeIntervals_[j-1];
                        int relativeEid= hEdgeIntervals_[j - 1] - hEdgeIntervals_[hClusterIds[i] - 1] + hEdgeOffsets[i];
                        for (int vid = 0; vid < localEdgeNum; ++vid) {
                            allClusters_[j].boundaryEdges_[jobs[i].first].push_back(hBoundaryEdgeList[relativeEid++]);
                        }
                    }
                    if (baseClusters_[relationId][jobs[i].first] == 0)
                        baseClusters_[relationId][jobs[i].first] = hClusterIds[i];
                    sem_post(&relationSems_[jobs[i].first]);
                }

                CHECK(cudaFree(dEdgeList));
                CHECK(cudaFree(dTriangleList));
                CHECK(cudaFree(dClusterIds));
                CHECK(cudaFree(dEdgeOffsets));
                CHECK(cudaFree(dBoundaryEdgeList));

                delete[] hBoundaryEdgeList;
                break;
            }

            case RelationType::BVRelation: 
            {
                int localVertexSum = 0;
                int hVertexOffsets[threadNum];
                int hClusterIds[threadNum];
                int hEndClusters[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hVertexOffsets[i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                    int localVertexNum = hVertexIntervals_[hEndClusters[i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                bool *dBoundaryVertexList;
                int *dTriangleList, *dClusterIds, *dVertexOffsets;
                CHECK(cudaMalloc((void **)&dClusterIds, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dVertexOffsets, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3));
                CHECK(cudaMalloc((void **)&dBoundaryVertexList, sizeof(bool) * localVertexSum));
                CHECK(cudaMemcpy(dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dVertexOffsets, hVertexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3, cudaMemcpyHostToDevice));
                CHECK(cudaMemset(dBoundaryVertexList, false, sizeof(bool) * localVertexSum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiBoundaryVertices<<<dimBlock, blockSize_>>>(
                    dTriangleList, dBoundaryTriangles_, dBoundaryVertexList,
                    clusterNumber_, dClusterIds, dVertexOffsets, 
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                bool *hBoundaryVertexList = new bool[localVertexSum];

                cudaError_t status = cudaDeviceSynchronize();
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

                CHECK(cudaMemcpy(hBoundaryVertexList, dBoundaryVertexList, sizeof(bool) * localVertexSum, cudaMemcpyDeviceToHost));

                
                for (int i = 0; i < threadNum; ++i) {
                    // might be useless if only one thread can write to the vector?
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    for (int j = hClusterIds[i]; j <= hEndClusters[i]; ++j) {
                        if (!allClusters_[j].boundaryVertices_[jobs[i].first].empty()) continue;
                        int localVertexNum = hVertexIntervals_[j] - hVertexIntervals_[j-1];
                        int relativeVid = hVertexIntervals_[j - 1] - hVertexIntervals_[hClusterIds[i] - 1] + hVertexOffsets[i];
                        for (int vid = 0; vid < localVertexNum; ++vid) {
                            allClusters_[j].boundaryVertices_[jobs[i].first].push_back(hBoundaryVertexList[relativeVid++]);
                        }
                    }
                    if (baseClusters_[relationId][jobs[i].first] == 0)
                        baseClusters_[relationId][jobs[i].first] = hClusterIds[i];
                    // lck.unlock();
                    sem_post(&relationSems_[jobs[i].first]);
                }

                CHECK(cudaFree(dClusterIds));
                CHECK(cudaFree(dVertexOffsets));
                CHECK(cudaFree(dTriangleList));
                CHECK(cudaFree(dBoundaryVertexList));

                delete[] hBoundaryVertexList;
                break;
            }
            
            case RelationType::EFRelation:
            {
                int localEdgeSum = 0;
                int localEdgeNums[threadNum];
                int hEdgeOffsets[threadNum];
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hEdgeOffsets[i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    int endCluster = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(endCluster, maxEndCluster);
                    localEdgeNums[i] = hEdgeIntervals_[endCluster] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNums[i];
                }
                int *dEdgeList, *dTriangleList, *dClusterIds, *dEdgeOffsets; 
                CHECK(cudaMalloc((void **)&dEdgeList, sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2));
                CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3));
                CHECK(cudaMalloc((void **)&dClusterIds, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dEdgeOffsets, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dEdgeTriangleSizes_, sizeof(int) * localEdgeSum));
                CHECK(cudaMalloc((void **)&dEdgeTriangleMatrix_, sizeof(int) * localEdgeSum * efColumnNum));
                CHECK(cudaMemcpy(dEdgeList, edgeList_.data(), sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dEdgeOffsets, hEdgeOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemset(dEdgeTriangleSizes_, 0, sizeof(int) * localEdgeSum));
                CHECK(cudaMemset(dEdgeTriangleMatrix_, -1, sizeof(int) * localEdgeSum * efColumnNum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiEdgeTriangles<<<dimBlock, blockSize_>>>(
                    dEdgeList, dTriangleList,
                    dEdgeTriangleMatrix_, dEdgeTriangleSizes_,
                    clusterNumber_, dClusterIds, dEdgeOffsets,
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hEdgeTriangleSizes_ = new int[localEdgeSum];
                hEdgeTriangleMatrix_ = new int[localEdgeSum * efColumnNum];

                cudaError_t status = cudaDeviceSynchronize();
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

                CHECK(cudaMemcpy(hEdgeTriangleSizes_, dEdgeTriangleSizes_, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(hEdgeTriangleMatrix_, dEdgeTriangleMatrix_, sizeof(int) * localEdgeSum * efColumnNum, cudaMemcpyDeviceToHost));

                for (int i = 0; i < threadNum; ++i) {
                    for (int j = 0; j < localEdgeNums[i]; ++j) {
                        SimplexId eid = hEdgeIntervals_[hClusterIds[i]-1] + j;
                        if (!edgeTriangleList_[eid].empty()) continue;
                        int rj = j + hEdgeOffsets[i];   // relative j value in the large matrix
                        edgeTriangleList_[eid] = std::vector<int>(hEdgeTriangleMatrix_ + rj * efColumnNum, hEdgeTriangleMatrix_ + rj * efColumnNum + hEdgeTriangleSizes_[rj]);
                    }
                    if (baseClusters_[relationId][jobs[i].first] == 0) 
                        baseClusters_[relationId][jobs[i].first] = hClusterIds[i];
                    sem_post(&relationSems_[jobs[i].first]);
                }

                CHECK(cudaFree(dEdgeList));
                CHECK(cudaFree(dTriangleList));
                CHECK(cudaFree(dClusterIds));
                CHECK(cudaFree(dEdgeOffsets));
                CHECK(cudaFree(dEdgeTriangleSizes_));
                dEdgeTriangleSizes_ = nullptr;
                CHECK(cudaFree(dEdgeTriangleMatrix_));
                dEdgeTriangleMatrix_ = nullptr;

                delete[] hEdgeTriangleMatrix_;
                delete[] hEdgeTriangleSizes_;
                break;
            }

            case RelationType::ETRelation:
            {
                int localEdgeSum = 0;
                int localEdgeNums[threadNum];
                int hEdgeOffsets[threadNum];
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hEdgeOffsets[i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    int endCluster = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(endCluster, maxEndCluster);
                    localEdgeNums[i] = hEdgeIntervals_[endCluster] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNums[i];
                }
                int *dEdgeList, *dClusterIds, *dEdgeOffsets; 
                CHECK(cudaMalloc((void **)&dClusterIds, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dEdgeOffsets, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dEdgeList, sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2));
                CHECK(cudaMalloc((void **)&dEdgeStarSizes_, sizeof(int) * localEdgeSum));
                CHECK(cudaMalloc((void **)&dEdgeStarMatrix_, sizeof(int) * localEdgeSum * etColumnNum));

                CHECK(cudaMemcpy(dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dEdgeOffsets, hEdgeOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dEdgeList, edgeList_.data(), sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2, cudaMemcpyHostToDevice));
                CHECK(cudaMemset(dEdgeStarSizes_, 0, sizeof(int) * localEdgeSum));
                CHECK(cudaMemset(dEdgeStarMatrix_, -1, sizeof(int) * localEdgeSum * etColumnNum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiEdgeStars<<<dimBlock, blockSize_>>>(
                    dEdgeList, dEdgeStarMatrix_, dEdgeStarSizes_,
                    clusterNumber_, dClusterIds, dEdgeOffsets, 
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hEdgeStarSizes_ = new int[localEdgeSum];
                hEdgeStarMatrix_ = new int[localEdgeSum * etColumnNum];

                cudaError_t status = cudaDeviceSynchronize();
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

                CHECK(cudaMemcpy(hEdgeStarSizes_, dEdgeStarSizes_, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(hEdgeStarMatrix_, dEdgeStarMatrix_, sizeof(int) * localEdgeSum * etColumnNum, cudaMemcpyDeviceToHost));
                
                for (int i = 0; i < threadNum; ++i) {
                    for (int j = 0; j < localEdgeNums[i]; ++j) {
                        SimplexId eid = hEdgeIntervals_[hClusterIds[i]-1] + j;
                        if (!edgeStarList_[eid].empty()) continue;
                        int rj = j + hEdgeOffsets[i];   // relative j value in the large matrix
                        edgeStarList_[eid] = std::vector<int>(hEdgeStarMatrix_ + rj * efColumnNum, hEdgeStarMatrix_ + rj * efColumnNum + hEdgeStarSizes_[rj]);
                    }
                    if (baseClusters_[relationId][jobs[i].first] == 0) 
                        baseClusters_[relationId][jobs[i].first] = hClusterIds[i];
                    sem_post(&relationSems_[jobs[i].first]);
                }

                CHECK(cudaFree(dEdgeList));
                CHECK(cudaFree(dClusterIds));
                CHECK(cudaFree(dEdgeOffsets));
                CHECK(cudaFree(dEdgeStarSizes_));
                dEdgeStarSizes_ = nullptr;
                CHECK(cudaFree(dEdgeStarMatrix_));
                dEdgeStarMatrix_ = nullptr;

                delete[] hEdgeStarMatrix_;
                delete[] hEdgeStarSizes_;
                break;
            }

            case RelationType::FTRelation:
            {
                int localTriangleSum = 0;
                int localTriangleNums[threadNum];
                int hTriangleOffsets[threadNum];
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hTriangleOffsets[i] = localTriangleSum;
                    hClusterIds[i] = jobs[i].second;
                    int endCluster = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(endCluster, maxEndCluster);
                    localTriangleNums[i] = hTriangleIntervals_[endCluster] - hTriangleIntervals_[hClusterIds[i] - 1];
                    localTriangleSum += localTriangleNums[i];
                }
                int *dTriangleList, *dClusterIds, *dTriangleOffsets; 
                CHECK(cudaMalloc((void **)&dClusterIds, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dTriangleOffsets, sizeof(int) * threadNum));
                CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3));
                CHECK(cudaMalloc((void **)&dTriangleStarSizes_, sizeof(int) * localTriangleSum));
                CHECK(cudaMalloc((void **)&dTriangleStarMatrix_, sizeof(int) * localTriangleSum * ftColumnNum));

                CHECK(cudaMemcpy(dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dTriangleOffsets, hTriangleOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3, cudaMemcpyHostToDevice));
                CHECK(cudaMemset(dTriangleStarSizes_, 0, sizeof(int) * localTriangleSum));
                CHECK(cudaMemset(dTriangleStarMatrix_, -1, sizeof(int) * localTriangleSum * ftColumnNum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiTriangleStars<<<dimBlock, blockSize_>>>(
                    dTriangleList, dTriangleStarMatrix_, dTriangleStarSizes_,
                    clusterNumber_, dClusterIds, dTriangleOffsets,
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hTriangleStarSizes_ = new int[localTriangleSum];
                hTriangleStarMatrix_ = new int[localTriangleSum * ftColumnNum];

                cudaError_t status = cudaDeviceSynchronize();
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

                CHECK(cudaMemcpy(hTriangleStarSizes_, dTriangleStarSizes_, sizeof(int) * localTriangleSum, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(hTriangleStarMatrix_, dTriangleStarMatrix_, sizeof(int) * localTriangleSum * ftColumnNum, cudaMemcpyDeviceToHost));

                for (int i = 0; i < threadNum; ++i) {
                    for (int j = 0; j < localTriangleNums[i]; ++j) {
                        SimplexId fid = hTriangleIntervals_[hClusterIds[i]-1] + j;
                        if (!triangleStarList_[fid].empty()) continue;
                        int rj = j + hTriangleOffsets[i];   // relative j value in the large matrix
                        triangleStarList_[fid] = std::vector<int>(hTriangleStarMatrix_ + rj * ftColumnNum, hTriangleStarMatrix_ + rj * ftColumnNum + hTriangleStarSizes_[rj]);
                    }
                    if (baseClusters_[relationId][jobs[i].first] == 0) 
                        baseClusters_[relationId][jobs[i].first] = hClusterIds[i];
                    sem_post(&relationSems_[jobs[i].first]);
                }

                CHECK(cudaFree(dClusterIds));
                CHECK(cudaFree(dTriangleOffsets));
                CHECK(cudaFree(dTriangleList));
                CHECK(cudaFree(dTriangleStarSizes_));
                dTriangleStarSizes_ = nullptr;
                CHECK(cudaFree(dTriangleStarMatrix_));
                dTriangleStarMatrix_ = nullptr;

                delete[] hTriangleStarMatrix_;
                delete[] hTriangleStarSizes_;
                break;
            }

            case RelationType::VERelation:
            {
                int localVertexSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMallocAsync((void **)&dEdgeLists[curStream], sizeof(int) * hEdgeIntervals_[maxEndCluster] * 2, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexEdgeSizes_, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexEdgeMatrix_, sizeof(int) * localVertexSum * veColumnNum, threadStreams[curStream]));

                CHECK(cudaMemcpyAsync(dEdgeLists[curStream], edgeList_.data(), sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexEdgeSizes_, 0, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexEdgeMatrix_, -1, sizeof(int) * localVertexSum * veColumnNum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiVertexEdges<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dEdgeLists[curStream], dVertexEdgeMatrix_, dVertexEdgeSizes_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream],
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hVertexEdgeSizes_ = new int[localVertexSum];
                hVertexEdgeMatrix_ = new int[localVertexSum * veColumnNum];

                CHECK(cudaMemcpyAsync(hVertexEdgeSizes_, dVertexEdgeSizes_, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(hVertexEdgeMatrix_, dVertexEdgeMatrix_, sizeof(int) * localVertexSum * veColumnNum, cudaMemcpyDeviceToHost, threadStreams[curStream]));

                break;
            }

            case RelationType::VFRelation:
            {
                int localVertexSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dTriangleLists[curStream], sizeof(int) * hTriangleIntervals_[maxEndCluster] * 3, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexTriangleSizes_, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexTriangleMatrix_, sizeof(int) * localVertexSum * vfColumnNum, threadStreams[curStream]));

                CHECK(cudaMemcpyAsync(dTriangleLists[curStream], triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexTriangleSizes_, 0, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexTriangleMatrix_, -1, sizeof(int) * localVertexSum * vfColumnNum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiVertexTriangles<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dTriangleLists[curStream], dVertexTriangleMatrix_, dVertexTriangleSizes_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream],
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hVertexTriangleSizes_ = new int [localVertexSum];
                hVertexTriangleMatrix_ = new int [localVertexSum * vfColumnNum];

                CHECK(cudaMemcpyAsync(hVertexTriangleSizes_, dVertexTriangleSizes_, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(hVertexTriangleMatrix_, dVertexTriangleMatrix_, sizeof(int) * localVertexSum * vfColumnNum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                
                break;
            }

            case RelationType::VTRelation:
            {
                int localVertexSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1 ;
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexStarSizes_, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexStarMatrix_, sizeof(int) * localVertexSum * vtColumnNum, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexStarSizes_, 0, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexStarMatrix_, -1, sizeof(int) * localVertexSum * vtColumnNum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiVertexStars<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dVertexStarMatrix_, dVertexStarSizes_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream],
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hVertexStarSizes_ = new int[localVertexSum];
                hVertexStarMatrix_ = new int[localVertexSum * vtColumnNum];

                CHECK(cudaMemcpyAsync(hVertexStarSizes_, dVertexStarSizes_, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(hVertexStarMatrix_, dVertexStarMatrix_, sizeof(int) * localVertexSum * vtColumnNum, cudaMemcpyDeviceToHost, threadStreams[curStream]));

                break;
            }

            case RelationType::VVRelation:
            {
                int localVertexSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexNeighborSizes_, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexNeighborMatrix_, sizeof(int) * localVertexSum * vvColumnNum, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemset(dVertexNeighborSizes_, 0, sizeof(int) * localVertexSum));
                CHECK(cudaMemset(dVertexNeighborMatrix_, -1, sizeof(int) * localVertexSum * vvColumnNum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiVertexNeighbors<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dVertexNeighborMatrix_, dVertexNeighborSizes_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream],
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hVertexNeighborSizes_ = new int[localVertexSum];
                hVertexNeighborMatrix_ = new int[localVertexSum * vvColumnNum];

                CHECK(cudaMemcpyAsync(hVertexNeighborSizes_, dVertexNeighborSizes_, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(hVertexNeighborMatrix_, dVertexNeighborMatrix_, sizeof(int) * localVertexSum * vvColumnNum, cudaMemcpyDeviceToHost, threadStreams[curStream]));

                break;
            }

            default:
                res = -1;
        }
    }

    // Get the relation arrays from the GPU
    for (int curStream = 0; curStream < numRelations; ++curStream) {
        // (relation, [(tid, clusterId), ...])
        RelationType &relation = relationOrders[curStream];
        int relationId = static_cast<int>(relation);
        std::vector<std::pair<int,int>> &jobs = requestMap[relation];
        int threadNum = jobs.size();

        // launches different kernels on different streams
        switch (relation) {
            case RelationType::VERelation:
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                        __FILE__, __LINE__); if (abort) exit(status);
                }

                for (int i = 0; i < threadNum; ++i) {
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].vertexEdges_[tid].empty()) {
                            std::vector<SimplexId> vertexEdgeData{};
                            int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                            std::vector<SimplexId> vertexEdgeOffsets(localVertexNum + 1);
                            vertexEdgeOffsets[0] = 0;
                            for (int j = 0; j < localVertexNum; ++j) {
                                int rj = j + diff; // relative j value in the large matrix
                                vertexEdgeData.insert(vertexEdgeData.end(), 
                                hVertexEdgeMatrix_ + rj * veColumnNum,
                                hVertexEdgeMatrix_ + rj * veColumnNum + hVertexEdgeSizes_[rj]);
                                vertexEdgeOffsets[j+1] = vertexEdgeOffsets[j] + hVertexEdgeSizes_[rj];
                            }
                            allClusters_[sid].vertexEdges_[tid].setData(std::move(vertexEdgeData), std::move(vertexEdgeOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dEdgeLists[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dVertexEdgeSizes_));
                CHECK(cudaFree(dVertexEdgeMatrix_));
                delete[] hVertexEdgeSizes_;
                delete[] hVertexEdgeMatrix_;

                break;
            }
            
            case RelationType::VFRelation:
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                        __FILE__, __LINE__); if (abort) exit(status);
                }

                for (int i = 0; i < threadNum; ++i) {
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].vertexTriangles_[tid].empty()) {
                            std::vector<SimplexId> vertexTriangleData{};
                            int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                            std::vector<SimplexId> vertexTriangleOffsets(localVertexNum + 1);
                            vertexTriangleOffsets[0] = 0;
                            for (int j = 0; j < localVertexNum; ++j) {
                                int rj = j + diff; // relative j value in the large matrix
                                vertexTriangleData.insert(vertexTriangleData.end(), 
                                hVertexTriangleMatrix_ + rj * vfColumnNum,
                                hVertexTriangleMatrix_ + rj * vfColumnNum + hVertexTriangleSizes_[rj]);
                                vertexTriangleOffsets[j+1] = vertexTriangleOffsets[j] + hVertexTriangleSizes_[rj];
                            }
                            allClusters_[sid].vertexTriangles_[tid].setData(std::move(vertexTriangleData), std::move(vertexTriangleOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dTriangleLists[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dVertexTriangleSizes_));
                CHECK(cudaFree(dVertexTriangleMatrix_));
                delete[] hVertexTriangleMatrix_;
                delete[] hVertexTriangleSizes_;

                break;
            }
            
            case RelationType::VTRelation:
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                        __FILE__, __LINE__); if (abort) exit(status);
                }

                for (int i = 0; i < threadNum; ++i) {
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].vertexStars_[tid].empty()) {
                            std::vector<SimplexId> vertexStarData{};
                            int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                            std::vector<SimplexId> vertexStarOffsets(localVertexNum + 1);
                            vertexStarOffsets[0] = 0;
                            for (int j = 0; j < localVertexNum; ++j) {
                                int rj = j + diff; // relative j value in the large matrix
                                vertexStarData.insert(vertexStarData.end(), 
                                hVertexStarMatrix_ + rj * vtColumnNum,
                                hVertexStarMatrix_ + rj * vtColumnNum + hVertexStarSizes_[rj]);
                                vertexStarOffsets[j+1] = vertexStarOffsets[j] + hVertexStarSizes_[rj];
                            }
                            allClusters_[sid].vertexStars_[tid].setData(std::move(vertexStarData), std::move(vertexStarOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dVertexStarSizes_));
                CHECK(cudaFree(dVertexStarMatrix_));
                delete[] hVertexStarMatrix_;
                delete[] hVertexStarSizes_;

                break;
            }

            case RelationType::VVRelation:
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                        __FILE__, __LINE__); if (abort) exit(status);
                }

                for (int i = 0; i < threadNum; ++i) {
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].vertexNeighbors_[tid].empty()) {
                            std::vector<SimplexId> vertexNeighborData{};
                            int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                            std::vector<SimplexId> vertexNeighborOffsets(localVertexNum + 1);
                            vertexNeighborOffsets[0] = 0;
                            for (int j = 0; j < localVertexNum; ++j) {
                                int rj = j + diff; // relative j value in the large matrix
                                vertexNeighborData.insert(vertexNeighborData.end(), 
                                hVertexNeighborMatrix_ + rj * vvColumnNum,
                                hVertexNeighborMatrix_ + rj * vvColumnNum + hVertexNeighborSizes_[rj]);
                                vertexNeighborOffsets[j+1] = vertexNeighborOffsets[j] + hVertexNeighborSizes_[rj];
                            }
                            allClusters_[sid].vertexNeighbors_[tid].setData(std::move(vertexNeighborData), std::move(vertexNeighborOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dVertexNeighborSizes_));
                CHECK(cudaFree(dVertexNeighborMatrix_));
                delete[] hVertexNeighborMatrix_;
                delete[] hVertexNeighborSizes_;

                break;
            }
        }
        CHECK(cudaStreamDestroy(threadStreams[curStream]));
    }
    
    return res;
}


void Gale::assistProcedure() {
    volatile SimplexId leaderCluster = 0;
    volatile RelationType leaderFunc = RelationType::BlankRelation;

    /* Job queue version */
    if (threadNumber_ == 1) {
        while (true) {
            std::unique_lock<std::mutex> qlock(qMutex_);
            int qSize = jobQueue.size();
            qlock.unlock();
            if (qSize > 0) {
                int tid = jobQueue.front();
                leaderCluster = currClusters_[tid], leaderFunc = currRelations_[tid];
                int res = computeRelation(leaderFunc, leaderCluster, tid, true);
                if (res == 0) {
                    std::unique_lock<std::mutex> qlock(qMutex_);
                    jobQueue.pop();
                    qlock.unlock();
                }
            }
            else {
                    leaderCluster = currClusters_[0], leaderFunc = currRelations_[0];
                    if (leaderCluster == -1) return;
                    else if (leaderCluster == 0) continue;
                    
                    // SimplexId diff = leaderCluster - baseClusters_[static_cast<int>(leaderFunc)][i];
                    // // check if the precomputation needs to be performed
                    // if (diff < 0 || diff > numBlocks_ * blockSize_ * 0.6) {
                    //     // std::cout << "precomputing for thread " << i << " cluster = " << leaderCluster << " diff = " << diff << std::endl;
                    //     computeRelation(leaderFunc, leaderCluster, i, false);
                    // }
                    
                    SimplexId startCluster = baseClusters_[static_cast<int>(leaderFunc)][0];
                    if (startCluster == 0) continue;
                    SimplexId endCluster = std::max(leaderCluster - 5, 1);
                    // different relations correspond to different vectors
                    switch (leaderFunc)
                    {
                        case RelationType::BVRelation:
                        {
                            for (int cid = startCluster; cid <= endCluster; ++cid) {
                                allClusters_[cid].boundaryVertices_[0] = std::vector<bool>();
                            }
                            break;
                        }

                        case RelationType::BERelation:
                        {
                            for (int cid = startCluster; cid <= endCluster; ++cid) {
                                allClusters_[cid].boundaryEdges_[0] = std::vector<bool>();
                            }
                            break;
                        }
                        
                        case RelationType::EFRelation:
                        {
                            for (int eid = hEdgeIntervals_[startCluster-1]; eid < hEdgeIntervals_[endCluster]; ++eid) {
                                edgeTriangleList_[eid].clear(); edgeTriangleList_[eid].shrink_to_fit();
                            }
                            break;
                        }
                        
                        case RelationType::ETRelation:
                        {
                            for (int eid = hEdgeIntervals_[startCluster-1]; eid < hEdgeIntervals_[endCluster]; ++eid) {
                                edgeStarList_[eid].clear(); edgeStarList_[eid].shrink_to_fit();
                            }
                            break;
                        }
                        
                        case RelationType::FTRelation:
                        {
                            for (int fid = hTriangleIntervals_[startCluster-1]; fid < hTriangleIntervals_[endCluster]; ++fid) {
                                triangleStarList_[fid].clear(); triangleStarList_[fid].shrink_to_fit();
                            }
                            break;
                        }

                        case RelationType::VVRelation:
                        {
                            for (int vid = hVertexIntervals_[startCluster-1]; vid < hVertexIntervals_[endCluster]; ++vid) {
                                vertexNeighborList_[vid].clear(); vertexNeighborList_[vid].shrink_to_fit();
                            }
                            break;
                        }
                        
                        case RelationType::VERelation:
                        {
                            for (int vid = hVertexIntervals_[startCluster-1]; vid < hVertexIntervals_[endCluster]; ++vid) {
                                vertexEdgeList_[vid].clear(); vertexEdgeList_[vid].shrink_to_fit();
                            }
                            break;
                        }

                        case RelationType::VFRelation:
                        {
                            for (int vid = hVertexIntervals_[startCluster-1]; vid < hVertexIntervals_[endCluster]; ++vid) {
                                vertexTriangleList_[vid].clear(); vertexTriangleList_[vid].shrink_to_fit();
                            }
                            break;
                        }

                        case RelationType::VTRelation:
                        {
                            for (int vid = hVertexIntervals_[startCluster-1]; vid < hVertexIntervals_[endCluster]; ++vid) {
                                vertexStarList_[vid].clear(); vertexStarList_[vid].shrink_to_fit();
                            }
                            break;
                        }

                        default:
                            break;
                    }
                    // reset the base cluster of the thread to 0
                    baseClusters_[static_cast<int>(leaderFunc)][0] = 0;
            }
        }
    }
    
    else {
        while (true) {
            /* Job queue version */
            std::unique_lock<std::mutex> qlock(qMutex_);
            int qSize = jobQueue.size();
            qlock.unlock();
            if (qSize > 0) {
                std::vector<RelationType> relationOrders;
                std::map<RelationType, std::vector<std::pair<int,int>>> requestMap;
                // handle the waiting jobs in the queue
                // combine the requests from all waiting consumers
                std::unique_lock<std::mutex> qlock(qMutex_);
                for (int i = 0; i < qSize; ++i) {
                    int tid = jobQueue.front();
                    jobQueue.pop();
                    RelationType jobType = currRelations_[tid];
                    if (requestMap.find(jobType) == requestMap.end()) {
                        relationOrders.push_back(jobType);
                    }
                    requestMap[jobType].push_back(std::make_pair(tid, currClusters_[tid]));
                }
                qlock.unlock();
                computeCombinedRelation(requestMap, relationOrders);
            }
            else {
                // clean the memory storage
                for (int i = 0; i < threadNumber_; ++i) {
                    SimplexId endCluster = currClusters_[i];
                    if (endCluster < 0) return;
                    else if (endCluster == 0) continue;
                    if (!jobQueue.empty()) break;
                    for (RelationType &relation : preconditionedRelations_) {
                        if (relation == currRelations_[i]) continue;
                        SimplexId startCluster = baseClusters_[static_cast<int>(relation)][i];
                        // endCluster = (endCluster - startCluster) * removalRatio + startCluster;
                        if (startCluster == 0) continue;
                        // different relations correspond to different vectors
                        switch (relation)
                        {
                            case RelationType::BVRelation:
                            {
                                for (int cid = startCluster; cid < endCluster - 1; ++cid) {
                                    allClusters_[cid].boundaryVertices_[i] = std::vector<bool>();
                                }
                                break;
                            }

                            case RelationType::BERelation:
                            {
                                for (int cid = startCluster; cid < endCluster - 1; ++cid) {
                                    allClusters_[cid].boundaryEdges_[i] = std::vector<bool>();
                                }
                                break;
                            }
                            
                            case RelationType::EFRelation:
                            {
                                for (int eid = hEdgeIntervals_[startCluster-1]; eid < hEdgeIntervals_[endCluster-5]; ++eid) {
                                    edgeTriangleList_[eid] = std::vector<int>();
                                }
                                break;
                            }
                            
                            case RelationType::ETRelation:
                            {
                                for (int eid = hEdgeIntervals_[startCluster-1]; eid < hEdgeIntervals_[endCluster-5]; ++eid) {
                                    edgeStarList_[eid] = std::vector<int>();
                                }
                                break;
                            }
                            
                            case RelationType::FTRelation:
                            {
                                for (int fid = hTriangleIntervals_[startCluster-1]; fid < hTriangleIntervals_[endCluster-5]; ++fid) {
                                    triangleStarList_[fid] = std::vector<int>();
                                }
                                break;
                            }

                            case RelationType::VVRelation:
                            {
                                for (int cid = startCluster; cid < endCluster - 1; ++cid) {
                                    allClusters_[cid].vertexNeighbors_[i].clear();
                                }
                                break;
                            }
                        
                            case RelationType::VERelation:
                            {
                                for (int cid = startCluster; cid < endCluster - 1; ++cid) {
                                    allClusters_[cid].vertexEdges_[i].clear();
                                }
                                break;
                            }

                            case RelationType::VFRelation:
                            {
                                for (int cid = startCluster; cid < endCluster - 1; ++cid) {
                                    allClusters_[cid].vertexTriangles_[i].clear();
                                }
                                break;
                            }

                            case RelationType::VTRelation:
                            {
                                for (int cid = startCluster; cid < endCluster - 1; ++cid) {
                                    allClusters_[cid].vertexStars_[i].clear();
                                }
                                break;
                            }
                            
                            default:
                                break;
                            
                            // reset the base cluster of the thread to 0
                            baseClusters_[static_cast<int>(relation)][i] = 0;
                        }
                    }
                }
            }
        }
    }
}

void Gale::initialize(vtkDataSet *dataSet, int blockNum, int blockSize, bool reorder) {
    if (dataSet->GetDataObjectType() != VTK_UNSTRUCTURED_GRID) {
        std::cerr << "The input dataset is not unstructured grid!" << std::endl;
        return;
    }
    vtkPointSet *pointSet = static_cast<vtkPointSet *>(dataSet);
    vtkPoints *points = pointSet->GetPoints();
    if(!points) {
        std::cerr << "DataSet has uninitialized `vtkPoints`." << std::endl;
    }

    vtkCellArray *cells = static_cast<vtkUnstructuredGrid *>(pointSet)->GetCells();
    if(!cells) {
        std::cerr << "DataSet has uninitialized `vtkCellArray`." << std::endl;
    }

    // Points
    auto pointDataType = points->GetDataType();
    if (pointDataType == VTK_FLOAT) {
        doublePrecision_ = false;
    } else if (pointDataType == VTK_DOUBLE) {
        doublePrecision_ = true;
    } else {
        std::cerr << "Unable to initialize 'ttk::Triangulation' for point: precision other than 'float' or 'double'." << std::endl;
    }

    pointSet_ = points->GetData()->GetVoidPointer(0);
    vertexNumber_ = points->GetNumberOfPoints();

    // Cells
    cellNumber_ = cells->GetNumberOfCells();
    if(cellNumber_ > 0) {
        if(!cells->IsStorage64Bit()) {
            if(cells->CanConvertTo64BitStorage()) {
                std::cout << "Converting the cell array to 64-bit storage" << std::endl;
                bool success = cells->ConvertTo64BitStorage();
                if(!success) {
                    std::cerr << 
                        "Error converting the provided cell array to 64-bit storage" << std::endl;
                }
            } else {
                std::cerr << 
                    "Cannot convert the provided cell array to 64-bit storage" << std::endl;
            }
        }
        hConnectivity_ = static_cast<vtkIdType *>(cells->GetConnectivityArray()->GetVoidPointer(0));
        hOffsets_ = static_cast<vtkIdType *>(cells->GetOffsetsArray()->GetVoidPointer(0));
        cellDimension_ = hOffsets_[1] - hOffsets_[0] - 1;
    }
    
    // index array
    hIndexArray_ = nullptr;
    int hasIndexArray = pointSet->GetPointData()->HasArray("_index");
    if (!hasIndexArray) {
        std::cerr << "Cannot detect index array! Gale must run with a predefined clustering information!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < threadNumber_; ++i) {
        sem_init(&relationSems_[i], 0, 0);
    }
    if (blockNum > 0) numBlocks_ = blockNum;
    if (blockSize > 0) {
        blockSize_ = blockSize;
        if (blockSize % threadsPerCluster_ != 0) {
            std::cerr << "Block size " << blockSize << " is not divisible by threads per cluster " << threadsPerCluster_ << std::endl;
            exit(EXIT_FAILURE); 
        }
    }

    if (reorder) {
        hIndexArray_ = new int[vertexNumber_];
        vtkDataArray *dataArray = pointSet->GetPointData()->GetArray("_index");
        memcpy(hIndexArray_, dataArray->GetVoidPointer(0), sizeof(int) * vertexNumber_);
        clusterNumber_ = 0;
        for (int i = 0; i < vertexNumber_; ++i) {
            hIndexArray_[i] += 1;
            clusterNumber_ = std::max(hIndexArray_[i], clusterNumber_);
        }
        clusterNumber_ += 1;
        
        std::cout << "[Gale] Reordering vertices and cells..." << std::endl;
        
        std::vector<SimplexId> vertexMap;       // old vid : new vid
        reorderVertices(vertexMap);
        reorderCells(vertexMap);
        // change vertex orders of point data arrays
        int arrNum = pointSet->GetPointData()->GetNumberOfArrays();
        for (int i = 0; i < arrNum; ++i) {
            vtkDataArray *dataArray = pointSet->GetPointData()->GetArray(i);
            vtkFieldData *fieldData = pointSet->GetAttributesAsFieldData(i);
            vtkDataArray *orderArray = nullptr;
            if (fieldData != nullptr)
                orderArray = fieldData->GetArray((std::string(dataArray->GetName()) + "_Order").data());

            if (dataArray->GetName() == "_index") continue;
            switch(dataArray->GetDataType()) {
                case VTK_INT:
                {
                    vtkIntArray *dataIntArray = vtkIntArray::SafeDownCast(dataArray);
                    vtkSmartPointer<vtkIntArray> tmpArray = vtkSmartPointer<vtkIntArray>::New();
                    tmpArray->DeepCopy(dataIntArray);
                    for (SimplexId vid = 0; vid < vertexNumber_; ++vid) {
                        dataIntArray->SetValue(vertexMap[vid], tmpArray->GetValue(vid));
                    }
                    break;
                }
                case VTK_FLOAT:
                {
                    vtkFloatArray *dataFloatArray = vtkFloatArray::SafeDownCast(dataArray);
                    vtkSmartPointer<vtkFloatArray> tmpArray = vtkSmartPointer<vtkFloatArray>::New();
                    tmpArray->DeepCopy(dataFloatArray);
                    for (SimplexId vid = 0; vid < vertexNumber_; ++vid) {
                        dataFloatArray->SetValue(vertexMap[vid], tmpArray->GetValue(vid));
                    }
                    break;
                }
                case VTK_DOUBLE:
                {
                    vtkDoubleArray *dataDoubleArray = vtkDoubleArray::SafeDownCast(dataArray);
                    vtkSmartPointer<vtkDoubleArray> tmpArray = vtkSmartPointer<vtkDoubleArray>::New();
                    tmpArray->DeepCopy(dataDoubleArray);
                    for (SimplexId vid = 0; vid < vertexNumber_; ++vid) {
                        dataDoubleArray->SetValue(vertexMap[vid], tmpArray->GetValue(vid));
                    }
                    break;
                }
                case VTK_ID_TYPE:
                {
                    vtkIdTypeArray *dataIdTypeArray = vtkIdTypeArray::SafeDownCast(dataArray);
                    vtkSmartPointer<vtkIdTypeArray> tmpArray = vtkSmartPointer<vtkIdTypeArray>::New();
                    tmpArray->DeepCopy(dataIdTypeArray);
                    for (SimplexId vid = 0; vid < vertexNumber_; ++vid) {
                        dataIdTypeArray->SetValue(vertexMap[vid], tmpArray->GetValue(vid));
                    }
                    break;
                }
                default:
                    std::cerr << "[Gale] Not supported data array type: " << dataArray->GetDataType() << std::endl;
                    break;
            }
            if (orderArray != nullptr) {
                switch(orderArray->GetDataType()) {
                    case VTK_INT:
                    {
                        vtkIntArray *orderIntArray = vtkIntArray::SafeDownCast(orderArray);
                        vtkSmartPointer<vtkIntArray> tmpArray = vtkSmartPointer<vtkIntArray>::New();
                        tmpArray->DeepCopy(orderIntArray);
                        for (SimplexId vid = 0; vid < vertexNumber_; ++vid) {
                            orderIntArray->SetValue(vertexMap[vid], tmpArray->GetValue(vid));
                        }
                        break;
                    }
                    case VTK_ID_TYPE:
                    {
                        vtkIdTypeArray *orderIdTypeArray = vtkIdTypeArray::SafeDownCast(orderArray);
                        vtkSmartPointer<vtkIdTypeArray> tmpArray = vtkSmartPointer<vtkIdTypeArray>::New();
                        tmpArray->DeepCopy(orderIdTypeArray);
                        for (SimplexId vid = 0; vid < vertexNumber_; ++vid) {
                            orderIdTypeArray->SetValue(vertexMap[vid], tmpArray->GetValue(vid));
                        }
                        break;
                    }
                    default:
                        std::cerr << "[Gale] Not supported data array type: " << orderArray->GetDataType() << std::endl;
                        break;
                }
            }
        }
    }
    else {
        hIndexArray_ = (int *) pointSet->GetPointData()->GetArray("_index")->GetVoidPointer(0);
        clusterNumber_ = 0;
        for (int i = 0; i < vertexNumber_; ++i) {
            hIndexArray_[i] += 1;
            clusterNumber_ = std::max(hIndexArray_[i], clusterNumber_);
        }
        clusterNumber_ += 1;
        computeCellIntervals();
        computeVertexIntervals();
    }
    
    allClusters_ = std::vector<ImplicitCluster>(clusterNumber_);
    for (int cid = 1; cid < clusterNumber_; ++cid) {
        allClusters_[cid] = ImplicitCluster(cid, threadNumber_);
    }
    producer_ = std::thread(&Gale::assistProcedure, this);
    printf("[Gale] GPU threads per cluster: %d, thread num: %d, block number: %d, block size: %d, cluster number: %d, cell number: %d, vertex number: %d\n", threadsPerCluster_, threadNumber_, numBlocks_, blockSize_, clusterNumber_, cellNumber_, vertexNumber_);

    // copy input mesh arrays to the GPU     
    CHECK(cudaMallocAsync(&dConnectivity_, sizeof(vtkIdType) * (cellNumber_ * 4), 0));
    CHECK(cudaMemcpyAsync(dConnectivity_, hConnectivity_, sizeof(vtkIdType) * (cellNumber_ * 4), cudaMemcpyHostToDevice));
    CHECK(cudaMallocAsync(&dOffsets_, sizeof(vtkIdType) * (cellNumber_+1), 0));
    CHECK(cudaMemcpyAsync(dOffsets_, hOffsets_, sizeof(vtkIdType) * (cellNumber_+1), cudaMemcpyHostToDevice));
    CHECK(cudaMallocAsync(&dIndexArray_, sizeof(int) * vertexNumber_, 0));
    CHECK(cudaMemcpyAsync(dIndexArray_, hIndexArray_, sizeof(int) * vertexNumber_, cudaMemcpyHostToDevice));
}


void Gale::computeCellIntervals() {
    std::vector<std::vector<int>> externalCells(clusterNumber_);
    hExternalCellOffsets_ = new int[clusterNumber_];
    hExternalCellOffsets_[0] = 0;
    hCellIntervals_ = (int *) calloc(clusterNumber_, sizeof(int));

    CHECK(cudaMallocAsync(&dCellIntervals_, sizeof(int) * clusterNumber_, 0));
    CHECK(cudaMallocAsync(&dExternalCellOffsets_, sizeof(int) * clusterNumber_, 0));

    for (int cid = 0; cid < cellNumber_; ++cid) {
        int clusterId = hIndexArray_[hConnectivity_[hOffsets_[cid]]];
        for (int j = 1; j <= cellDimension_; ++j) {
            int tmpCluster = hIndexArray_[hConnectivity_[hOffsets_[cid] + j]];
            if (tmpCluster > clusterId && tmpCluster < clusterNumber_) {
                if (externalCells[tmpCluster].empty() || externalCells[tmpCluster].back() != cid) {
                    externalCells[tmpCluster].push_back(cid);
                }
            }
        }
        hCellIntervals_[clusterId]++;
    }

    for (int i = 1; i < clusterNumber_; ++i) {
        hCellIntervals_[i] += hCellIntervals_[i-1];
        hExternalCellOffsets_[i] = hExternalCellOffsets_[i-1] + externalCells[i].size();
    }
    CHECK(cudaMemcpyAsync(dCellIntervals_, hCellIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(dExternalCellOffsets_, hExternalCellOffsets_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));

    // Convert the 2D vector to 1D flat vector
    hExternalCells_ = new int[hExternalCellOffsets_[clusterNumber_-1]];
    int *tmp = hExternalCells_;
    CHECK(cudaMalloc((void **) &dExternalCells_, sizeof(int) * hExternalCellOffsets_[clusterNumber_-1]));
    for (int i = 1; i < externalCells.size(); ++i) {
        std::copy(externalCells[i].begin(), externalCells[i].end(), tmp);
        tmp += externalCells[i].size();
    }
    CHECK(cudaMemcpy(dExternalCells_, hExternalCells_, sizeof(int) * hExternalCellOffsets_[clusterNumber_-1], cudaMemcpyHostToDevice));

    // int blockSize = 32;
    // int numBlocks = vertexNumber_ / blockSize + 1;
    // cudaMalloc(&dCellIntervals_, sizeof(int) * clusterNumber_);
    // cudaMemset(dCellIntervals_, 0, clusterNumber_);

    // countCells<<<numBlocks, blockSize>>>(dCellIntervals_, dConnectivity_, dOffsets_, dIndexArray_, cellNumber_, 3);

    // hCellIntervals_ = (int *) malloc(sizeof(int) * clusterNumber_);
    // cudaMemcpy(hCellIntervals_, dCellIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyDeviceToHost);
}


void Gale::computeVertexIntervals() {
    // int blockSize = 64;
    // int numBlocks = vertexNumber_ / blockSize + 1;

    // cudaMalloc(&dVertexIntervals_, sizeof(int) * (clusterNumber_+1));
    // cudaMemset(dVertexIntervals_, 0, clusterNumber_+1);
    // countVertices<<<numBlocks, blockSize>>>(dVertexIntervals_, dIndexArray_, vertexNumber_);

    // hVertexIntervals_ = (int *) malloc(sizeof(int) * (clusterNumber_+1));
    // cudaMemcpy(hVertexIntervals_, dVertexIntervals_, sizeof(int) * (clusterNumber_+1), cudaMemcpyDeviceToHost);
    // hVertexIntervals_[0] = -1;
    // for (int i = 1; i < clusterNumber_; ++i) hVertexIntervals_[i] = hVertexIntervals_[i-1] + hVertexIntervals_[i];

    CHECK(cudaMallocAsync(&dVertexIntervals_, sizeof(int) * clusterNumber_, 0));

    hVertexIntervals_ = (int *) calloc(clusterNumber_, sizeof(int));
    for (int vid = 0; vid < vertexNumber_; ++vid) {
        int clusterId = hIndexArray_[vid];
        hVertexIntervals_[clusterId]++;
    }
    for (int cid = 1; cid < clusterNumber_; ++cid) {
        hVertexIntervals_[cid] += hVertexIntervals_[cid-1];
    }

    CHECK(cudaMemcpyAsync(dVertexIntervals_, hVertexIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice, 0));
}


int Gale::reorderVertices(std::vector<SimplexId> &vertexMap) {
    CHECK(cudaMallocAsync(&dVertexIntervals_, sizeof(int) * clusterNumber_, 0));
    std::vector<std::vector<SimplexId>> clusterVertices(clusterNumber_);
    for(SimplexId vid = 0; vid < vertexNumber_; ++vid) {
        clusterVertices[hIndexArray_[vid]].push_back(vid);
    }

    // update the vertex intervals and vertex mapping
    vertexMap.resize(vertexNumber_);
    hVertexIntervals_ = new int[clusterNumber_];
    hVertexIntervals_[0] = 0;
    SimplexId vertexCount = 0;
    for(SimplexId nid = 1; nid < clusterNumber_; ++nid) {
        for(SimplexId &vid : clusterVertices[nid]) {
            vertexMap[vid] = vertexCount++;
        }
        hVertexIntervals_[nid] = vertexCount;
    }
    CHECK(cudaMemcpyAsync(dVertexIntervals_, hVertexIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice, 0));

    // rearange the vertex coordinate values
    if(doublePrecision_) {
        double *newPointSet = new double[3 * vertexNumber_];
        mempcpy(newPointSet, pointSet_, sizeof(double) * 3 * vertexNumber_);
        // for(SimplexId vid = 0; vid < vertexNumber_; vid++) {
        //     for(int j = 0; j < 3; j++) {
        //         newPointSet[3 * vertexMap[vid] + j]
        //         = ((double *)pointSet_)[3 * vid + j];
        //     }
        // }
        for(SimplexId vid = 0; vid < vertexNumber_; vid++) {
            for(int j = 0; j < 3; j++) {
                ((double *)pointSet_)[3 * vertexMap[vid] + j] = newPointSet[3 * vid + j];
            }
        }
        delete[] newPointSet;
    } else {
        float *newPointSet = new float[3 * vertexNumber_];
        mempcpy(newPointSet, pointSet_, sizeof(float) * 3 * vertexNumber_);
        for(SimplexId vid = 0; vid < vertexNumber_; vid++) {
            for(int j = 0; j < 3; j++) {
                ((float *)pointSet_)[3 * vertexMap[vid] + j] = newPointSet[3 * vid + j];
            }
        }
        delete[] newPointSet;
    }

    // change the vertex indices
    for(SimplexId nid = 1; nid < clusterNumber_; nid++) {
        for(SimplexId vid = hVertexIntervals_[nid - 1];
            vid < hVertexIntervals_[nid]; ++vid) {
                hIndexArray_[vid] = nid;
        }
    }

    return 0;
}

int Gale::reorderCells(const std::vector<SimplexId> &vertexMap) {
    // change the indices in cell array
    SimplexId cellCount = 0, verticesPerCell = hOffsets_[1] - hOffsets_[0];
    std::vector<std::vector<SimplexId>> clusterCells(clusterNumber_ + 1);

    for(SimplexId cid = 0; cid < cellNumber_; cid++) {
        SimplexId cellId = hOffsets_[cid];
        for(int j = 0; j < verticesPerCell; j++) {
            hConnectivity_[cellId + j] = vertexMap[hConnectivity_[cellId + j]];
        }
        std::sort(hConnectivity_ + cellId, hConnectivity_ + cellId + verticesPerCell);
        clusterCells[hIndexArray_[hConnectivity_[cellId]]].push_back(cid);
    }

    // rearange the cell std::array
    hCellIntervals_ = new int[clusterNumber_];
    std::vector<std::vector<int>> externalCells(clusterNumber_);
    hCellIntervals_[0] = 0;

    vtkIdType *newCellArray
    = new vtkIdType[verticesPerCell * cellNumber_];
    for(SimplexId clusterId = 1; clusterId < clusterNumber_; ++clusterId) {
        for(SimplexId cid : clusterCells[clusterId]) {
            SimplexId cellId = verticesPerCell * cid;
            SimplexId newCellId = verticesPerCell * cellCount;
            for(int j = 0; j < verticesPerCell; j++) {
                newCellArray[newCellId + j] = hConnectivity_[cellId + j];
                if(newCellArray[newCellId + j] >= hVertexIntervals_[clusterId]) {
                    SimplexId nodeNum = hIndexArray_[newCellArray[newCellId + j]];
                    if(externalCells[nodeNum].empty()
                        || externalCells[nodeNum].back() != cid) {
                        externalCells[nodeNum].push_back(cid);
                    }
                }
            }
            cellCount++;
        }
        hCellIntervals_[clusterId] = cellCount;
    }

    hExternalCellOffsets_ = new int[clusterNumber_];
    hExternalCellOffsets_[0] = 0;
    for (int i = 1; i < clusterNumber_; ++i) {
        hExternalCellOffsets_[i] = hExternalCellOffsets_[i-1] + externalCells[i].size();
    }
    

    CHECK(cudaMallocAsync(&dCellIntervals_, sizeof(int) * clusterNumber_, 0));
    CHECK(cudaMemcpyAsync(dCellIntervals_, hCellIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));

    CHECK(cudaMallocAsync(&dExternalCellOffsets_, sizeof(int) * clusterNumber_, 0));
    CHECK(cudaMemcpyAsync(dExternalCellOffsets_, hExternalCellOffsets_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));

    // Convert the 2D vector to 1D flat vector
    hExternalCells_ = new int[hExternalCellOffsets_[clusterNumber_-1]];
    int *tmp = hExternalCells_;
    CHECK(cudaMallocAsync((void **) &dExternalCells_, sizeof(int) * hExternalCellOffsets_[clusterNumber_-1], 0));
    for (int i = 1; i < externalCells.size(); ++i) {
        std::copy(externalCells[i].begin(), externalCells[i].end(), tmp);
        tmp += externalCells[i].size();
    }
    CHECK(cudaMemcpyAsync(dExternalCells_, hExternalCells_, sizeof(int) * hExternalCellOffsets_[clusterNumber_-1], cudaMemcpyHostToDevice));
    
    // copy the new cell array back to original one
    memcpy(hConnectivity_, newCellArray, sizeof(vtkIdType) * cellNumber_ * verticesPerCell);
    delete[] newCellArray;

    return 0;
}


int Gale::preconditionEdgesInternal() {
    if(edgeList_.empty()) {
        Timer t;
        std::vector<std::vector<std::array<SimplexId, 2>>> clusterEdges(clusterNumber_);
        hEdgeIntervals_ = new int[clusterNumber_];
        hEdgeIntervals_[0] = 0;
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic)
#endif
        for(SimplexId nid = 1; nid < clusterNumber_; nid++) {
            hEdgeIntervals_[nid] = buildInternalEdgeList(nid, clusterEdges);
        }

        for(SimplexId nid = 1; nid < clusterNumber_; nid++) {
            hEdgeIntervals_[nid] = hEdgeIntervals_[nid - 1] + hEdgeIntervals_[nid];
        }
        CHECK(cudaMallocAsync(&dEdgeIntervals_, sizeof(int) * clusterNumber_, 0));
        CHECK(cudaMemcpyAsync(dEdgeIntervals_, hEdgeIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice, 0));

        edgeNumber_ = hEdgeIntervals_[clusterNumber_ - 1];
        edgeList_.reserve(hEdgeIntervals_[clusterNumber_-1]);
        for(SimplexId nid = 1; nid < clusterNumber_; nid++) {
            for(auto &edge : clusterEdges[nid]) {
                edgeList_.push_back(std::move(edge));
            }
        }
        edgeList_.shrink_to_fit();

        std::cout << "[Gale] Edges preconditioned in " << t.getElapsedTime() << "s." << std::endl;
    }

    return 0;
}

int Gale::preconditionTrianglesInternal() {
    if(triangleList_.empty()) {
        Timer t;
        std::vector<std::vector<std::array<SimplexId, 3>>> clusterTriangles(clusterNumber_);
        hTriangleIntervals_ = new int[clusterNumber_];
        hTriangleIntervals_[0] = 0;

        // use the producer threads to compute the triangles
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic)
#endif
        for(SimplexId nid = 1; nid < clusterNumber_; nid++) {
            hTriangleIntervals_[nid] = buildInternalTriangleList(nid, clusterTriangles);
        }

        for(SimplexId nid = 1; nid < clusterNumber_; nid++) {
            hTriangleIntervals_[nid] = hTriangleIntervals_[nid - 1] + hTriangleIntervals_[nid];
        }
        CHECK(cudaMallocAsync(&dTriangleIntervals_, sizeof(int) * clusterNumber_, 0));
        CHECK(cudaMemcpyAsync(dTriangleIntervals_, hTriangleIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));

        triangleNumber_ = hTriangleIntervals_[clusterNumber_ - 1];
        triangleList_.reserve(hTriangleIntervals_[clusterNumber_-1]);
        for(SimplexId nid = 1; nid < clusterNumber_; nid++) {
            for(auto &triangle : clusterTriangles[nid]) {
                triangleList_.push_back(std::move(triangle));
            }
        }
        triangleList_.shrink_to_fit();

        std::cout << "[Gale] Triangles preconditioned in " << t.getElapsedTime() << "s." << std::endl;
    }

    return 0;
}

/**
// GPU version is slower than the CPU version due to the variable size of edge list
int Gale::preconditionEdgesInternal() {
    if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
        std::cerr << "Empty dataset, precondition skipped";
        return 1;
    }

    if (edgeList_.empty()) {
        Timer t;
        // preallocate enough memory for each cluster
        hEdgeIntervals_ = new int[clusterNumber_];
        hEdgeIntervals_[0] = 0;
        for (int clusterId = 1; clusterId < clusterNumber_; ++clusterId) {
            hEdgeIntervals_[clusterId] = hEdgeIntervals_[clusterId-1];
            hEdgeIntervals_[clusterId] += (hCellIntervals_[clusterId] - hCellIntervals_[clusterId - 1]) * (cellDimension_ - 1) * 3;
            // hEdgeIntervals_[clusterId] += (hExternalCellOffsets_[clusterId] - hExternalCellOffsets_[clusterId - 1]) * cellDimension_;
        }

        int *hEdgeList, *dEdgeList, *hEdgeCounts, *dEdgeCounts;
        CHECK(cudaMalloc(&dEdgeList, 2 * sizeof(int) * hEdgeIntervals_[clusterNumber_-1]));
        hEdgeList = new int[hEdgeIntervals_[clusterNumber_-1] * 2];
        hEdgeCounts = new int[clusterNumber_];
        CHECK(cudaMalloc(&dEdgeCounts, sizeof(int) * clusterNumber_));
        CHECK(cudaMalloc(&dEdgeIntervals_, sizeof(int) * clusterNumber_));
        CHECK(cudaMemcpy(dEdgeIntervals_, hEdgeIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dEdgeCounts, 0, sizeof(int) * clusterNumber_));
        CHECK(cudaMemset(dEdgeList, -1, 2 * sizeof(int) * hEdgeIntervals_[clusterNumber_-1]));

        int blockNum = clusterNumber_ / (blockSize_ / threadsPerCluster_) + 1;
        buildEdges<<<blockNum, blockSize_>>>(dEdgeList, dEdgeCounts, 
            dIndexArray_, dCellIntervals_, dVertexIntervals_, 
            dEdgeIntervals_, dExternalCells_, dExternalCellOffsets_, 
            dConnectivity_, dOffsets_, clusterNumber_, cellNumber_, 
            cellDimension_);
        
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                __FILE__, __LINE__); if (abort) exit(status);
        }
        
        CHECK(cudaMemcpy(hEdgeList, dEdgeList, 2 * sizeof(int) * hEdgeIntervals_[clusterNumber_-1], cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hEdgeCounts, dEdgeCounts, sizeof(int) * clusterNumber_, cudaMemcpyDeviceToHost));

        // copy the edge list 
        edgeList_.reserve(hEdgeIntervals_[clusterNumber_-1]);
        for (int clusterId = 1; clusterId < clusterNumber_; ++clusterId) {
            std::vector<std::array<SimplexId, 2>> clusterEdges;
            for (int i = 0; i < hEdgeCounts[clusterId]; ++i) {
                int startId = hEdgeIntervals_[clusterId-1];
                clusterEdges.push_back({hEdgeList[startId + 2*i], hEdgeList[startId + 2*i + 1]});
            }
            std::sort(clusterEdges.begin(), clusterEdges.end());
            edgeList_.insert(edgeList_.end(), clusterEdges.begin(), clusterEdges.end());
        }
        edgeList_.shrink_to_fit();
        for (int clusterId = 1; clusterId < clusterNumber_; ++clusterId) {
            hEdgeIntervals_[clusterId] = hEdgeCounts[clusterId] + hEdgeIntervals_[clusterId - 1];
            // std::cout << "edge interval " << clusterId << ": " << hEdgeIntervals_[clusterId] << std::endl;
        }

        delete[] hEdgeList;
        delete[] hEdgeCounts;
        CHECK(cudaFree(dEdgeList));
        CHECK(cudaFree(dEdgeCounts));

        std::cout << "Edges preconditioned in " << t.getElapsedTime() << " s." << std::endl;
    }

    return 0;
}
**/

// int Gale::preconditionVertexNeighborsInternal() {
//     if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
//         std::cerr << "Empty dataset, precondition skipped";
//         return 1;
//     }
//     vertexNeighborList_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});

    /* No precomputation during the preconditioning phase as there are multiple consumers */
    // // let GPU compute vertex neighbor data
    // int endCluster = std::min(numBlocks_ * blockSize_, clusterNumber_ - 1);
    // const int computedVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[0];

    // CHECK(cudaMalloc((void **) &dVertexNeighborSizes_, sizeof(int) * computedVertexNum));
    // CHECK(cudaMalloc((void **) &dVertexNeighborMatrix_, sizeof(int) * computedVertexNum * vvColumnNum));

    // computeVertexNeighbors<<<numBlocks_, blockSize_>>>(
    //     dVertexNeighborMatrix_, dVertexNeighborSizes_,
    //     1, clusterNumber_,
    //     dConnectivity_, dOffsets_, dIndexArray_,
    //     dVertexIntervals_, dCellIntervals_, dExternalCells_,
    //     dExternalCellOffsets_, cellDimension_);

    // hVertexNeighborSizes_ = new int[computedVertexNum];
    // hVertexNeighborMatrix_ = new int[computedVertexNum * vvColumnNum];

    // cudaError_t status = cudaDeviceSynchronize();
    // if (status != cudaSuccess) {
    //     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //         __FILE__, __LINE__); if (abort) exit(status);
    // }

    // CHECK(cudaMemcpy(hVertexNeighborSizes_, dVertexNeighborSizes_, sizeof(int) * computedVertexNum, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(hVertexNeighborMatrix_, dVertexNeighborMatrix_, sizeof(int) * computedVertexNum * vvColumnNum, cudaMemcpyDeviceToHost));

    // std::unique_lock<std::mutex> lck(VVMutex_);
    // for (int vid = 0; vid < computedVertexNum; ++vid) {
    //     if (!vertexNeighborList_[vid].empty()) continue;
    //     vertexNeighborList_[vid] = std::vector<int>(hVertexNeighborMatrix_ + vid * vvColumnNum, hVertexNeighborMatrix_ + vid * vvColumnNum + hVertexNeighborSizes_[vid]);
    // }
    // baseClusters_[static_cast<int>(RelationType::VVRelation)][0] = 1;

    // CHECK(cudaFree(dVertexNeighborSizes_)); dVertexNeighborSizes_ = nullptr;
    // CHECK(cudaFree(dVertexNeighborMatrix_)); dVertexNeighborMatrix_ = nullptr;

    // delete[] hVertexNeighborMatrix_;
    // delete[] hVertexNeighborSizes_;

//     return 0;
// }

// int Gale::preconditionVertexStarsInternal() {
//     if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
//         std::cerr << "Empty dataset, precondition skipped";
//         return 1;
//     }
//     vertexStarList_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});

    /* No precomputation during the preconditioning phase as there are multiple consumers */
    // // let GPU compute vertex neighbor data
    // int endCluster = std::min(numBlocks_ * blockSize_, clusterNumber_ - 1);
    // const int computedVertexNum = hVertexIntervals_[endCluster];

    // CHECK(cudaMalloc((void **) &dVertexStarSizes_, sizeof(int) * computedVertexNum));
    // CHECK(cudaMalloc((void **) &dVertexStarMatrix_, sizeof(int) * computedVertexNum * vvColumnNum));

    // computeVertexStars<<<numBlocks_, blockSize_>>>(
    //     dVertexStarMatrix_, dVertexStarSizes_,
    //     1, clusterNumber_,
    //     dConnectivity_, dOffsets_, dIndexArray_,
    //     dVertexIntervals_, dCellIntervals_, dExternalCells_,
    //     dExternalCellOffsets_, cellDimension_);

    // hVertexStarSizes_ = new int[computedVertexNum];
    // hVertexStarMatrix_ = new int[computedVertexNum * vvColumnNum];

    // cudaError_t status = cudaDeviceSynchronize();
    // if (status != cudaSuccess) {
    //     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //         __FILE__, __LINE__); if (abort) exit(status);
    // }

    // CHECK(cudaMemcpy(hVertexStarSizes_, dVertexStarSizes_, sizeof(int) * computedVertexNum, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(hVertexStarMatrix_, dVertexStarMatrix_, sizeof(int) * computedVertexNum * vvColumnNum, cudaMemcpyDeviceToHost));

    // std::unique_lock<std::mutex> lck(VTMutex_);
    // for (int vid = 0; vid < computedVertexNum; ++vid) {
    //     if (!vertexStarList_[vid].empty()) continue;
    //     vertexStarList_[vid] = std::vector<int>(hVertexStarMatrix_ + vid * vvColumnNum, hVertexStarMatrix_ + vid * vvColumnNum + hVertexStarSizes_[vid]);
    // }
    // lck.unlock();

    // CHECK(cudaFree(dVertexStarSizes_)); dVertexStarSizes_ = nullptr;
    // CHECK(cudaFree(dVertexStarMatrix_)); dVertexStarMatrix_ = nullptr;

    // delete[] hVertexStarMatrix_;
    // delete[] hVertexStarSizes_;

//     return 0;
// }
