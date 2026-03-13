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

/**
 * Constructor of GALE
 * `tnum` specifies the number of consumer threads
 * `tnumPerCluster` specifies the GPU threads per segment 
*/
Gale::Gale(int tnum, int tnumPerCluster, int SFCP_ColumnNum_) {
    this->setDebugMsgPrefix("Gale");
    threadNumber_ = tnum;
    threadsPerCluster_ = tnumPerCluster;
    currClusters_ = std::vector<SimplexId>(threadNumber_, 0);
    currRelations_ = std::vector<RelationType>(threadNumber_, RelationType::BlankRelation);
    relationSems_ = std::vector<sem_t>(threadNumber_);
    for (int i = 0; i < 25; ++i) {
        baseClusters_[i] = std::vector<int>(threadNumber_, 0);
    }
    SFCP_ColumnNum = SFCP_ColumnNum_;
    #ifdef MEASURE_TIME
    startTracking_ = false;
    timers_ = std::vector<Timer>(threadNumber_);
    waitingTimes_ = std::vector<std::array<double, 6>>(threadNumber_, std::array<double, 6>{});
    waitingCounts_ = std::vector<int>(threadNumber_, 0);
    // waitingTimes_.resize(threadNumber_);
    // waitingCounts_.resize(threadNumber_);
    #endif

    // GPU parameters
    blocksPerThread_ = 1, blockSize_ = 32;
}


/**
 * Destructor of GALE
*/
Gale::~Gale() {
    for (int i = 0; i < threadNumber_; ++i)
        currClusters_[i] = -1;
    for (int p = 0; p < producers_.size(); ++p) {
        if (producers_[p].joinable()) {
            producers_[p].join();
        }
    }

    #ifdef MEASURE_TIME
    // for (int i = 0; i < threadNumber_; ++i) {
    //     std::cout << "thread " << i << ": ";
    //     for (auto &e : waitingCounts_[i]) {
    //         std::cout << relationToString(e.first) << " missing counts: " << e.second << ", waiting time: " << waitingTimes_[i].at(e.first) << "; ";
    //         // std::cout << static_cast<int>(e.first) << " missing counts: " << e.second << ", waiting time: " << waitingTimes_[i].at(e.first) << "; ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // time for consumer thread to wait for the relation 
    for (int i = 0; i < threadNumber_; ++i) {
        // for (int j = 0; j < consumerTimes_[i].size(); ++j) {
        //     std::cout << "consumer " << i << ": " << consumerTimes_[i][j] << std::endl;
        // }
        // std::cout << "consumer " << i << ": " << waitingTimes_[i] << ", " << waitingCounts_[i] << std::endl;
        std::cout << "consumer " << i << ": ";
        for (int j = 0; j < 6; ++j) {
            std::cout << waitingTimes_[i][j] << ", ";
        }
        std::cout << waitingCounts_[i] << std::endl;
    }

    // for (int i = 0; i < producers_.size(); ++i) {
    //     for (int j = 0; j < qSizes_[i].size(); ++j) {
    //         std::cout << "queue length: " << qSizes_[i][j] << ", total time: " << qTimes_[i][j] << ", kernel time: " << kernelTimes_[i][j] << ", return time: " << returnTimes_[i][j] << std::endl;
    //     }
    // }

    #endif

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
    if (dEdgeList_ != nullptr) cudaFree(dEdgeList_);
    if (dTriangleList_ != nullptr) cudaFree(dTriangleList_);
    if (dBoundaryTriangles_ != nullptr) cudaFree(dBoundaryTriangles_);
}


/**
 * Extraction of edge list during the initialization phase.
*/
int Gale::buildInternalEdgeList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 2>>> &clusterEdges) {

    if(clusterId <= 0 || clusterId > clusterNumber_)
        return -1;

    SimplexId edgeCount = 0;
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
    return edgeCount;
}

/**
 * Extraction of edge list during the initialization phase.
*/
int Gale::buildInternalTriangleList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 3>>> &clusterTriangles) {

    if(clusterId <= 0 || clusterId > clusterNumber_)
        return -1;

    SimplexId triangleCount = 0;

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
    return triangleCount;
}

// Testing function for the new vertex neighbor kernel
void Gale::computeVertexNeighborWithExternals(const SimplexId clusterId) {
    int localVertexNum = hVertexIntervals_[clusterId] - hVertexIntervals_[clusterId - 1];
    int vertexCount = localVertexNum + (hCellIntervals_[clusterId] - hCellIntervals_[clusterId - 1] + hExternalCellOffsets_[clusterId] - hExternalCellOffsets_[clusterId - 1]) * 3; // other vertices (upper bound)
    int hTotalVertexNum = localVertexNum;
    
    // Use local variables instead of relationCompVecs_[0] to avoid conflicts
    int *dVertexMap, *dTotalVertexNum;
    int *dSizeArray, *dDataArray;
    int *hVertexMap = new int[vertexCount];
    int *hSizeArray = new int[vertexCount];
    int *hDataArray = new int[vertexCount * SFCP_ColumnNum];
    std::cout << "localVertexNum: " << localVertexNum << ", vertexCount (over-estimated): " << vertexCount << std::endl;

    CHECK(cudaMalloc(&dVertexMap, vertexCount * sizeof(int)));
    CHECK(cudaMalloc(&dTotalVertexNum, sizeof(int)));
    CHECK(cudaMalloc(&dSizeArray, sizeof(int) * vertexCount));
    CHECK(cudaMalloc(&dDataArray, sizeof(int) * vertexCount * SFCP_ColumnNum));
    CHECK(cudaMemset(dVertexMap, -1, vertexCount * sizeof(int)));
    CHECK(cudaMemcpy(dTotalVertexNum, &hTotalVertexNum, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dSizeArray, 0, sizeof(int) * vertexCount));
    CHECK(cudaMemset(dDataArray, -1, sizeof(int) * vertexCount * SFCP_ColumnNum));
    
    computeVertexNeighborsWithExternals<<<1, 32>>>(
        dDataArray, dSizeArray,
        dVertexMap, dTotalVertexNum,
        clusterId, clusterNumber_,
        dConnectivity_, dOffsets_,
        dVertexIntervals_, dCellIntervals_,
        dExternalCells_, dExternalCellOffsets_, 
        cellDimension_, SFCP_ColumnNum);
    
    // Check for kernel errors
    CHECK(cudaMemcpy(&hTotalVertexNum, dTotalVertexNum, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hVertexMap, dVertexMap, sizeof(int) * hTotalVertexNum, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hSizeArray, dSizeArray, sizeof(int) * hTotalVertexNum, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hDataArray, dDataArray, sizeof(int) * hTotalVertexNum * vvColumnNum, cudaMemcpyDeviceToHost));

    std::cout << "Total vertex number (actual): " << hTotalVertexNum << std::endl;
    std::vector<SimplexId> vertexNeighborData{};
    std::vector<SimplexId> vertexNeighborOffsets(hTotalVertexNum + 1);
    vertexNeighborOffsets[0] = 0;
    int rvid = 0;
    for (int vid = 0; vid < hTotalVertexNum; ++vid) {
        vertexNeighborData.insert(vertexNeighborData.end(), hDataArray + rvid * vvColumnNum, hDataArray + rvid * vvColumnNum + hSizeArray[rvid]);
        vertexNeighborOffsets[vid+1] = vertexNeighborOffsets[vid] + hSizeArray[rvid];
        ++rvid;
    }
    
    // verify the relation computation
    for (int i = 0; i < hTotalVertexNum; ++i) {
        for (int j = vertexNeighborOffsets[i]; j < vertexNeighborOffsets[i+1]; ++j) {
            SimplexId localNeighborId = vertexNeighborData[j];
            bool exist = false;
            for (int k = vertexNeighborOffsets[localNeighborId]; k < vertexNeighborOffsets[localNeighborId+1]; ++k) {
                if (vertexNeighborData[k] == i) {
                    exist = true;
                    break;
                }
            }
            if (!exist) {
                std::cerr << "[Gale] Error: local vertex " << i << " missing neighbor " << localNeighborId << std::endl;
            }
        }
    }

    // Clean up device memory
    CHECK(cudaFree(dVertexMap));
    CHECK(cudaFree(dTotalVertexNum));
    CHECK(cudaFree(dSizeArray));
    CHECK(cudaFree(dDataArray));
    
    // Clean up host memory
    delete[] hVertexMap;
    delete[] hSizeArray;
    delete[] hDataArray;
}


/**
 * Compute relations when there is only one single consumer thread
 * WARNING: will be deprecated later
*/
int Gale::computeRelation(const RelationType relation,
                        const SimplexId clusterId,
                        bool global,
                        const int threadId,
                        bool signal) {

    // std::cout << "computing cluster " << clusterId << " relation " << static_cast<int>(relation) << std::endl;
    int res = 0;
    int relationId = static_cast<int>(relation);
    int numBlocks = 0, endCluster = 0;
    if (global) {
        numBlocks = clusterNumber_ / (blockSize_ / threadsPerCluster_) + 1;
        endCluster = clusterNumber_ - 1;
    }
    else {
        numBlocks = blocksPerThread_;
        endCluster = std::min(clusterId + numBlocks * (blockSize_ / threadsPerCluster_), clusterNumber_) - 1;
    }

    // different relations correspond to different GPU kernels
    switch (relation) {
        case RelationType::BERelation:
        {
            int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
            int localEdgeNum = hEdgeIntervals_[endCluster] - edgeIdOffset;
            bool *dBoundaryEdgeList;
            CHECK(cudaMalloc((void **)&dBoundaryEdgeList, sizeof(bool) * localEdgeNum));
            CHECK(cudaMemset(dBoundaryEdgeList, false, sizeof(bool) * localEdgeNum));

            computeBoundaryEdges<<<numBlocks, blockSize_>>>(
                dEdgeList_, dTriangleList_, 
                dBoundaryEdgeList, dBoundaryTriangles_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            bool *hBoundaryEdgeList = new bool[localEdgeNum];
            CHECK(cudaMemcpy(hBoundaryEdgeList, dBoundaryEdgeList, sizeof(bool) * localEdgeNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(BEMutex_);
            for (int i = clusterId; i <= endCluster; ++i) {
                if (allClusters_[i].boundaryEdges_.empty()) {
                    allClusters_[i].boundaryEdges_.resize(hEdgeIntervals_[i] - hEdgeIntervals_[i-1]);
                    int idOffset = hEdgeIntervals_[i-1] - edgeIdOffset;
                    for (int j = 0; j < allClusters_[i].boundaryEdges_.size(); ++j) {
                        allClusters_[i].boundaryEdges_[j] = hBoundaryEdgeList[j + idOffset];
                    }
                }
            }

            // lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            CHECK(cudaFree(dBoundaryEdgeList));
            delete[] hBoundaryEdgeList;

            break;
        }

        case RelationType::BVRelation: 
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
            bool *dBoundaryVertexList;
            int *dTriangleList; 
            CHECK(cudaMalloc((void **)&dTriangleList, sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3));
            CHECK(cudaMalloc((void **)&dBoundaryVertexList, sizeof(bool) * localVertexNum));
            CHECK(cudaMemcpy(dTriangleList, triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[endCluster] * 3, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(dBoundaryVertexList, false, sizeof(bool) * localVertexNum));

            computeBoundaryVertices<<<numBlocks, blockSize_>>>(
                dTriangleList, dBoundaryTriangles_, dBoundaryVertexList,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            bool *hBoundaryVertexList = new bool[localVertexNum];
            CHECK(cudaMemcpy(hBoundaryVertexList, dBoundaryVertexList, sizeof(bool) * localVertexNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(BVMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].boundaryVertices_.empty()) {
                    allClusters_[sid].boundaryVertices_.resize(hVertexIntervals_[sid] - hVertexIntervals_[sid-1]);
                    int rvid = hVertexIntervals_[sid-1] - vertexIdOffset;
                    for (int vid = 0; vid < allClusters_[sid].boundaryVertices_.size(); ++vid) {
                        allClusters_[sid].boundaryVertices_[vid] = hBoundaryVertexList[rvid++];
                    }
                }
            }

            // lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            CHECK(cudaFree(dTriangleList));
            CHECK(cudaFree(dBoundaryVertexList));

            delete[] hBoundaryVertexList;
            break;
        }

        case RelationType::BFRelation:
        {
            int *dTriangleStarMatrix, *dTriangleStarSizes;
            int localTriangleNum = triangleNumber_;
            CHECK(cudaMalloc((void **)&dBoundaryTriangles_, sizeof(bool) * localTriangleNum));
            CHECK(cudaMalloc((void **)&dTriangleStarSizes, sizeof(int) * localTriangleNum));
            CHECK(cudaMalloc((void **)&dTriangleStarMatrix, sizeof(int) * localTriangleNum * ftColumnNum));
            CHECK(cudaMemset(dBoundaryTriangles_, false, sizeof(bool) * localTriangleNum));
            CHECK(cudaMemset(dTriangleStarSizes, 0, sizeof(int) * localTriangleNum));
            CHECK(cudaMemset(dTriangleStarMatrix, -1, sizeof(int) * localTriangleNum * ftColumnNum));

            computeBoundaryTriangles<<<numBlocks, blockSize_>>>(
                dTriangleList_, dTriangleStarMatrix, 
                dTriangleStarSizes, dBoundaryTriangles_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // cannot copy directly to vector<bool>, since it uses one bit for a boolean instead of one byte
            bool *hBoundaryTriangleList = new bool[localTriangleNum];
            CHECK(cudaMemcpy(hBoundaryTriangleList, dBoundaryTriangles_, sizeof(bool) * localTriangleNum, cudaMemcpyDeviceToHost));
            for (int fid = 0; fid < localTriangleNum; ++fid) {
                boundaryTriangles_[fid] = hBoundaryTriangleList[fid];
            }

            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }

            CHECK(cudaFree(dTriangleStarSizes));
            CHECK(cudaFree(dTriangleStarMatrix));
            delete[] hBoundaryTriangleList;

            break;
        }

        case RelationType::EFRelation:
        {
            int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
            int localEdgeSum = hEdgeIntervals_[endCluster] - edgeIdOffset;
            CHECK(cudaMemset(relationCompVecs_[0].dSizeArray, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(relationCompVecs_[0].dDataArray, -1, sizeof(int) * localEdgeSum * efColumnNum));

            computeEdgeTriangles<<<numBlocks, blockSize_>>>(
                dEdgeList_, dTriangleList_,
                relationCompVecs_[0].dDataArray, relationCompVecs_[0].dSizeArray,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            CHECK(cudaMemcpy(relationCompVecs_[0].hSizeArray, relationCompVecs_[0].dSizeArray, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(relationCompVecs_[0].hDataArray, relationCompVecs_[0].dDataArray, sizeof(int) * localEdgeSum * efColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(EFMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].edgeTriangles_.empty()) {
                    int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                    std::vector<SimplexId> edgeTriangleData{};
                    std::vector<SimplexId> edgeTriangleOffsets(localEdgeNum + 1);
                    edgeTriangleOffsets[0] = 0;
                    int reid = hEdgeIntervals_[sid - 1] - edgeIdOffset;
                    for (int eid = 0; eid < localEdgeNum; ++eid) {
                        edgeTriangleData.insert(edgeTriangleData.end(), relationCompVecs_[0].hDataArray + reid * efColumnNum, relationCompVecs_[0].hDataArray + reid * efColumnNum + relationCompVecs_[0].hSizeArray[reid]);
                        edgeTriangleOffsets[eid+1] = edgeTriangleOffsets[eid] + relationCompVecs_[0].hSizeArray[reid];
                        ++reid;
                    }
                    allClusters_[sid].edgeTriangles_.setData(std::move(edgeTriangleData), std::move(edgeTriangleOffsets));
                }
            }

            // lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            break;
        }

        case RelationType::ETRelation:
        {
            int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
            int localEdgeSum = hEdgeIntervals_[endCluster] - edgeIdOffset;
            CHECK(cudaMemset(relationCompVecs_[0].dSizeArray, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(relationCompVecs_[0].dDataArray, -1, sizeof(int) * localEdgeSum * etColumnNum));

            computeEdgeStars<<<numBlocks, blockSize_>>>(
                dEdgeList_, relationCompVecs_[0].dDataArray, relationCompVecs_[0].dSizeArray,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            CHECK(cudaMemcpy(relationCompVecs_[0].hSizeArray, relationCompVecs_[0].dSizeArray, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(relationCompVecs_[0].hDataArray, relationCompVecs_[0].dDataArray, sizeof(int) * localEdgeSum * etColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(ETMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].edgeStars_.empty()) {
                    int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                    std::vector<SimplexId> edgeStarData{};
                    std::vector<SimplexId> edgeStarOffsets(localEdgeNum + 1);
                    edgeStarOffsets[0] = 0;
                    int reid = hEdgeIntervals_[sid - 1] - edgeIdOffset;
                    for (int eid = 0; eid < localEdgeNum; ++eid) {
                        edgeStarData.insert(edgeStarData.end(), relationCompVecs_[0].hDataArray + reid * etColumnNum, relationCompVecs_[0].hDataArray + reid * etColumnNum + relationCompVecs_[0].hSizeArray[reid]);
                        edgeStarOffsets[eid+1] = edgeStarOffsets[eid] + relationCompVecs_[0].hSizeArray[reid];
                        ++reid;
                    }
                    allClusters_[sid].edgeStars_.setData(std::move(edgeStarData), std::move(edgeStarOffsets));
                }
            }

            // lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            break;
        }

        case RelationType::FTRelation:
        {
            int triangleIdOffset = hTriangleIntervals_[clusterId - 1];
            int localTriangleSum = hTriangleIntervals_[endCluster] - triangleIdOffset;
            CHECK(cudaMemset(relationCompVecs_[0].dSizeArray, 0, sizeof(int) * localTriangleSum));
            CHECK(cudaMemset(relationCompVecs_[0].dDataArray, -1, sizeof(int) * localTriangleSum * ftColumnNum));

            computeTriangleStars<<<numBlocks, blockSize_>>>(
                dTriangleList_, relationCompVecs_[0].dDataArray, relationCompVecs_[0].dSizeArray,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            CHECK(cudaMemcpy(relationCompVecs_[0].hSizeArray, relationCompVecs_[0].dSizeArray, sizeof(int) * localTriangleSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(relationCompVecs_[0].hDataArray, relationCompVecs_[0].dDataArray, sizeof(int) * localTriangleSum * ftColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(FTMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].triangleStars_.empty()) {
                    int localTriangleNum = hTriangleIntervals_[sid] - hTriangleIntervals_[sid - 1];
                    std::vector<SimplexId> triangleStarData{};
                    std::vector<SimplexId> triangleStarOffsets(localTriangleNum + 1);
                    triangleStarOffsets[0] = 0;
                    int rfid = hTriangleIntervals_[sid - 1] - triangleIdOffset;
                    for (int fid = 0; fid < localTriangleNum; ++fid) {
                        triangleStarData.insert(triangleStarData.end(), relationCompVecs_[0].hDataArray + rfid * ftColumnNum, relationCompVecs_[0].hDataArray + rfid * ftColumnNum + relationCompVecs_[0].hSizeArray[rfid]);
                        triangleStarOffsets[fid+1] = triangleStarOffsets[fid] + relationCompVecs_[0].hSizeArray[rfid];
                        ++rfid;
                    }
                    allClusters_[sid].triangleStars_.setData(std::move(triangleStarData), std::move(triangleStarOffsets));
                }
            }

            // lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            break;
        }

        case RelationType::VERelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexSum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMemset(relationCompVecs_[0].dSizeArray, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(relationCompVecs_[0].dDataArray, -1, sizeof(int) * localVertexSum * veColumnNum));

            computeVertexEdges<<<numBlocks, blockSize_>>>(
                dEdgeList_, relationCompVecs_[0].dDataArray, relationCompVecs_[0].dSizeArray,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            CHECK(cudaMemcpy(relationCompVecs_[0].hSizeArray, relationCompVecs_[0].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(relationCompVecs_[0].hDataArray, relationCompVecs_[0].dDataArray, sizeof(int) * localVertexSum * veColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VEMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexEdges_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexEdgeData{};
                    std::vector<SimplexId> vertexEdgeOffsets(localVertexNum + 1);
                    vertexEdgeOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexEdgeData.insert(vertexEdgeData.end(), relationCompVecs_[0].hDataArray + rvid * veColumnNum, relationCompVecs_[0].hDataArray + rvid * veColumnNum + relationCompVecs_[0].hSizeArray[rvid]);
                        vertexEdgeOffsets[vid+1] = vertexEdgeOffsets[vid] + relationCompVecs_[0].hSizeArray[rvid];
                        ++rvid;
                    }
                    allClusters_[sid].vertexEdges_.setData(std::move(vertexEdgeData), std::move(vertexEdgeOffsets));
                }
            }

            // lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            break;
        }

        case RelationType::VFRelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexSum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMemset(relationCompVecs_[0].dSizeArray, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(relationCompVecs_[0].dDataArray, -1, sizeof(int) * localVertexSum * vfColumnNum));

            computeVertexTriangles<<<numBlocks, blockSize_>>>(
                dTriangleList_, relationCompVecs_[0].dDataArray, relationCompVecs_[0].dSizeArray,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            CHECK(cudaMemcpy(relationCompVecs_[0].hSizeArray, relationCompVecs_[0].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(relationCompVecs_[0].hDataArray, relationCompVecs_[0].dDataArray, sizeof(int) * localVertexSum * vfColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VFMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexTriangles_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexTriangleData{};
                    std::vector<SimplexId> vertexTriangleOffsets(localVertexNum + 1);
                    vertexTriangleOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexTriangleData.insert(vertexTriangleData.end(), relationCompVecs_[0].hDataArray + rvid * vfColumnNum, relationCompVecs_[0].hDataArray + rvid * vfColumnNum + relationCompVecs_[0].hSizeArray[rvid]);
                        vertexTriangleOffsets[vid+1] = vertexTriangleOffsets[vid] + relationCompVecs_[0].hSizeArray[rvid];
                        ++rvid;
                    }
                    allClusters_[sid].vertexTriangles_.setData(std::move(vertexTriangleData), std::move(vertexTriangleOffsets));
                }
            }

            // lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            break;
        }

        case RelationType::VTRelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMemset(relationCompVecs_[0].dSizeArray, 0, sizeof(int) * localVertexNum));
            CHECK(cudaMemset(relationCompVecs_[0].dDataArray, -1, sizeof(int) * localVertexNum * vtColumnNum));

            computeVertexStars<<<numBlocks, blockSize_>>>(
                relationCompVecs_[0].dDataArray, relationCompVecs_[0].dSizeArray,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            CHECK(cudaMemcpy(relationCompVecs_[0].hSizeArray, relationCompVecs_[0].dSizeArray, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(relationCompVecs_[0].hDataArray, relationCompVecs_[0].dDataArray, sizeof(int) * localVertexNum * vtColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VTMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexStars_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexStarData{};
                    std::vector<SimplexId> vertexStarOffsets(localVertexNum + 1);
                    vertexStarOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexStarData.insert(vertexStarData.end(), relationCompVecs_[0].hDataArray + rvid * vtColumnNum, relationCompVecs_[0].hDataArray + rvid * vtColumnNum + relationCompVecs_[0].hSizeArray[rvid]);
                        vertexStarOffsets[vid+1] = vertexStarOffsets[vid] + relationCompVecs_[0].hSizeArray[rvid];
                        ++rvid;
                    }
                    allClusters_[sid].vertexStars_.setData(std::move(vertexStarData), std::move(vertexStarOffsets));
                }
            }
            // lck.unlock();
            if (signal) {
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;

            break;
        }

        case RelationType::VVRelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMemset(relationCompVecs_[0].dSizeArray, 0, sizeof(int) * localVertexNum));
            CHECK(cudaMemset(relationCompVecs_[0].dDataArray, -1, sizeof(int) * localVertexNum * SFCP_ColumnNum));
            
            computeVertexNeighbors<<<numBlocks, blockSize_>>>(
                relationCompVecs_[0].dDataArray, relationCompVecs_[0].dSizeArray,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_,
                SFCP_ColumnNum);

            CHECK(cudaMemcpy(relationCompVecs_[0].hSizeArray, relationCompVecs_[0].dSizeArray, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(relationCompVecs_[0].hDataArray, relationCompVecs_[0].dDataArray, sizeof(int) * localVertexNum * vvColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VVMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexNeighbors_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexNeighborData{};
                    std::vector<SimplexId> vertexNeighborOffsets(localVertexNum + 1);
                    vertexNeighborOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexNeighborData.insert(vertexNeighborData.end(), relationCompVecs_[0].hDataArray + rvid * vvColumnNum, relationCompVecs_[0].hDataArray + rvid * vvColumnNum + relationCompVecs_[0].hSizeArray[rvid]);
                        vertexNeighborOffsets[vid+1] = vertexNeighborOffsets[vid] + relationCompVecs_[0].hSizeArray[rvid];
                        ++rvid;
                    }
                    allClusters_[sid].vertexNeighbors_.setData(std::move(vertexNeighborData), std::move(vertexNeighborOffsets));
                }
            }
            // lck.unlock();

            if (signal) {
            // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
            // else res = 1;
                sem_post(&relationSems_[threadId]);
            }
            if (baseClusters_[relationId][threadId] == 0)
                baseClusters_[relationId][threadId] = clusterId;
            
            break;
        }

        default:
            res = -1;
    }

    return res;
}

/**
 * Compute topological relations for multiple consumer.
*/
int Gale::computeCombinedRelation(const std::vector<int> &threadIds,
            const int producerId = 0) {

    // std::cout << "request size: " << requestMap.size() << ": [";
    // for (auto &e : requestMap) {
    //     std::cout << static_cast<int>(e.first) << "(" << threadIds.size() << ") ";
    // }
    // std::cout << "]" << std::endl;

    int res = 0;

    #ifdef MEASURE_TIME
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    #endif

    // launch GPU kernels for the topological relation
    Timer t;
    RelationType &relation = preconditionedRelations_[producerId];
    int threadNum = threadIds.size();
    std::vector<SimplexId> hEndClusters(threadNum);
    int hClusterIds[threadNum];
    int hSimplexOffsets[threadNum];

    // launches kernels on different streams
    switch (relation) {
        case RelationType::BERelation:
        {
            int localEdgeSum = 0;
            // int maxEndCluster = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localEdgeSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                // maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                int localEdgeNum = hEdgeIntervals_[hEndClusters[i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                localEdgeSum += localEdgeNum;
            }
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, 0, sizeof(int) * localEdgeSum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            // data perparation time
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiBoundaryEdges<<<dimBlock, blockSize_>>>(
                dEdgeList_, dTriangleList_,
                dBoundaryTriangles_, relationCompVecs_[producerId].dDataArray, 
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        case RelationType::BVRelation: 
        {
            int localVertexSum = 0;
            // int maxEndCluster = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localVertexSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                // maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                int localVertexNum = hVertexIntervals_[hEndClusters[i]] - hVertexIntervals_[hClusterIds[i] - 1];
                localVertexSum += localVertexNum;
            }

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, 0, sizeof(int) * localVertexSum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiBoundaryVertices<<<dimBlock, blockSize_>>>(
                dTriangleList_, dBoundaryTriangles_, relationCompVecs_[producerId].dDataArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets, 
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }
        
        case RelationType::EFRelation:
        {
            int localEdgeSum = 0;
            // int maxEndCluster = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localEdgeSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                // maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                int localEdgeNum = hEdgeIntervals_[hEndClusters[i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                localEdgeSum += localEdgeNum;
            }
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dSizeArray, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, -1, sizeof(int) * localEdgeSum * efColumnNum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiEdgeTriangles<<<dimBlock, blockSize_>>>(
                dEdgeList_, dTriangleList_,
                relationCompVecs_[producerId].dDataArray, relationCompVecs_[producerId].dSizeArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hSizeArray, relationCompVecs_[producerId].dSizeArray, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localEdgeSum * efColumnNum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        case RelationType::ETRelation:
        {
            int localEdgeSum = 0;
            // int maxEndCluster = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localEdgeSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                // maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                int localEdgeNum = hEdgeIntervals_[hEndClusters[i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                localEdgeSum += localEdgeNum;
            }

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dSizeArray, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, -1, sizeof(int) * localEdgeSum * etColumnNum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiEdgeStars<<<dimBlock, blockSize_>>>(
                dEdgeList_, relationCompVecs_[producerId].dDataArray, relationCompVecs_[producerId].dSizeArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets, 
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hSizeArray, relationCompVecs_[producerId].dSizeArray, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localEdgeSum * etColumnNum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        case RelationType::FTRelation:
        {
            int localTriangleSum = 0;
            // int maxEndCluster = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localTriangleSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                // maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                int localTriangleNum = hTriangleIntervals_[hEndClusters[i]] - hTriangleIntervals_[hClusterIds[i] - 1];
                localTriangleSum += localTriangleNum;
            }

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dSizeArray, 0, sizeof(int) * localTriangleSum));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, -1, sizeof(int) * localTriangleSum * ftColumnNum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiTriangleStars<<<dimBlock, blockSize_>>>(
                dTriangleList_, relationCompVecs_[producerId].dDataArray, relationCompVecs_[producerId].dSizeArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hSizeArray, relationCompVecs_[producerId].dSizeArray, sizeof(int) * localTriangleSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localTriangleSum * ftColumnNum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        case RelationType::VERelation:
        {
            int localVertexSum = 0;
            // int maxEndCluster = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localVertexSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                // maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                int localVertexNum = hVertexIntervals_[hEndClusters[i]] - hVertexIntervals_[hClusterIds[i] - 1];
                localVertexSum += localVertexNum;
            }

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dSizeArray, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, -1, sizeof(int) * localVertexSum * veColumnNum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiVertexEdges<<<dimBlock, blockSize_>>>(
                dEdgeList_, relationCompVecs_[producerId].dDataArray, relationCompVecs_[producerId].dSizeArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif
            
            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hSizeArray, relationCompVecs_[producerId].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localVertexSum * veColumnNum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        case RelationType::VFRelation:
        {
            int localVertexSum = 0;
            // int maxEndCluster = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localVertexSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                // maxEndCluster = std::max(hEndClusters[i], maxEndCluster);
                int localVertexNum = hVertexIntervals_[hEndClusters[i]] - hVertexIntervals_[hClusterIds[i] - 1];
                localVertexSum += localVertexNum;
            }

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dSizeArray, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, -1, sizeof(int) * localVertexSum * vfColumnNum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiVertexTriangles<<<dimBlock, blockSize_>>>(
                dTriangleList_, relationCompVecs_[producerId].dDataArray, relationCompVecs_[producerId].dSizeArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);
            
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hSizeArray, relationCompVecs_[producerId].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localVertexSum * vfColumnNum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        case RelationType::VTRelation:
        {
            int localVertexSum = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localVertexSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                int localVertexNum = hVertexIntervals_[hEndClusters[i]] - hVertexIntervals_[hClusterIds[i] - 1];
                localVertexSum += localVertexNum;
            }

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dSizeArray, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, -1, sizeof(int) * localVertexSum * vtColumnNum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiVertexStars<<<dimBlock, blockSize_>>>(
                relationCompVecs_[producerId].dDataArray, relationCompVecs_[producerId].dSizeArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);
            
            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hSizeArray, relationCompVecs_[producerId].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localVertexSum * vtColumnNum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        case RelationType::VVRelation:
        {
            int localVertexSum = 0;
            for (int i = 0; i < threadNum; ++i) {
                hSimplexOffsets[i] = localVertexSum;
                hClusterIds[i] = currClusters_[threadIds[i]];
                hEndClusters[i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                int localVertexNum = hVertexIntervals_[hEndClusters[i]] - hVertexIntervals_[hClusterIds[i] - 1];
                localVertexSum += localVertexNum;
            }

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].dSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(relationCompVecs_[producerId].dSizeArray, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(relationCompVecs_[producerId].dDataArray, -1, sizeof(int) * localVertexSum * SFCP_ColumnNum));

            dim3 dimBlock(threadNum, blocksPerThread_);
            #ifdef MEASURE_TIME
            for (int i = 0; i < threadNum; ++i) {
                waitingTimes_[threadIds[i]][3] += t.getElapsedTime();
            }
            cudaEventRecord(startEvent);
            #endif
            computeMultiVertexNeighbors<<<dimBlock, blockSize_>>>(
                relationCompVecs_[producerId].dDataArray, relationCompVecs_[producerId].dSizeArray,
                clusterNumber_, relationCompVecs_[producerId].dClusterIds, relationCompVecs_[producerId].dSimplexOffsets,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);
            
            // #ifdef MEASURE_TIME
            // cudaEventRecord(endEvent);
            // #endif

            memcpy(relationCompVecs_[producerId].hSimplexOffsets, hSimplexOffsets, sizeof(int) * threadNum);

            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hSizeArray, relationCompVecs_[producerId].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpyAsync(relationCompVecs_[producerId].hDataArray, relationCompVecs_[producerId].dDataArray, sizeof(int) * localVertexSum * vvColumnNum, cudaMemcpyDeviceToHost));
            #ifdef MEASURE_TIME
            cudaEventRecord(endEvent);
            #endif

            break;
        }

        default:
            res = -1;
    }

    // Get the relation arrays from the GPU
    // (relation, [(tid, clusterId), ...])
    cudaError_t status = cudaStreamSynchronize(0);
    if (status != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
            __FILE__, __LINE__); if (abort) exit(status);
    }

    #ifdef MEASURE_TIME
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
    // kernel time
    if (startTracking_) {
        for (int i = 0; i < threadNum; ++i) {
            waitingTimes_[threadIds[i]][4] += milliseconds / 1000;
        }
    }
    Timer tmp;
    #endif

    // transform the data 
    int relationId = static_cast<int>(relation);
    switch (relation) {
        case RelationType::BERelation: 
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (int sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    if (allClusters_[sid].boundaryEdges_.empty()) {
                        int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                        int relativeEid = hEdgeIntervals_[sid - 1] - hEdgeIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                        allClusters_[sid].boundaryEdges_.resize(localEdgeNum);
                        for (int eid = 0; eid < localEdgeNum; ++eid) {
                            allClusters_[sid].boundaryEdges_[eid] = relationCompVecs_[producerId].hDataArray[relativeEid++];
                        }
                    }
                }
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                // data transfer time
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0)
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }
        case RelationType::BVRelation:
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (int sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    if (allClusters_[sid].boundaryVertices_.empty()) {
                        int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid-1];
                        int relativeVid = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                        allClusters_[sid].boundaryVertices_.resize(localVertexNum);
                        for (int vid = 0; vid < localVertexNum; ++vid) {
                            allClusters_[sid].boundaryVertices_[vid] = relationCompVecs_[producerId].hDataArray[relativeVid++];
                        }
                    }
                }
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0)
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }
        
        case RelationType::EFRelation:
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (SimplexId sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    if (allClusters_[sid].edgeTriangles_.empty()) {
                        std::vector<SimplexId> edgeTriangleData{};
                        int reid = hEdgeIntervals_[sid - 1] - hEdgeIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                        int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                        std::vector<SimplexId> edgeTriangleOffsets(localEdgeNum + 1);
                        edgeTriangleOffsets[0] = 0;
                        for (int eid = 0; eid < localEdgeNum; ++eid) {
                            edgeTriangleData.insert(edgeTriangleData.end(), 
                            relationCompVecs_[producerId].hDataArray + reid * efColumnNum,
                            relationCompVecs_[producerId].hDataArray + reid * efColumnNum + relationCompVecs_[producerId].hSizeArray[reid]);
                            edgeTriangleOffsets[eid+1] = edgeTriangleOffsets[eid] + relationCompVecs_[producerId].hSizeArray[reid];
                            ++reid;
                        }
                        allClusters_[sid].edgeTriangles_.setData(std::move(edgeTriangleData), std::move(edgeTriangleOffsets));
                    }
                }
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0) 
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }
        
        case RelationType::ETRelation: 
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (SimplexId sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    if (allClusters_[sid].edgeStars_.empty()) {
                        std::vector<SimplexId> edgeStarData{};
                        int reid = hEdgeIntervals_[sid - 1] - hEdgeIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                        int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                        std::vector<SimplexId> edgeStarOffsets(localEdgeNum + 1);
                        edgeStarOffsets[0] = 0;
                        for (int j = 0; j < localEdgeNum; ++j) {
                            edgeStarData.insert(edgeStarData.end(), 
                            relationCompVecs_[producerId].hDataArray + reid * etColumnNum,
                            relationCompVecs_[producerId].hDataArray + reid * etColumnNum + relationCompVecs_[producerId].hSizeArray[reid]);
                            edgeStarOffsets[j+1] = edgeStarOffsets[j] + relationCompVecs_[producerId].hSizeArray[reid];
                            ++reid;
                        }
                        allClusters_[sid].edgeStars_.setData(std::move(edgeStarData), std::move(edgeStarOffsets));
                    }
                }
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0) 
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }

        case RelationType::FTRelation:
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (SimplexId sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    if (allClusters_[sid].triangleStars_.empty()) {
                        std::vector<SimplexId> triangleStarData{};
                        int reid = hTriangleIntervals_[sid - 1] - hTriangleIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                        int localTriangleNum = hTriangleIntervals_[sid] - hTriangleIntervals_[sid - 1];
                        std::vector<SimplexId> triangleStarOffsets(localTriangleNum + 1);
                        triangleStarOffsets[0] = 0;
                        for (int j = 0; j < localTriangleNum; ++j) {
                            triangleStarData.insert(triangleStarData.end(), 
                            relationCompVecs_[producerId].hDataArray + reid * ftColumnNum,
                            relationCompVecs_[producerId].hDataArray + reid * ftColumnNum + relationCompVecs_[producerId].hSizeArray[reid]);
                            triangleStarOffsets[j+1] = triangleStarOffsets[j] + relationCompVecs_[producerId].hSizeArray[reid];
                            ++reid;
                        }
                        allClusters_[sid].triangleStars_.setData(std::move(triangleStarData), std::move(triangleStarOffsets));
                    }
                }
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0)
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }

        case RelationType::VERelation:
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                // std::unique_lock<std::mutex> lck(VVMutex_);
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (SimplexId sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    if (allClusters_[sid].vertexEdges_.empty()) {
                        std::vector<SimplexId> vertexEdgeData{};
                        int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                        int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                        std::vector<SimplexId> vertexEdgeOffsets(localVertexNum + 1);
                        vertexEdgeOffsets[0] = 0;
                        for (int j = 0; j < localVertexNum; ++j) {
                            int rj = j + diff; // relative j value in the large matrix
                            vertexEdgeData.insert(vertexEdgeData.end(), 
                            relationCompVecs_[producerId].hDataArray + rj * veColumnNum,
                            relationCompVecs_[producerId].hDataArray + rj * veColumnNum + relationCompVecs_[producerId].hSizeArray[rj]);
                            vertexEdgeOffsets[j+1] = vertexEdgeOffsets[j] + relationCompVecs_[producerId].hSizeArray[rj];
                        }
                        allClusters_[sid].vertexEdges_.setData(std::move(vertexEdgeData), std::move(vertexEdgeOffsets));
                    }
                }
                // lck.unlock();
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0)
                    baseClusters_[relationId][tid] = baseClusterId;
            }


            break;
        }
        
        case RelationType::VFRelation:
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                // std::unique_lock<std::mutex> lck(VVMutex_);
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (SimplexId sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    if (allClusters_[sid].vertexTriangles_.empty()) {
                        std::vector<SimplexId> vertexTriangleData{};
                        int rvid = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                        int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                        std::vector<SimplexId> vertexTriangleOffsets(localVertexNum + 1);
                        vertexTriangleOffsets[0] = 0;
                        for (int vid = 0; vid < localVertexNum; ++vid) {
                            vertexTriangleData.insert(vertexTriangleData.end(), 
                            relationCompVecs_[producerId].hDataArray + rvid * vfColumnNum,
                            relationCompVecs_[producerId].hDataArray + rvid * vfColumnNum + relationCompVecs_[producerId].hSizeArray[rvid]);
                            vertexTriangleOffsets[vid+1] = vertexTriangleOffsets[vid] + relationCompVecs_[producerId].hSizeArray[rvid];
                            ++rvid;
                        }
                        allClusters_[sid].vertexTriangles_.setData(std::move(vertexTriangleData), std::move(vertexTriangleOffsets));
                    }
                }
                // lck.unlock();
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0)
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }
        
        case RelationType::VTRelation:
        {
            

            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                // std::unique_lock<std::mutex> lck(VVMutex_);
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (SimplexId sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    std::vector<SimplexId> vertexStarData{};
                    int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexStarOffsets(localVertexNum + 1);
                    vertexStarOffsets[0] = 0;
                    for (int j = 0; j < localVertexNum; ++j) {
                        int rj = j + diff; // relative j value in the large matrix
                        vertexStarData.insert(vertexStarData.end(), 
                        relationCompVecs_[producerId].hDataArray + rj * vtColumnNum,
                        relationCompVecs_[producerId].hDataArray + rj * vtColumnNum + relationCompVecs_[producerId].hSizeArray[rj]);
                        vertexStarOffsets[j+1] = vertexStarOffsets[j] + relationCompVecs_[producerId].hSizeArray[rj];
                    }
                    if (allClusters_[sid].vertexStars_.empty()) {
                        allClusters_[sid].vertexStars_.setData(std::move(vertexStarData), std::move(vertexStarOffsets));
                    }
                }
                // lck.unlock();
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0)
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }

        case RelationType::VVRelation:
        {
            // #ifdef TTK_ENABLE_OPENMP
            // #pragma omp parallel for num_threads(threadNum)
            // #endif
            for (int i = 0; i < threadNum; ++i) {
                SimplexId tid = threadIds[i];
                SimplexId baseClusterId = hClusterIds[i];
                for (SimplexId sid = baseClusterId; sid <= hEndClusters[i]; ++sid) {
                    // boost::unique_lock<boost::shared_mutex> writeLock(clusterMutexes_[sid]);
                    std::vector<SimplexId> vertexNeighborData{};
                    int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[i];
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexNeighborOffsets(localVertexNum + 1);
                    vertexNeighborOffsets[0] = 0;
                    for (int j = 0; j < localVertexNum; ++j) {
                        int rj = j + diff; // relative j value in the large matrix
                        vertexNeighborData.insert(vertexNeighborData.end(), 
                        relationCompVecs_[producerId].hDataArray + rj * SFCP_ColumnNum,
                        relationCompVecs_[producerId].hDataArray + rj * SFCP_ColumnNum + relationCompVecs_[producerId].hSizeArray[rj]);
                        vertexNeighborOffsets[j+1] = vertexNeighborOffsets[j] + relationCompVecs_[producerId].hSizeArray[rj];
                    }
                    if (allClusters_[sid].vertexNeighbors_.empty()) {
                        allClusters_[sid].vertexNeighbors_.setData(std::move(vertexNeighborData), std::move(vertexNeighborOffsets));
                    }
                    // writeLock.unlock();
                }
                sem_post(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                waitingTimes_[tid][5] += tmp.getElapsedTime();
                #endif
                if (baseClusters_[relationId][tid] == 0)
                    baseClusters_[relationId][tid] = baseClusterId;
            }

            break;
        }
    }

    return res;
}


/**
 * Procedure of the leader producer.
*/
void Gale::leaderProducer(const int pid) {
    volatile SimplexId leaderCluster = 0;
    volatile RelationType leaderFunc = RelationType::BlankRelation;
    // multiple threads are used
    // if (threadNumber_ > 1) {
        // initialization that executes only once
        // preallocate a large chunk of memory for GPU computation 
        // consumerCounters_[pid] = 0;
        RelationComputation relationComp;
        // pre-allocate the memory for the computation
        int vertexAllocNum = threadNumber_ * precompClusterNumber_ * 150;
        int edgeAllocNum = vertexAllocNum*5;
        int triangleAllocNum = vertexAllocNum*6;
        CHECK(cudaMallocAsync((void **)&relationComp.dClusterIds, sizeof(int) * threadNumber_, 0));
        CHECK(cudaMallocAsync((void **)&relationComp.dSimplexOffsets, sizeof(int) * threadNumber_, 0));
        switch (preconditionedRelations_[pid]) {
            case RelationType::BERelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * edgeAllocNum, 0));
            case RelationType::BVRelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * vertexAllocNum, 0));
            case RelationType::EFRelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dSizeArray, sizeof(int) * edgeAllocNum, 0));
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * edgeAllocNum * efColumnNum, 0));
                relationComp.hSizeArray = new int[edgeAllocNum];
                relationComp.hDataArray = new int[edgeAllocNum * efColumnNum];
                break;
            case RelationType::ETRelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dSizeArray, sizeof(int) * edgeAllocNum, 0));
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * edgeAllocNum * etColumnNum, 0));
                relationComp.hSizeArray = new int[edgeAllocNum];
                relationComp.hDataArray = new int[edgeAllocNum * etColumnNum];
                break;
            case RelationType::FTRelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dSizeArray, sizeof(int) * triangleAllocNum, 0));
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * triangleAllocNum * ftColumnNum, 0));
                relationComp.hSizeArray = new int[triangleAllocNum];
                relationComp.hDataArray = new int[triangleAllocNum * ftColumnNum];
                break;
            case RelationType::VERelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dSizeArray, sizeof(int) * vertexAllocNum, 0));
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * vertexAllocNum * veColumnNum, 0));
                relationComp.hSizeArray = new int[vertexAllocNum];
                relationComp.hDataArray = new int[vertexAllocNum * veColumnNum];
                break;
            case RelationType::VFRelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dSizeArray, sizeof(int) * vertexAllocNum, 0));
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * vertexAllocNum * vfColumnNum, 0));
                relationComp.hSizeArray = new int[vertexAllocNum];
                relationComp.hDataArray = new int[vertexAllocNum * vfColumnNum];
                break;
            case RelationType::VTRelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dSizeArray, sizeof(int) * vertexAllocNum, 0));
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * vertexAllocNum * vtColumnNum, 0));
                relationComp.hSizeArray = new int[vertexAllocNum];
                relationComp.hDataArray = new int[vertexAllocNum * vtColumnNum];
                break;
            case RelationType::VVRelation:
                CHECK(cudaMallocAsync((void **)&relationComp.dSizeArray, sizeof(int) * vertexAllocNum, 0));
                CHECK(cudaMallocAsync((void **)&relationComp.dDataArray, sizeof(int) * vertexAllocNum * SFCP_ColumnNum, 0));
                relationComp.hSizeArray = new int[vertexAllocNum];
                relationComp.hDataArray = new int[vertexAllocNum * SFCP_ColumnNum];
                break;
            default:
                break;
        }
        
        relationComp.hSimplexOffsets = new int[threadNumber_];
        {
            std::lock_guard<std::mutex> pLock(producerMutex_);
            relationCompVecs_[pid] = relationComp;
        }
        // std::vector<std::vector<std::tuple<int, int, RelationType>>> qstats;
        while (true) {
            /* Job queue version */
            // save threadIds in the current queue to the vector
            std::unique_lock<std::mutex> qlock(qMutexes_[pid]);
            std::vector<int> localQ;
            while (!jobQueues_[pid].empty()) {
                #ifdef MEASURE_TIME
                int tid = jobQueues_[pid].front().first;
                localQ.push_back(tid);
                // time for one request to be processed 
                waitingTimes_[tid][2] += jobQueues_[pid].front().second.getElapsedTime();
                #else
                localQ.push_back(jobQueues_[pid].front());
                #endif
                jobQueues_[pid].pop();
            }
            qlock.unlock();

            if (!localQ.empty()) {
                Timer t;
                // handle the waiting threadIds in the queue
                // combine the requests from all waiting consumers
                computeCombinedRelation(localQ, pid);
                // #ifdef MEASURE_TIME
                // qTimes_[pid].push_back(t.getElapsedTime());
                // qSizes_[pid].push_back(localQ.size());
                // #endif
            }
            else {
                // clean the memory storage
                for (int i = 0; i < threadNumber_; ++i) {
                    leaderCluster = currClusters_[i];
                    if (leaderCluster < 0) {
                        CHECK(cudaFree(relationComp.dClusterIds));
                        CHECK(cudaFree(relationComp.dSimplexOffsets));
                        CHECK(cudaFree(relationComp.dSizeArray));
                        CHECK(cudaFree(relationComp.dDataArray));
                        delete[] relationComp.hSizeArray;
                        delete[] relationComp.hDataArray;
                        delete[] relationComp.hSimplexOffsets;
                        return;
                    }
                    else if (leaderCluster == 0) continue;
                    if (!jobQueues_[pid].empty()) break;
                    RelationType relation = currRelations_[i];
                    SimplexId startCluster = baseClusters_[static_cast<int>(relation)][i];
                    // leaderCluster = (leaderCluster - startCluster) * removalRatio + startCluster;
                    if (startCluster == 0) continue;
                    // different relations correspond to different vectors
                    switch (relation)
                    {
                        case RelationType::BVRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].boundaryVertices_ = std::vector<bool>();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }

                        case RelationType::BERelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].boundaryEdges_ = std::vector<bool>();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }
                        
                        case RelationType::EFRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].edgeTriangles_.clear();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }
                        
                        case RelationType::ETRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].edgeStars_.clear();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }
                        
                        case RelationType::FTRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].triangleStars_.clear();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }

                        case RelationType::VVRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].vertexNeighbors_.clear();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }
                    
                        case RelationType::VERelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].vertexEdges_.clear();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }

                        case RelationType::VFRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].vertexTriangles_.clear();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }

                        case RelationType::VTRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                if (clusterMutexes_[cid].try_lock()) {
                                    allClusters_[cid].vertexStars_.clear();
                                    clusterMutexes_[cid].unlock();
                                }
                            }
                            break;
                        }
                        
                        default:
                            break;
                        
                    }
                    // reset the base cluster of the thread to 0
                    baseClusters_[static_cast<int>(relation)][i] = 0;
                }
            }
        }
    // }

    // else {
    //     while (true) {
    //         std::unique_lock<std::mutex> qlock(qMutexes_[pid]);
    //         int qSize = jobQueues_[pid].size();
    //         qlock.unlock();
    //         if (qSize > 0) {
    //             #ifdef MEASURE_TIME
    //             int tid = jobQueues_[pid].front().first;
    //             waitingTimes_[tid][1] += jobQueues_[pid].front().second.getElapsedTime();
    //             #else
    //             int tid = jobQueues_[pid].front();
    //             #endif
    //             leaderCluster = currClusters_[tid], leaderFunc = currRelations_[tid];
    //             int res = computeRelation(leaderFunc, leaderCluster, false, tid, true);
    //             if (res == 0) {
    //                 std::unique_lock<std::mutex> qlock(qMutexes_[pid]);
    //                 jobQueues_[pid].pop();
    //                 qlock.unlock();
    //             }
    //         }
    //         else {
    //             leaderCluster = currClusters_[0], leaderFunc = currRelations_[0];
    //             if (leaderCluster == -1) return;
    //             else if (leaderCluster == 0) continue;
                
    //             // SimplexId diff = leaderCluster - baseClusters_[static_cast<int>(leaderFunc)][i];
    //             // // check if the precomputation needs to be performed
    //             // if (diff < 0 || diff > blocksPerThread_ * blockSize_ * 0.6) {
    //             //     // std::cout << "precomputing for thread " << i << " cluster = " << leaderCluster << " diff = " << diff << std::endl;
    //             //     computeRelation(leaderFunc, leaderCluster, i, false);
    //             // }
                
    //             for (RelationType &relation : preconditionedRelations_) {
    //                 if (relation == currRelations_[0]) continue;
    //                 SimplexId startCluster = baseClusters_[static_cast<int>(relation)][0];
    //                 // leaderCluster = (leaderCluster - startCluster) * removalRatio + startCluster;
    //                 if (startCluster == 0) continue;
    //                 // different relations correspond to different vectors
    //                 switch (relation)
    //                 {
    //                     case RelationType::BVRelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].boundaryVertices_ = std::vector<bool>();
    //                         }
    //                         break;
    //                     }

    //                     case RelationType::BERelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].boundaryEdges_ = std::vector<bool>();
    //                         }
    //                         break;
    //                     }
                        
    //                     case RelationType::EFRelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].edgeTriangles_.clear();
    //                         }
    //                         break;
    //                     }
                        
    //                     case RelationType::ETRelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].edgeStars_.clear();
    //                         }
    //                         break;
    //                     }
                        
    //                     case RelationType::FTRelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].triangleStars_.clear();
    //                         }
    //                         break;
    //                     }

    //                     case RelationType::VVRelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].vertexNeighbors_.clear();
    //                         }
    //                         break;
    //                     }
                    
    //                     case RelationType::VERelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].vertexEdges_.clear();
    //                         }
    //                         break;
    //                     }

    //                     case RelationType::VFRelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].vertexTriangles_.clear();
    //                         }
    //                         break;
    //                     }

    //                     case RelationType::VTRelation:
    //                     {
    //                         for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
    //                             allClusters_[cid].vertexStars_.clear();
    //                         }
    //                         break;
    //                     }
                        
    //                     default:
    //                         break;
                        
    //                 }
    //                 // reset the base cluster of the thread to 0
    //                 baseClusters_[static_cast<int>(relation)][0] = 0;
    //             }
    //         }
    //     }
    // }
    
}

/**
 * Initialize GALE with custom parameters.
*/
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
        // convert to 32-bit storage to reduce memory consumption
        if(cells->IsStorage64Bit()) {
            if(cells->CanConvertTo32BitStorage()) {
                std::cout << "[Gale] Converting the cell array to 32-bit storage" << std::endl;
                bool success = cells->ConvertTo32BitStorage();
                if(!success) {
                    std::cerr << "[Gale] Error converting the provided cell array to 32-bit storage" << std::endl;
                }
            } else {
                std::cerr << "[Gale] Cannot convert the provided cell array to 32-bit storage" << std::endl;
            }
        }
        hConnectivity_ = static_cast<int *>(cells->GetConnectivityArray()->GetVoidPointer(0));
        hOffsets_ = static_cast<int *>(cells->GetOffsetsArray()->GetVoidPointer(0));
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
    if (blockNum > 0) blocksPerThread_ = blockNum;
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
    precompClusterNumber_ = blockSize_ / threadsPerCluster_ * blocksPerThread_;
    allClusters_ = std::vector<MeshSegment>(clusterNumber_);
    clusterMutexes_ = std::vector<boost::shared_mutex>(clusterNumber_);
    for (int cid = 1; cid < clusterNumber_; ++cid) {
        allClusters_[cid] = MeshSegment(cid);
    }

    printf("[Gale] GPU threads per cluster: %d, thread num: %d, block number: %d, block size: %d, cluster number: %d, cell number: %d, vertex number: %d\n", threadsPerCluster_, threadNumber_, blocksPerThread_, blockSize_, clusterNumber_, cellNumber_, vertexNumber_);

    // #ifdef MEASURE_TIME
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // #endif

    // copy input mesh arrays to the GPU
    CHECK(cudaMallocAsync(&dConnectivity_, sizeof(int) * (cellNumber_ * 4), 0));
    CHECK(cudaMemcpyAsync(dConnectivity_, hConnectivity_, sizeof(int) * (cellNumber_ * 4), cudaMemcpyHostToDevice));
    CHECK(cudaMallocAsync(&dOffsets_, sizeof(int) * (cellNumber_+1), 0));
    CHECK(cudaMemcpyAsync(dOffsets_, hOffsets_, sizeof(int) * (cellNumber_+1), cudaMemcpyHostToDevice));
    CHECK(cudaMallocAsync(&dIndexArray_, sizeof(int) * vertexNumber_, 0));
    CHECK(cudaMemcpyAsync(dIndexArray_, hIndexArray_, sizeof(int) * vertexNumber_, cudaMemcpyHostToDevice));

    // copy cell intervals and vertex intervals
    CHECK(cudaMemcpyAsync(dCellIntervals_, hCellIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(dExternalCellOffsets_, hExternalCellOffsets_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(dExternalCells_, hExternalCells_, sizeof(int) * hExternalCellOffsets_[clusterNumber_-1], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(dVertexIntervals_, hVertexIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice, 0));

    // #ifdef MEASURE_TIME
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // float elapsedTime;
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("[Gale] Cuda memory copy time measured for initialization: %f s\n", elapsedTime / 1000);
    // #endif
}

/**
 * Compute the cell interval array.
*/
void Gale::computeCellIntervals() {
    std::vector<std::vector<int>> externalCells(clusterNumber_, std::vector<int>());
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
    
    // Convert the 2D vector to 1D flat vector
    hExternalCells_ = new int[hExternalCellOffsets_[clusterNumber_-1]];
    int *tmp = hExternalCells_;
    CHECK(cudaMalloc((void **) &dExternalCells_, sizeof(int) * hExternalCellOffsets_[clusterNumber_-1]));
    for (int i = 1; i < externalCells.size(); ++i) {
        std::copy(externalCells[i].begin(), externalCells[i].end(), tmp);
        tmp += externalCells[i].size();
    }
}

/**
 * Compute the vertex interval array.
*/
void Gale::computeVertexIntervals() {

    CHECK(cudaMallocAsync(&dVertexIntervals_, sizeof(int) * clusterNumber_, 0));

    hVertexIntervals_ = (int *) calloc(clusterNumber_, sizeof(int));
    for (int vid = 0; vid < vertexNumber_; ++vid) {
        int clusterId = hIndexArray_[vid];
        hVertexIntervals_[clusterId]++;
    }
    for (int cid = 1; cid < clusterNumber_; ++cid) {
        hVertexIntervals_[cid] += hVertexIntervals_[cid-1];
    }
}

/**
 * Reindex the vertices to make the GALE work with the input segment.
*/
int Gale::reorderVertices(std::vector<SimplexId> &vertexMap) {
    CHECK(cudaMallocAsync(&dVertexIntervals_, sizeof(int) * clusterNumber_, 0));
    clusterVertices.resize(clusterNumber_);
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

/**
 * Reindex the cells to make the GALE work with the input segment.
*/
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

    int *newCellArray = new int[verticesPerCell * cellNumber_];
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
    memcpy(hConnectivity_, newCellArray, sizeof(int) * cellNumber_ * verticesPerCell);
    delete[] newCellArray;

    return 0;
}

/**
 * Compute all vertex neighbors using the GPU here.
 */
int Gale::preconditionVertexNeighborsInternal() {
    if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
        std::cerr << "Empty dataset, precondition skipped";
        return 1;
    }
    std::cout << "[Gale] Preconditioning vertex neighbors..." << std::endl;
    int numBlocks = clusterNumber_ / (blockSize_ / threadsPerCluster_) + 1, endCluster = clusterNumber_ - 1;
    CHECK(cudaMalloc(&dVVIndexArray_, sizeof(int) * vertexNumber_));
    CHECK(cudaMalloc(&dVVDataArray_, sizeof(int) * vertexNumber_ * SFCP_ColumnNum));
    CHECK(cudaMemset(dVVIndexArray_, 0, sizeof(int) * vertexNumber_));
    CHECK(cudaMemset(dVVDataArray_, -1, sizeof(int) * vertexNumber_ * SFCP_ColumnNum));

    std::cout << "[Gale] Number of blocks: " << numBlocks << std::endl;
    std::cout << "[Gale] cluster number: " << clusterNumber_ << std::endl;
    std::cout << "[Gale] vertex number: " << vertexNumber_ << std::endl;

    // int hSimplexOffsets[] = {0};
    // int hClusterIds[] = {1};
    // int *dClusterIds, *dSimplexOffsets;
    // CHECK(cudaMallocAsync(&dClusterIds, sizeof(int), 0));
    // CHECK(cudaMallocAsync(&dSimplexOffsets, sizeof(int), 0));
    // CHECK(cudaMemcpyAsync(dClusterIds, hClusterIds, sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpyAsync(dSimplexOffsets, hSimplexOffsets, sizeof(int), cudaMemcpyHostToDevice));

    // dim3 dimBlock(1, numBlocks);
    // computeMultiVertexNeighbors<<<dimBlock, blockSize_>>>(
    //     dVVDataArray_, dVVIndexArray_,
    //     clusterNumber_, dClusterIds, dSimplexOffsets,
    //     dConnectivity_, dOffsets_,
    //     dVertexIntervals_, dCellIntervals_,
    //     dExternalCells_, dExternalCellOffsets_,
    //     cellDimension_, threadsPerCluster_);

    //std::cout << "[GALE] Start compute vertex neighbors with " << numBlocks << " blocks and " << blockSize_ << " threads per block using SFCP_ColumnNum == " << SFCP_ColumnNum << std::endl;
    computeVertexNeighbors<<<numBlocks, blockSize_>>>(
        dVVDataArray_, dVVIndexArray_,
        1, endCluster,
        dConnectivity_, dOffsets_,
        dVertexIntervals_, dCellIntervals_,
        dExternalCells_, dExternalCellOffsets_,
        cellDimension_, threadsPerCluster_,
        SFCP_ColumnNum);

    // Check for kernel launch errors immediately
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(launchErr) << std::endl;
        return 1;
    }
    //std::cout << "[GALE] End compute vertex neighbors" << std::endl;

    /*
    int *hVVIndexArray = new int[vertexNumber_];
    int *hVVArray = new int[vertexNumber_ * SFCP_ColumnNum];
    CHECK(cudaMemcpy(hVVIndexArray, dVVIndexArray_, sizeof(int) * vertexNumber_, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hVVArray, dVVDataArray_, sizeof(int) * (vertexNumber_ * SFCP_ColumnNum), cudaMemcpyDeviceToHost));

    // Testing if the VV relation is computed correctly
    const int MAX_PRINT = 100;
    std::cout << "Testing the first 4 vertices in VV relation (up to " << MAX_PRINT << " values):" << std::endl;
    int printed = 0;
    for (int i = 0; i < 4 && printed < MAX_PRINT; i++) {
        std::cout << "Vertex " << i << " should have " << hVVIndexArray[i] << " items" << std::endl;
        for (int j = 0; j < SFCP_ColumnNum && printed < MAX_PRINT; j++) {
            std::cout << "\t" << i << ": " << hVVArray[(i*SFCP_ColumnNum)+j] << std::endl;
            printed++;
        }
    }
    delete[] hVVIndexArray, hVVArray;
    */

    // relationCompVecs_[0].hSizeArray = new int[vertexNumber_];
    // relationCompVecs_[0].hDataArray = new int[vertexNumber_ * vvColumnNum];

    // CHECK(cudaMallocAsync((void **)&relationCompVecs_[0].dSizeArray, sizeof(int) * vertexNumber_, 0));
    // CHECK(cudaMallocAsync((void **)&relationCompVecs_[0].dDataArray, sizeof(int) * vertexNumber_ * vvColumnNum, 0));

    // computeRelation(RelationType::VVRelation, 1, true, 0, false);

    // for (int vid = 1; vid < 20; ++vid) {
    //     std::cout << vid << ": " << relationCompVecs_[0].hSizeArray[vid] << std::endl;
    // }
    CHECK(cudaDeviceSynchronize());
    return 0;
}

/**
 * Compute the edge list from the input cell array.
*/
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

        CHECK(cudaMallocAsync(&dEdgeList_, sizeof(SimplexId) * edgeNumber_ * 2, 0));
        CHECK(cudaMemcpyAsync(dEdgeList_, edgeList_.data(), sizeof(SimplexId) * edgeNumber_ * 2, cudaMemcpyHostToDevice));

        std::cout << "[Gale] Edges preconditioned in " << t.getElapsedTime() << "s." << std::endl;
    }

    return 0;
}

/**
 * Compute the triangle list from the input cell array.
*/
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
            // std::cout << nid << ": " << hTriangleIntervals_[nid] << std::endl;
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

        CHECK(cudaMallocAsync(&dTriangleList_, sizeof(SimplexId) * triangleNumber_ * 3, 0));
        CHECK(cudaMemcpyAsync(dTriangleList_, triangleList_.data(), sizeof(SimplexId) * triangleNumber_ * 3, cudaMemcpyHostToDevice));

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

int Gale::getClusterVertexStarsVector(const SimplexId &clusterId, 
                std::vector<std::vector<SimplexId>> &vertexStarVec) const {

#ifndef TTK_ENABLE_KAMIKAZE
  if(clusterId <= 0 || clusterId > nodeNumber_)
    return -1;
#endif

  // get the vertex stars into a vector of vector
  SimplexId verticesPerCell = cellDimension_ + 1;
  SimplexId localVertexNum
    = hVertexIntervals_[clusterId] - hVertexIntervals_[clusterId - 1];
  vertexStarVec = std::vector<std::vector<SimplexId>>(localVertexNum);

  // internal cells
  for(SimplexId cid = hCellIntervals_[clusterId - 1] + 1;
      cid <= hCellIntervals_[clusterId]; cid++) {
    SimplexId localVertexId = hConnectivity_[hOffsets_[cid]]
                              - hVertexIntervals_[clusterId - 1] - 1;
    
    vertexStarVec[localVertexId].push_back(cid);
    for(SimplexId j = 1; j < verticesPerCell; j++) {
      SimplexId vertexId = hConnectivity_[hOffsets_[cid] + j];
      // see if it is in the current node
      if(vertexId > hVertexIntervals_[clusterId - 1]
          && vertexId <= hVertexIntervals_[clusterId]) {
        localVertexId = vertexId - hVertexIntervals_[clusterId - 1] - 1;
        vertexStarVec[localVertexId].push_back(cid);
      }
    }
  }
  for(int i = hExternalCellOffsets_[clusterId - 1]; i < hExternalCellOffsets_[clusterId]; ++i) {
    SimplexId cid = hExternalCells_[i];
    for(SimplexId j = 0; j < verticesPerCell; j++) {
      // see if it is in the current node
      SimplexId vertexId = hConnectivity_[hOffsets_[cid] + j];
      if(vertexId > hVertexIntervals_[clusterId - 1]
          && vertexId <= hVertexIntervals_[clusterId]) {
        SimplexId localVertexId
          = vertexId - hVertexIntervals_[clusterId - 1] - 1;
        vertexStarVec[localVertexId].push_back(cid);
      }
    }
  }

  for(size_t i = 0; i < vertexStarVec.size(); i++) {
    std::sort(vertexStarVec[i].begin(), vertexStarVec[i].end());
  }

  return 0;
}


int Gale::getClusterCellNeighbors(const SimplexId &clusterId) const {

#ifndef TTK_ENABLE_KAMIKAZE
  if(clusterId <= 0 || clusterId > nodeNumber_)
    return -1;
#endif

  std::vector<std::vector<SimplexId>> localCellNeighbors(
    hCellIntervals_[clusterId] - hCellIntervals_[clusterId - 1]);
  MeshSegment &segment = allClusters_.at(clusterId);

  // get the vertex stars into vector of vector
  SimplexId verticesPerCell = cellDimension_ + 1;
  std::vector<std::vector<SimplexId>> localVertexStars;
  getClusterVertexStarsVector(clusterId, localVertexStars);

  std::unordered_map<SimplexId, std::vector<std::vector<SimplexId>>>
    nodeMaps;
  for(SimplexId cid = hCellIntervals_[clusterId - 1] + 1;
      cid <= hCellIntervals_[clusterId]; cid++) {
    for(SimplexId j = 1; j < verticesPerCell; j++) {
      if(hConnectivity_[hOffsets_[cid] + j]> hVertexIntervals_[clusterId]) {
        SimplexId nodeId
          = hIndexArray_[hConnectivity_[hOffsets_[cid] + j]];
        if(nodeMaps.find(nodeId) == nodeMaps.end()) {
          std::vector<std::vector<SimplexId>> tmpVec;
          getClusterVertexStarsVector(nodeId, tmpVec);
          nodeMaps[nodeId] = std::move(tmpVec);
        }
      }
    }
  }


  if(cellDimension_ == 2) {
    for(SimplexId cid = hCellIntervals_[clusterId - 1] + 1;
        cid <= hCellIntervals_[clusterId]; cid++) {
      for(SimplexId j = 0; j < 3; j++) {

        SimplexId v0 = hConnectivity_[hOffsets_[cid + j]];
        SimplexId v1 = hConnectivity_[hOffsets_[cid + (j + 1) % 3]];
        SimplexId localV0 = v0 - hVertexIntervals_[clusterId - 1] - 1;
        SimplexId localV1 = v1 - hVertexIntervals_[clusterId - 1] - 1;

        std::vector<SimplexId> stars0, stars1;
        if(v0 <= hVertexIntervals_[clusterId]) {
          stars0 = localVertexStars[localV0];
        } else {
          localV0 = v0 - hVertexIntervals_[hIndexArray_[v0] - 1] - 1;
          stars0 = nodeMaps[hIndexArray_[v0]][localV0];
        }
        if(v1 <= hVertexIntervals_[clusterId]) {
          stars1 = localVertexStars[localV1];
        } else {
          localV1 = v1 - hVertexIntervals_[hIndexArray_[v1] - 1] - 1;
          stars1 = nodeMaps[hIndexArray_[v1]][localV1];
        }

        // perform an intersection of the 2 sorted star lists
        SimplexId pos0 = 0, pos1 = 0;
        SimplexId intersection = -1;

        while((pos0 < (SimplexId)stars0.size())
              && (pos1 < (SimplexId)stars1.size())) {
          SimplexId biggest = stars0[pos0];
          if(stars1[pos1] > biggest) {
            biggest = stars1[pos1];
          }

          for(SimplexId l = pos0; l < (SimplexId)stars0.size(); l++) {
            if(stars0[l] < biggest) {
              pos0++;
            } else {
              break;
            }
          }
          for(SimplexId l = pos1; l < (SimplexId)stars1.size(); l++) {
            if(stars1[l] < biggest) {
              pos1++;
            } else {
              break;
            }
          }

          if(pos0 >= (SimplexId)stars0.size()
              || pos1 >= (SimplexId)stars1.size())
            break;

          if(stars0[pos0] == stars1[pos1]) {
            if(stars0[pos0] != cid) {
              intersection = stars0[pos0];
              break;
            }
            pos0++;
            pos1++;
          }
        }

        if(intersection != -1) {
          localCellNeighbors[cid - hCellIntervals_[clusterId - 1] - 1]
            .push_back(intersection);
        }
      }
    }
  }

  else if(cellDimension_ == 3) {
    for(SimplexId cid = hCellIntervals_[clusterId - 1] + 1;
        cid <= hCellIntervals_[clusterId]; cid++) {
      // go triangle by triangle
      for(SimplexId j = 0; j < 4; j++) {

        SimplexId v0 = hConnectivity_[hOffsets_[cid + j % 4]];
        SimplexId v1 = hConnectivity_[hOffsets_[cid + (j + 1) % 4]];
        SimplexId v2 = hConnectivity_[hOffsets_[cid + (j + 2) % 4]];

        SimplexId localV0 = v0 - hVertexIntervals_[clusterId - 1] - 1;
        SimplexId localV1 = v1 - hVertexIntervals_[clusterId - 1] - 1;
        SimplexId localV2 = v2 - hVertexIntervals_[clusterId - 1] - 1;

        std::vector<SimplexId> stars0, stars1, stars2;
        if(v0 <= hVertexIntervals_[clusterId]) {
          stars0 = localVertexStars[localV0];
        } else {
          localV0 = v0 - hVertexIntervals_[hIndexArray_[v0] - 1] - 1;
          stars0 = nodeMaps[hIndexArray_[v0]][localV0];
        }
        if(v1 <= hVertexIntervals_[clusterId]) {
          stars1 = localVertexStars[localV1];
        } else {
          localV1 = v1 - hVertexIntervals_[hIndexArray_[v1] - 1] - 1;
          stars1 = nodeMaps[hIndexArray_[v1]][localV1];
        }
        if(v2 <= hVertexIntervals_[clusterId]) {
          stars2 = localVertexStars[localV2];
        } else {
          localV2 = v2 - hVertexIntervals_[hIndexArray_[v2] - 1] - 1;
          stars2 = nodeMaps[hIndexArray_[v2]][localV2];
        }

        // perform an intersection of the 3 (sorted) star lists
        SimplexId pos0 = 0, pos1 = 0, pos2 = 0;
        SimplexId intersection = -1;

        while((pos0 < (SimplexId)stars0.size())
              && (pos1 < (SimplexId)stars1.size())
              && (pos2 < (SimplexId)stars2.size())) {

          SimplexId biggest = stars0[pos0];
          if(stars1[pos1] > biggest) {
            biggest = stars1[pos1];
          }
          if(stars2[pos2] > biggest) {
            biggest = stars2[pos2];
          }

          for(SimplexId l = pos0; l < (SimplexId)stars0.size(); l++) {
            if(stars0[l] < biggest) {
              pos0++;
            } else {
              break;
            }
          }
          for(SimplexId l = pos1; l < (SimplexId)stars1.size(); l++) {
            if(stars1[l] < biggest) {
              pos1++;
            } else {
              break;
            }
          }
          for(SimplexId l = pos2; l < (SimplexId)stars2.size(); l++) {
            if(stars2[l] < biggest) {
              pos2++;
            } else {
              break;
            }
          }

          if(pos0 >= (SimplexId)stars0.size()
              || pos1 >= (SimplexId)stars1.size()
              || pos2 >= (SimplexId)stars2.size())
            break;

          if((stars0[pos0] == stars1[pos1])
              && (stars0[pos0] == stars2[pos2])) {
            if(stars0[pos0] != cid) {
              intersection = stars0[pos0];
              break;
            }
            pos0++;
            pos1++;
            pos2++;
          }
        }

        if(intersection != -1) {
          localCellNeighbors[cid - hCellIntervals_[clusterId - 1] - 1]
            .push_back(intersection);
        }
      }
    }
  }

  segment.cellNeighbors_ = FlatJaggedArray(localCellNeighbors);

  return 0;
}

