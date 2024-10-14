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
Gale::Gale(int tnum, int tnumPerCluster) {
    this->setDebugMsgPrefix("Gale");
    threadNumber_ = tnum;
    threadsPerCluster_ = tnumPerCluster;
    currClusters_ = std::vector<SimplexId>(threadNumber_, 0);
    currRelations_ = std::vector<RelationType>(threadNumber_, RelationType::BlankRelation);
    // waitings_ = std::vector<int>(threadNumber_, 0);
    relationSems_ = std::vector<sem_t>(threadNumber_);
    for (int i = 0; i < 25; ++i) {
        baseClusters_[i] = std::vector<int>(threadNumber_, 0);
    }
    #ifdef MEASURE_WAITING
    timers_ = std::vector<Timer>(threadNumber_);
    // waitingTimes_ = std::vector<double>(threadNumber_, 0);
    // waitingCounts_ = std::vector<int>(threadNumber_, 0);
    waitingTimes_.resize(threadNumber_);
    waitingCounts_.resize(threadNumber_);
    consumerTimes_.resize(threadNumber_);
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
    if (producer_.joinable()) {
        producer_.join();
    }

    #ifdef MEASURE_WAITING
    // for (int i = 0; i < threadNumber_; ++i) {
    //     std::cout << "thread " << i << ": ";
    //     for (auto &e : waitingCounts_[i]) {
    //         std::cout << relationToString(e.first) << " missing counts: " << e.second << ", waiting time: " << waitingTimes_[i].at(e.first) << "; ";
    //         // std::cout << static_cast<int>(e.first) << " missing counts: " << e.second << ", waiting time: " << waitingTimes_[i].at(e.first) << "; ";
    //     }
    //     std::cout << std::endl;
    //     // std::cout << "waiting count " << i << ": " << waitingCounts_[i] << ", waiting time: " << waitingTimes_[i] << std::endl;
    // }
    
    // time for consumer thread to make the request 
    // for (int i = 0; i < threadNumber_; ++i) {
    //     for (int j = 0; j < consumerTimes_[i].size(); ++j) {
    //         std::cout << "consumer " << i << ": " << consumerTimes_[i][j] << std::endl;
    //     }
    // }

    for (int j = 0; j < qSizes_.size(); ++j) {
        std::cout << "queue length: " << qSizes_[j] << ", total time: " << qTimes_[j] << ", kernel time: " << kernelTimes_[j] << ", return time: " << returnTimes_[j] << std::endl;
    }

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
            int numBlocks = clusterNumber_ / (blockSize_ / threadsPerCluster_) + 1;
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
            int *dEdgeTriangleMatrix, *dEdgeTriangleSizes;
            int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
            int localEdgeSum = hEdgeIntervals_[endCluster] - edgeIdOffset;
            CHECK(cudaMalloc((void **)&dEdgeTriangleSizes, sizeof(int) * localEdgeSum));
            CHECK(cudaMalloc((void **)&dEdgeTriangleMatrix, sizeof(int) * localEdgeSum * efColumnNum));
            CHECK(cudaMemset(dEdgeTriangleSizes, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(dEdgeTriangleMatrix, -1, sizeof(int) * localEdgeSum * efColumnNum));

            computeEdgeTriangles<<<numBlocks, blockSize_>>>(
                dEdgeList_, dTriangleList_,
                dEdgeTriangleMatrix, dEdgeTriangleSizes,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            int *hEdgeTriangleSizes = new int[localEdgeSum];
            int *hEdgeTriangleMatrix = new int[localEdgeSum * efColumnNum];

            CHECK(cudaMemcpy(hEdgeTriangleSizes, dEdgeTriangleSizes, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hEdgeTriangleMatrix, dEdgeTriangleMatrix, sizeof(int) * localEdgeSum * efColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(EFMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].edgeTriangles_.empty()) {
                    int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                    std::vector<SimplexId> edgeTriangleData{};
                    std::vector<SimplexId> edgeTriangleOffsets(localEdgeNum + 1);
                    edgeTriangleOffsets[0] = 0;
                    int reid = hEdgeIntervals_[sid - 1] - edgeIdOffset;
                    for (int eid = 0; eid < localEdgeNum; ++eid) {
                        edgeTriangleData.insert(edgeTriangleData.end(), hEdgeTriangleMatrix + reid * efColumnNum, hEdgeTriangleMatrix + reid * efColumnNum + hEdgeTriangleSizes[reid]);
                        edgeTriangleOffsets[eid+1] = edgeTriangleOffsets[eid] + hEdgeTriangleSizes[reid];
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

            CHECK(cudaFree(dEdgeTriangleSizes));
            CHECK(cudaFree(dEdgeTriangleMatrix));

            delete[] hEdgeTriangleMatrix;
            delete[] hEdgeTriangleSizes;
            break;
        }

        case RelationType::ETRelation:
        {
            int *dEdgeStarMatrix, *dEdgeStarSizes;
            int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
            int localEdgeSum = hEdgeIntervals_[endCluster] - edgeIdOffset;
            CHECK(cudaMalloc((void **)&dEdgeStarSizes, sizeof(int) * localEdgeSum));
            CHECK(cudaMalloc((void **)&dEdgeStarMatrix, sizeof(int) * localEdgeSum * etColumnNum));
            CHECK(cudaMemset(dEdgeStarSizes, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(dEdgeStarMatrix, -1, sizeof(int) * localEdgeSum * etColumnNum));

            computeEdgeStars<<<numBlocks, blockSize_>>>(
                dEdgeList_, dEdgeStarMatrix, dEdgeStarSizes,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            int *hEdgeStarSizes_ = new int[localEdgeSum];
            int *hEdgeStarMatrix_ = new int[localEdgeSum * etColumnNum];

            CHECK(cudaMemcpy(hEdgeStarSizes_, dEdgeStarSizes, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hEdgeStarMatrix_, dEdgeStarMatrix, sizeof(int) * localEdgeSum * etColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(ETMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].edgeStars_.empty()) {
                    int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                    std::vector<SimplexId> edgeStarData{};
                    std::vector<SimplexId> edgeStarOffsets(localEdgeNum + 1);
                    edgeStarOffsets[0] = 0;
                    int reid = hEdgeIntervals_[sid - 1] - edgeIdOffset;
                    for (int eid = 0; eid < localEdgeNum; ++eid) {
                        edgeStarData.insert(edgeStarData.end(), hEdgeStarMatrix_ + reid * etColumnNum, hEdgeStarMatrix_ + reid * etColumnNum + hEdgeStarSizes_[reid]);
                        edgeStarOffsets[eid+1] = edgeStarOffsets[eid] + hEdgeStarSizes_[reid];
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

            CHECK(cudaFree(dEdgeStarSizes));
            CHECK(cudaFree(dEdgeStarMatrix));

            delete[] hEdgeStarMatrix_;
            delete[] hEdgeStarSizes_;
            break;
        }

        case RelationType::FTRelation:
        {
            int *dTriangleStarMatrix, *dTriangleStarSizes;
            int triangleIdOffset = hTriangleIntervals_[clusterId - 1];
            int localTriangleSum = hTriangleIntervals_[endCluster] - triangleIdOffset;
            CHECK(cudaMalloc((void **)&dTriangleStarSizes, sizeof(int) * localTriangleSum));
            CHECK(cudaMalloc((void **)&dTriangleStarMatrix, sizeof(int) * localTriangleSum * ftColumnNum));
            CHECK(cudaMemset(dTriangleStarSizes, 0, sizeof(int) * localTriangleSum));
            CHECK(cudaMemset(dTriangleStarMatrix, -1, sizeof(int) * localTriangleSum * ftColumnNum));

            computeTriangleStars<<<numBlocks, blockSize_>>>(
                dTriangleList_, dTriangleStarMatrix, dTriangleStarSizes,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            int *hTriangleStarSizes = new int[localTriangleSum];
            int *hTriangleStarMatrix = new int[localTriangleSum * ftColumnNum];

            CHECK(cudaMemcpy(hTriangleStarSizes, dTriangleStarSizes, sizeof(int) * localTriangleSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hTriangleStarMatrix, dTriangleStarMatrix, sizeof(int) * localTriangleSum * ftColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(FTMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].triangleStars_.empty()) {
                    int localTriangleNum = hTriangleIntervals_[sid] - hTriangleIntervals_[sid - 1];
                    std::vector<SimplexId> triangleStarData{};
                    std::vector<SimplexId> triangleStarOffsets(localTriangleNum + 1);
                    triangleStarOffsets[0] = 0;
                    int rfid = hTriangleIntervals_[sid - 1] - triangleIdOffset;
                    for (int fid = 0; fid < localTriangleNum; ++fid) {
                        triangleStarData.insert(triangleStarData.end(), hTriangleStarMatrix + rfid * ftColumnNum, hTriangleStarMatrix + rfid * ftColumnNum + hTriangleStarSizes[rfid]);
                        triangleStarOffsets[fid+1] = triangleStarOffsets[fid] + hTriangleStarSizes[rfid];
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

            CHECK(cudaFree(dTriangleStarSizes));
            CHECK(cudaFree(dTriangleStarMatrix));
            delete[] hTriangleStarMatrix;
            delete[] hTriangleStarSizes;

            break;
        }

        case RelationType::VERelation:
        {
            int *dVertexEdgeMatrix, *dVertexEdgeSizes;
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexSum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexEdgeSizes, sizeof(int) * localVertexSum));
            CHECK(cudaMalloc((void **)&dVertexEdgeMatrix, sizeof(int) * localVertexSum * veColumnNum));
            CHECK(cudaMemset(dVertexEdgeSizes, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(dVertexEdgeMatrix, -1, sizeof(int) * localVertexSum * veColumnNum));

            computeVertexEdges<<<numBlocks, blockSize_>>>(
                dEdgeList_, dVertexEdgeMatrix, dVertexEdgeSizes,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            int *hVertexEdgeSizes = new int[localVertexSum];
            int *hVertexEdgeMatrix = new int[localVertexSum * veColumnNum];

            CHECK(cudaMemcpy(hVertexEdgeSizes, dVertexEdgeSizes, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexEdgeMatrix, dVertexEdgeMatrix, sizeof(int) * localVertexSum * veColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VEMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexEdges_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexEdgeData{};
                    std::vector<SimplexId> vertexEdgeOffsets(localVertexNum + 1);
                    vertexEdgeOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexEdgeData.insert(vertexEdgeData.end(), hVertexEdgeMatrix + rvid * veColumnNum, hVertexEdgeMatrix + rvid * veColumnNum + hVertexEdgeSizes[rvid]);
                        vertexEdgeOffsets[vid+1] = vertexEdgeOffsets[vid] + hVertexEdgeSizes[rvid];
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

            CHECK(cudaFree(dVertexEdgeSizes));
            CHECK(cudaFree(dVertexEdgeMatrix));
            delete[] hVertexEdgeMatrix;
            delete[] hVertexEdgeSizes;

            break;
        }

        case RelationType::VFRelation:
        {
            int *dVertexTriangleMatrix, *dVertexTriangleSizes;
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexSum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexTriangleSizes, sizeof(int) * localVertexSum));
            CHECK(cudaMalloc((void **)&dVertexTriangleMatrix, sizeof(int) * localVertexSum * vfColumnNum));
            CHECK(cudaMemset(dVertexTriangleSizes, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(dVertexTriangleMatrix, -1, sizeof(int) * localVertexSum * vfColumnNum));

            computeVertexTriangles<<<numBlocks, blockSize_>>>(
                dTriangleList_, dVertexTriangleMatrix, dVertexTriangleSizes,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            int *hVertexTriangleSizes = new int[localVertexSum];
            int *hVertexTriangleMatrix = new int[localVertexSum * vfColumnNum];

            CHECK(cudaMemcpy(hVertexTriangleSizes, dVertexTriangleSizes, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexTriangleMatrix, dVertexTriangleMatrix, sizeof(int) * localVertexSum * vfColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VFMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexTriangles_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexTriangleData{};
                    std::vector<SimplexId> vertexTriangleOffsets(localVertexNum + 1);
                    vertexTriangleOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexTriangleData.insert(vertexTriangleData.end(), hVertexTriangleMatrix + rvid * vfColumnNum, hVertexTriangleMatrix + rvid * vfColumnNum + hVertexTriangleSizes[rvid]);
                        vertexTriangleOffsets[vid+1] = vertexTriangleOffsets[vid] + hVertexTriangleSizes[rvid];
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

            CHECK(cudaFree(dVertexTriangleSizes));
            CHECK(cudaFree(dVertexTriangleMatrix));
            delete[] hVertexTriangleMatrix;
            delete[] hVertexTriangleSizes;

            break;
        }

        case RelationType::VTRelation:
        {
            int *dVertexStarMatrix, *dVertexStarSizes;
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexStarSizes, sizeof(int) * localVertexNum));
            CHECK(cudaMalloc((void **)&dVertexStarMatrix, sizeof(int) * localVertexNum * vtColumnNum));
            CHECK(cudaMemset(dVertexStarSizes, 0, sizeof(int) * localVertexNum));
            CHECK(cudaMemset(dVertexStarMatrix, -1, sizeof(int) * localVertexNum * vtColumnNum));

            computeVertexStars<<<numBlocks, blockSize_>>>(
                dVertexStarMatrix, dVertexStarSizes,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            int *hVertexStarSizes = new int[localVertexNum];
            int *hVertexStarMatrix = new int[localVertexNum * vtColumnNum];

            CHECK(cudaMemcpy(hVertexStarSizes, dVertexStarSizes, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexStarMatrix, dVertexStarMatrix, sizeof(int) * localVertexNum * vtColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VTMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexStars_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexStarData{};
                    std::vector<SimplexId> vertexStarOffsets(localVertexNum + 1);
                    vertexStarOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexStarData.insert(vertexStarData.end(), hVertexStarMatrix + rvid * vtColumnNum, hVertexStarMatrix + rvid * vtColumnNum + hVertexStarSizes[rvid]);
                        vertexStarOffsets[vid+1] = vertexStarOffsets[vid] + hVertexStarSizes[rvid];
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

            CHECK(cudaFree(dVertexStarSizes));
            CHECK(cudaFree(dVertexStarMatrix));
            delete[] hVertexStarMatrix;
            delete[] hVertexStarSizes;

            break;
        }

        case RelationType::VVRelation:
        {
            int *dVertexNeighborMatrix, *dVertexNeighborSizes;
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexNeighborSizes, sizeof(int) * localVertexNum));
            CHECK(cudaMalloc((void **)&dVertexNeighborMatrix, sizeof(int) * localVertexNum * vvColumnNum));
            CHECK(cudaMemset(dVertexNeighborSizes, 0, sizeof(int) * localVertexNum));
            CHECK(cudaMemset(dVertexNeighborMatrix, -1, sizeof(int) * localVertexNum * vvColumnNum));
            
            computeVertexNeighbors<<<numBlocks, blockSize_>>>(
                dVertexNeighborMatrix, dVertexNeighborSizes,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            int *hVertexNeighborSizes_ = new int[localVertexNum];
            int *hVertexNeighborMatrix_ = new int[localVertexNum * vvColumnNum];

            CHECK(cudaMemcpy(hVertexNeighborSizes_, dVertexNeighborSizes, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexNeighborMatrix_, dVertexNeighborMatrix, sizeof(int) * localVertexNum * vvColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VVMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexNeighbors_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexNeighborData{};
                    std::vector<SimplexId> vertexNeighborOffsets(localVertexNum + 1);
                    vertexNeighborOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexNeighborData.insert(vertexNeighborData.end(), hVertexNeighborMatrix_ + rvid * vvColumnNum, hVertexNeighborMatrix_ + rvid * vvColumnNum + hVertexNeighborSizes_[rvid]);
                        vertexNeighborOffsets[vid+1] = vertexNeighborOffsets[vid] + hVertexNeighborSizes_[rvid];
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

            CHECK(cudaFree(dVertexNeighborSizes));
            CHECK(cudaFree(dVertexNeighborMatrix));

            delete[] hVertexNeighborMatrix_;
            delete[] hVertexNeighborSizes_;
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
    std::vector<std::vector<SimplexId>> hEndClusters(numRelations);
    std::vector<std::vector<int>> hSimplexOffsets(numRelations);
    cudaEvent_t startEvents[numRelations];
    cudaEvent_t endEvents[numRelations];
    float kernelTime = 0, returnTime = 0;

    // launch GPU kernels for different topological relations
    for (int curStream = 0; curStream < numRelations; ++curStream) {
        // (relation, [(tid, clusterId), ...])
        cudaEventCreate(&startEvents[curStream]);
        cudaEventCreate(&endEvents[curStream]);
        RelationType &relation = relationOrders[curStream];
        std::vector<std::pair<int,int>> &jobs = requestMap[relation];
        int threadNum = jobs.size();

        // launches kernels on different streams
        switch (relation) {
            case RelationType::BERelation:
            {
                int localEdgeSum = 0;
                int hClusterIds[threadNum];
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                // int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    // maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localEdgeNum = hEdgeIntervals_[hEndClusters[curStream][i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNum;
                }
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dDataArray, false, sizeof(bool) * localEdgeSum, threadStreams_[curStream]));

                dim3 dimBlock(threadNum, blocksPerThread_);
                computeMultiBoundaryEdges<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    dEdgeList_, dTriangleList_,
                    (bool *)relationCompVecs_[curStream].dDataArray, dBoundaryTriangles_,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets,
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                relationCompVecs_[curStream].hDataArray = (int *)malloc(sizeof(bool) * localEdgeSum);
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(bool) * localEdgeSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                break;
            }

            case RelationType::BVRelation: 
            {
                int localVertexSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                // int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    // maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dDataArray, false, sizeof(bool) * localVertexSum, threadStreams_[curStream]));

                dim3 dimBlock(threadNum, blocksPerThread_);
                computeMultiBoundaryVertices<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    dTriangleList_, dBoundaryTriangles_, (bool *)relationCompVecs_[curStream].dDataArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets, 
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                relationCompVecs_[curStream].hDataArray = (int *)malloc(sizeof(bool) * localVertexSum);
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(bool) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                break;
            }
            
            case RelationType::EFRelation:
            {
                int localEdgeSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                // int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    // maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localEdgeNums = hEdgeIntervals_[hEndClusters[curStream][i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNums;
                }
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dSizeArray, 0, sizeof(int) * localEdgeSum, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dDataArray, -1, sizeof(int) * localEdgeSum * efColumnNum, threadStreams_[curStream]));

                dim3 dimBlock(threadNum, blocksPerThread_);
                computeMultiEdgeTriangles<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    dEdgeList_, dTriangleList_,
                    relationCompVecs_[curStream].dDataArray, relationCompVecs_[curStream].dSizeArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets,
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                relationCompVecs_[curStream].hSizeArray = new int[localEdgeSum];
                relationCompVecs_[curStream].hDataArray = new int[localEdgeSum * efColumnNum];

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hSizeArray, relationCompVecs_[curStream].dSizeArray, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(int) * localEdgeSum * efColumnNum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                break;
            }

            case RelationType::ETRelation:
            {
                int localEdgeSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                // int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    // maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localEdgeNums = hEdgeIntervals_[hEndClusters[curStream][i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNums;
                }

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemset(relationCompVecs_[curStream].dSizeArray, 0, sizeof(int) * localEdgeSum));
                CHECK(cudaMemset(relationCompVecs_[curStream].dDataArray, -1, sizeof(int) * localEdgeSum * etColumnNum));

                dim3 dimBlock(threadNum, blocksPerThread_);
                computeMultiEdgeStars<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    dEdgeList_, relationCompVecs_[curStream].dDataArray, relationCompVecs_[curStream].dSizeArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets, 
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                relationCompVecs_[curStream].hSizeArray = new int[localEdgeSum];
                relationCompVecs_[curStream].hDataArray = new int[localEdgeSum * etColumnNum];

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hSizeArray, relationCompVecs_[curStream].dSizeArray, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(int) * localEdgeSum * etColumnNum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                break;
            }

            case RelationType::FTRelation:
            {
                int localTriangleSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                // int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localTriangleSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    // maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localTriangleNums = hTriangleIntervals_[hEndClusters[curStream][i]] - hTriangleIntervals_[hClusterIds[i] - 1];
                    localTriangleSum += localTriangleNums;
                }

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemset(relationCompVecs_[curStream].dSizeArray, 0, sizeof(int) * localTriangleSum));
                CHECK(cudaMemset(relationCompVecs_[curStream].dDataArray, -1, sizeof(int) * localTriangleSum * ftColumnNum));

                dim3 dimBlock(threadNum, blocksPerThread_);
                computeMultiTriangleStars<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    dTriangleList_, relationCompVecs_[curStream].dDataArray, relationCompVecs_[curStream].dSizeArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets,
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                relationCompVecs_[curStream].hSizeArray = new int[localTriangleSum];
                relationCompVecs_[curStream].hDataArray = new int[localTriangleSum * ftColumnNum];

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hSizeArray, relationCompVecs_[curStream].dSizeArray, sizeof(int) * localTriangleSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(int) * localTriangleSum * ftColumnNum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                break;
            }

            case RelationType::VERelation:
            {
                int localVertexSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                // int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    // maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dSizeArray, 0, sizeof(int) * localVertexSum, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dDataArray, -1, sizeof(int) * localVertexSum * veColumnNum, threadStreams_[curStream]));

                dim3 dimBlock(threadNum, blocksPerThread_);
                #ifdef MEASURE_WAITING
                cudaEventRecord(startEvents[curStream], threadStreams_[curStream]);
                #endif
                computeMultiVertexEdges<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    dEdgeList_, relationCompVecs_[curStream].dDataArray, relationCompVecs_[curStream].dSizeArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets,
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                #ifdef MEASURE_WAITING
                cudaEventRecord(endEvents[curStream], threadStreams_[curStream]);
                #endif

                relationCompVecs_[curStream].hSizeArray = new int[localVertexSum];
                relationCompVecs_[curStream].hDataArray = new int[localVertexSum * veColumnNum];

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hSizeArray, relationCompVecs_[curStream].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(int) * localVertexSum * veColumnNum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                break;
            }

            case RelationType::VFRelation:
            {
                int localVertexSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                // int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localVertexSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    // maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dSizeArray, 0, sizeof(int) * localVertexSum, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dDataArray, -1, sizeof(int) * localVertexSum * vfColumnNum, threadStreams_[curStream]));

                dim3 dimBlock(threadNum, blocksPerThread_);
                #ifdef MEASURE_WAITING
                cudaEventRecord(startEvents[curStream], threadStreams_[curStream]);
                #endif
                computeMultiVertexTriangles<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    dTriangleList_, relationCompVecs_[curStream].dDataArray, relationCompVecs_[curStream].dSizeArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets,
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                #ifdef MEASURE_WAITING
                cudaEventRecord(startEvents[curStream], threadStreams_[curStream]);
                #endif

                relationCompVecs_[curStream].hSizeArray = new int [localVertexSum];
                relationCompVecs_[curStream].hDataArray = new int [localVertexSum * vfColumnNum];

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hSizeArray, relationCompVecs_[curStream].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(int) * localVertexSum * vfColumnNum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
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
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dSizeArray, 0, sizeof(int) * localVertexSum, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dDataArray, -1, sizeof(int) * localVertexSum * vtColumnNum, threadStreams_[curStream]));

                dim3 dimBlock(threadNum, blocksPerThread_);
                #ifdef MEASURE_WAITING
                cudaEventRecord(startEvents[curStream], threadStreams_[curStream]);
                #endif
                computeMultiVertexStars<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    relationCompVecs_[curStream].dDataArray, relationCompVecs_[curStream].dSizeArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets,
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);
                #ifdef MEASURE_WAITING
                cudaEventRecord(endEvents[curStream], threadStreams_[curStream]);
                #endif

                relationCompVecs_[curStream].hSizeArray = new int[localVertexSum];
                relationCompVecs_[curStream].hDataArray = new int[localVertexSum * vtColumnNum];

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hSizeArray, relationCompVecs_[curStream].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(int) * localVertexSum * vtColumnNum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
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
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + precompClusterNumber_, clusterNumber_) - 1;
                    int localVertexNum = hVertexIntervals_[hEndClusters[curStream][i]] - hVertexIntervals_[hClusterIds[i] - 1];
                    localVertexSum += localVertexNum;
                }

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dClusterIds, hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].dSimplexOffsets, hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dSizeArray, 0, sizeof(int) * localVertexSum, threadStreams_[curStream]));
                CHECK(cudaMemsetAsync(relationCompVecs_[curStream].dDataArray, -1, sizeof(int) * localVertexSum * vvColumnNum, threadStreams_[curStream]));

                dim3 dimBlock(threadNum, blocksPerThread_);
                #ifdef MEASURE_WAITING
                cudaEventRecord(startEvents[curStream], threadStreams_[curStream]);
                #endif
                computeMultiVertexNeighbors<<<dimBlock, blockSize_, 0, threadStreams_[curStream]>>>(
                    relationCompVecs_[curStream].dDataArray, relationCompVecs_[curStream].dSizeArray,
                    clusterNumber_, relationCompVecs_[curStream].dClusterIds, relationCompVecs_[curStream].dSimplexOffsets,
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);
                #ifdef MEASURE_WAITING
                cudaEventRecord(endEvents[curStream], threadStreams_[curStream]);
                #endif

                relationCompVecs_[curStream].hSizeArray = new int[localVertexSum];
                relationCompVecs_[curStream].hDataArray = new int[localVertexSum * vvColumnNum];

                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hSizeArray, relationCompVecs_[curStream].dSizeArray, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
                CHECK(cudaMemcpyAsync(relationCompVecs_[curStream].hDataArray, relationCompVecs_[curStream].dDataArray, sizeof(int) * localVertexSum * vvColumnNum, cudaMemcpyDeviceToHost, threadStreams_[curStream]));
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

        cudaError_t status = cudaStreamSynchronize(threadStreams_[curStream]);
        if (status != cudaSuccess) {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                __FILE__, __LINE__); if (abort) exit(status);
        }

        #ifdef MEASURE_WAITING
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvents[curStream], endEvents[curStream]);
        kernelTime += milliseconds;
        Timer tmp;
        #endif

        // launches different kernels on different streams
        switch (relation) {
            case RelationType::BERelation: 
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (int sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].boundaryEdges_.empty()) {
                            int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                            int relativeEid = hEdgeIntervals_[sid - 1] - hEdgeIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            allClusters_[sid].boundaryEdges_.resize(localEdgeNum);
                            for (int eid = 0; eid < localEdgeNum; ++eid) {
                                allClusters_[sid].boundaryEdges_[eid] = relationCompVecs_[curStream].hDataArray[relativeEid++];
                            }
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                free(relationCompVecs_[curStream].hDataArray);
                break;
            }
            case RelationType::BVRelation:
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClustersId = jobs[i].second;
                    for (int sid = baseClustersId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].boundaryVertices_.empty()) {
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid-1];
                            int relativeVid = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClustersId - 1] + hSimplexOffsets[curStream][i];
                            allClusters_[sid].boundaryVertices_.resize(localVertexNum);
                            for (int vid = 0; vid < localVertexNum; ++vid) {
                                allClusters_[sid].boundaryVertices_[vid] = relationCompVecs_[curStream].hDataArray[relativeVid++];
                            }
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClustersId;
                    sem_post(&relationSems_[tid]);
                }

                free(relationCompVecs_[curStream].hDataArray);
                break;
            }
            
            case RelationType::EFRelation:
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].edgeTriangles_.empty()) {
                            std::vector<SimplexId> edgeTriangleData{};
                            int reid = hEdgeIntervals_[sid - 1] - hEdgeIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                            std::vector<SimplexId> edgeTriangleOffsets(localEdgeNum + 1);
                            edgeTriangleOffsets[0] = 0;
                            for (int eid = 0; eid < localEdgeNum; ++eid) {
                                edgeTriangleData.insert(edgeTriangleData.end(), 
                                relationCompVecs_[curStream].hDataArray + reid * efColumnNum,
                                relationCompVecs_[curStream].hDataArray + reid * efColumnNum + relationCompVecs_[curStream].hSizeArray[reid]);
                                edgeTriangleOffsets[eid+1] = edgeTriangleOffsets[eid] + relationCompVecs_[curStream].hSizeArray[reid];
                                ++reid;
                            }
                            allClusters_[sid].edgeTriangles_.setData(std::move(edgeTriangleData), std::move(edgeTriangleOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0) 
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                delete[] relationCompVecs_[curStream].hDataArray;
                delete[] relationCompVecs_[curStream].hSizeArray;
                break;
            }
            
            case RelationType::ETRelation: 
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].edgeStars_.empty()) {
                            std::vector<SimplexId> edgeStarData{};
                            int reid = hEdgeIntervals_[sid - 1] - hEdgeIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                            std::vector<SimplexId> edgeStarOffsets(localEdgeNum + 1);
                            edgeStarOffsets[0] = 0;
                            for (int j = 0; j < localEdgeNum; ++j) {
                                edgeStarData.insert(edgeStarData.end(), 
                                relationCompVecs_[curStream].hDataArray + reid * etColumnNum,
                                relationCompVecs_[curStream].hDataArray + reid * etColumnNum + relationCompVecs_[curStream].hSizeArray[reid]);
                                edgeStarOffsets[j+1] = edgeStarOffsets[j] + relationCompVecs_[curStream].hSizeArray[reid];
                                ++reid;
                            }
                            allClusters_[sid].edgeStars_.setData(std::move(edgeStarData), std::move(edgeStarOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0) 
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                delete[] relationCompVecs_[curStream].hDataArray;
                delete[] relationCompVecs_[curStream].hSizeArray;
                break;
            }

            case RelationType::FTRelation:
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].triangleStars_.empty()) {
                            std::vector<SimplexId> triangleStarData{};
                            int reid = hTriangleIntervals_[sid - 1] - hTriangleIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localTriangleNum = hTriangleIntervals_[sid] - hTriangleIntervals_[sid - 1];
                            std::vector<SimplexId> triangleStarOffsets(localTriangleNum + 1);
                            triangleStarOffsets[0] = 0;
                            for (int j = 0; j < localTriangleNum; ++j) {
                                triangleStarData.insert(triangleStarData.end(), 
                                relationCompVecs_[curStream].hDataArray + reid * ftColumnNum,
                                relationCompVecs_[curStream].hDataArray + reid * ftColumnNum + relationCompVecs_[curStream].hSizeArray[reid]);
                                triangleStarOffsets[j+1] = triangleStarOffsets[j] + relationCompVecs_[curStream].hSizeArray[reid];
                                ++reid;
                            }
                            allClusters_[sid].triangleStars_.setData(std::move(triangleStarData), std::move(triangleStarOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                delete[] relationCompVecs_[curStream].hDataArray;
                delete[] relationCompVecs_[curStream].hSizeArray;
                break;
            }

            case RelationType::VERelation:
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].vertexEdges_.empty()) {
                            std::vector<SimplexId> vertexEdgeData{};
                            int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                            std::vector<SimplexId> vertexEdgeOffsets(localVertexNum + 1);
                            vertexEdgeOffsets[0] = 0;
                            for (int j = 0; j < localVertexNum; ++j) {
                                int rj = j + diff; // relative j value in the large matrix
                                vertexEdgeData.insert(vertexEdgeData.end(), 
                                relationCompVecs_[curStream].hDataArray + rj * veColumnNum,
                                relationCompVecs_[curStream].hDataArray + rj * veColumnNum + relationCompVecs_[curStream].hSizeArray[rj]);
                                vertexEdgeOffsets[j+1] = vertexEdgeOffsets[j] + relationCompVecs_[curStream].hSizeArray[rj];
                            }
                            allClusters_[sid].vertexEdges_.setData(std::move(vertexEdgeData), std::move(vertexEdgeOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }

                #ifdef MEASURE_WAITING
                returnTime += tmp.getElapsedTime();
                #endif

                delete[] relationCompVecs_[curStream].hSizeArray;
                delete[] relationCompVecs_[curStream].hDataArray;
                break;
            }
            
            case RelationType::VFRelation:
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].vertexTriangles_.empty()) {
                            std::vector<SimplexId> vertexTriangleData{};
                            int rvid = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                            std::vector<SimplexId> vertexTriangleOffsets(localVertexNum + 1);
                            vertexTriangleOffsets[0] = 0;
                            for (int vid = 0; vid < localVertexNum; ++vid) {
                                vertexTriangleData.insert(vertexTriangleData.end(), 
                                relationCompVecs_[curStream].hDataArray + rvid * vfColumnNum,
                                relationCompVecs_[curStream].hDataArray + rvid * vfColumnNum + relationCompVecs_[curStream].hSizeArray[rvid]);
                                vertexTriangleOffsets[vid+1] = vertexTriangleOffsets[vid] + relationCompVecs_[curStream].hSizeArray[rvid];
                                ++rvid;
                            }
                            allClusters_[sid].vertexTriangles_.setData(std::move(vertexTriangleData), std::move(vertexTriangleOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }
                #ifdef MEASURE_WAITING
                returnTime += tmp.getElapsedTime();
                #endif

                delete[] relationCompVecs_[curStream].hDataArray;
                delete[] relationCompVecs_[curStream].hSizeArray;
                break;
            }
            
            case RelationType::VTRelation:
            {
                

                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        std::vector<SimplexId> vertexStarData{};
                        int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                        int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                        std::vector<SimplexId> vertexStarOffsets(localVertexNum + 1);
                        vertexStarOffsets[0] = 0;
                        for (int j = 0; j < localVertexNum; ++j) {
                            int rj = j + diff; // relative j value in the large matrix
                            vertexStarData.insert(vertexStarData.end(), 
                            relationCompVecs_[curStream].hDataArray + rj * vtColumnNum,
                            relationCompVecs_[curStream].hDataArray + rj * vtColumnNum + relationCompVecs_[curStream].hSizeArray[rj]);
                            vertexStarOffsets[j+1] = vertexStarOffsets[j] + relationCompVecs_[curStream].hSizeArray[rj];
                        }
                        if (allClusters_[sid].vertexStars_.empty()) {
                            allClusters_[sid].vertexStars_.setData(std::move(vertexStarData), std::move(vertexStarOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }
                #ifdef MEASURE_WAITING
                returnTime += tmp.getElapsedTime();
                #endif

                delete[] relationCompVecs_[curStream].hDataArray;
                delete[] relationCompVecs_[curStream].hSizeArray;
                break;
            }

            case RelationType::VVRelation:
            {
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNum)
                // #endif
                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (SimplexId sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        // boost::unique_lock<boost::shared_mutex> writeLock(clusterMutexes_[sid]);
                        std::vector<SimplexId> vertexNeighborData{};
                        int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                        int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                        std::vector<SimplexId> vertexNeighborOffsets(localVertexNum + 1);
                        vertexNeighborOffsets[0] = 0;
                        for (int j = 0; j < localVertexNum; ++j) {
                            int rj = j + diff; // relative j value in the large matrix
                            vertexNeighborData.insert(vertexNeighborData.end(), 
                            relationCompVecs_[curStream].hDataArray + rj * vvColumnNum,
                            relationCompVecs_[curStream].hDataArray + rj * vvColumnNum + relationCompVecs_[curStream].hSizeArray[rj]);
                            vertexNeighborOffsets[j+1] = vertexNeighborOffsets[j] + relationCompVecs_[curStream].hSizeArray[rj];
                        }
                        if (allClusters_[sid].vertexNeighbors_.empty()) {
                            allClusters_[sid].vertexNeighbors_.setData(std::move(vertexNeighborData), std::move(vertexNeighborOffsets));
                        }
                        // writeLock.unlock();
                    }
                    sem_post(&relationSems_[tid]);
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                }
                #ifdef MEASURE_WAITING
                returnTime += tmp.getElapsedTime();
                #endif

                delete[] relationCompVecs_[curStream].hDataArray;
                delete[] relationCompVecs_[curStream].hSizeArray;
                break;
            }
        }
    }

    #ifdef MEASURE_WAITING
    kernelTimes_.push_back(kernelTime/1000);    // convert to seconds
    returnTimes_.push_back(returnTime);
    #endif
    
    return res;
}


/**
 * Procedure of the leader producer.
*/
void Gale::leaderProducer() {
    volatile SimplexId leaderCluster = 0;
    volatile RelationType leaderFunc = RelationType::BlankRelation;

    if (threadNumber_ == 1) {
        while (true) {
            std::unique_lock<std::mutex> qlock(qMutex_);
            int qSize = jobQueue.size();
            qlock.unlock();
            if (qSize > 0) {
                int tid = jobQueue.front();
                leaderCluster = currClusters_[tid], leaderFunc = currRelations_[tid];
                int res = computeRelation(leaderFunc, leaderCluster, false, tid, true);
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
                // if (diff < 0 || diff > blocksPerThread_ * blockSize_ * 0.6) {
                //     // std::cout << "precomputing for thread " << i << " cluster = " << leaderCluster << " diff = " << diff << std::endl;
                //     computeRelation(leaderFunc, leaderCluster, i, false);
                // }
                
                for (RelationType &relation : preconditionedRelations_) {
                    if (relation == currRelations_[0]) continue;
                    SimplexId startCluster = baseClusters_[static_cast<int>(relation)][0];
                    // leaderCluster = (leaderCluster - startCluster) * removalRatio + startCluster;
                    if (startCluster == 0) continue;
                    // different relations correspond to different vectors
                    switch (relation)
                    {
                        case RelationType::BVRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].boundaryVertices_ = std::vector<bool>();
                            }
                            break;
                        }

                        case RelationType::BERelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].boundaryEdges_ = std::vector<bool>();
                            }
                            break;
                        }
                        
                        case RelationType::EFRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].edgeTriangles_.clear();
                            }
                            break;
                        }
                        
                        case RelationType::ETRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].edgeStars_.clear();
                            }
                            break;
                        }
                        
                        case RelationType::FTRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].triangleStars_.clear();
                            }
                            break;
                        }

                        case RelationType::VVRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].vertexNeighbors_.clear();
                            }
                            break;
                        }
                    
                        case RelationType::VERelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].vertexEdges_.clear();
                            }
                            break;
                        }

                        case RelationType::VFRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].vertexTriangles_.clear();
                            }
                            break;
                        }

                        case RelationType::VTRelation:
                        {
                            for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                                allClusters_[cid].vertexStars_.clear();
                            }
                            break;
                        }
                        
                        default:
                            break;
                        
                    }
                    // reset the base cluster of the thread to 0
                    baseClusters_[static_cast<int>(relation)][0] = 0;
                }
            }
        }
    }
    // multiple threads are used
    else {
        // std::vector<std::vector<std::tuple<int, int, RelationType>>> qstats;
        while (true) {
            /* Job queue version */
            // save jobs in the current queue to the vector
            std::unique_lock<std::mutex> qlock(qMutex_);
            std::vector<int> localQ;
            while (!jobQueue.empty()) {
                localQ.push_back(jobQueue.front());
                jobQueue.pop();
            }
            qlock.unlock();

            if (!localQ.empty()) {
                // initialization that executes only once
                for (static bool first = true; first; first = false) {
                    threadStreams_.resize(preconditionedRelations_.size());
                    relationCompVecs_.resize(preconditionedRelations_.size());
                    // preallocate a large chunk of memory for GPU computation 
                    int allocatedSize = threadNumber_ * precompClusterNumber_ * 500;
                    for (int i = 0; i < relationCompVecs_.size(); ++i) {
                        CHECK(cudaStreamCreate(&threadStreams_[i]));
                        CHECK(cudaMallocAsync((void **)&relationCompVecs_[i].dClusterIds, sizeof(int) * threadNumber_, threadStreams_[i]));
                        CHECK(cudaMallocAsync((void **)&relationCompVecs_[i].dSimplexOffsets, sizeof(int) * threadNumber_, threadStreams_[i]));
                        CHECK(cudaMallocAsync((void **)&relationCompVecs_[i].dSizeArray, sizeof(int) * allocatedSize, threadStreams_[i]));
                        CHECK(cudaMallocAsync((void **)&relationCompVecs_[i].dDataArray, sizeof(int) * allocatedSize * 80, threadStreams_[i]));
                    }
                }

                Timer t;
                // handle the waiting jobs in the queue
                // combine the requests from all waiting consumers
                std::vector<RelationType> relationOrders;
                std::map<RelationType, std::vector<std::pair<int,int>>> requestMap;
                for (int i = 0; i < localQ.size(); ++i) {
                    int tid = localQ[i];
                    RelationType jobType = currRelations_[tid];
                    if (requestMap.find(jobType) == requestMap.end()) {
                        relationOrders.push_back(jobType);
                    }
                    requestMap[jobType].push_back(std::make_pair(tid, currClusters_[tid]));
                    // curq.push_back(std::make_tuple(tid, currClusters_[tid], jobType));
                }
                computeCombinedRelation(requestMap, relationOrders);
                qTimes_.push_back(t.getElapsedTime());
                qSizes_.push_back(localQ.size());
            }
            else {
                // clean the memory storage
                for (int i = 0; i < threadNumber_; ++i) {
                    leaderCluster = currClusters_[i];
                    if (leaderCluster < 0) {
                        for (int i = 0; i < relationCompVecs_.size(); ++i) {
                            CHECK(cudaStreamDestroy(threadStreams_[i]));
                            CHECK(cudaFree(relationCompVecs_[i].dClusterIds));
                            CHECK(cudaFree(relationCompVecs_[i].dSimplexOffsets));
                            CHECK(cudaFree(relationCompVecs_[i].dSizeArray));
                            CHECK(cudaFree(relationCompVecs_[i].dDataArray));
                        }
                        return;
                    }
                    else if (leaderCluster == 0) continue;
                    if (!jobQueue.empty()) break;
                    // SimplexId startCluster = baseClusters_[static_cast<int>(currRelations_[i])][i];
                    // for (int cid = startCluster; cid < leaderCluster - 1; ++cid) {
                    //     if (clusterMutexes_[cid].try_lock())
                    //         allClusters_[cid].clear();
                    // }
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
    }
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

    producer_ = std::thread(&Gale::leaderProducer, this);
    printf("[Gale] GPU threads per cluster: %d, thread num: %d, block number: %d, block size: %d, cluster number: %d, cell number: %d, vertex number: %d\n", threadsPerCluster_, threadNumber_, blocksPerThread_, blockSize_, clusterNumber_, cellNumber_, vertexNumber_);

    // #ifdef MEASURE_WAITING
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
    CHECK(cudaMemcpy(dExternalCells_, hExternalCells_, sizeof(int) * hExternalCellOffsets_[clusterNumber_-1], cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(dVertexIntervals_, hVertexIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice, 0));

    // #ifdef MEASURE_WAITING
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
