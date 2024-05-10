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
    waitingTimes_ = std::vector<double>(threadNumber_, 0);
    waitingCounts_ = std::vector<int>(threadNumber_, 0);
    #endif

    // GPU parameters
    blocksPerThread_ = 1, blockSize_ = 32;
}


/**
 * Destructor of GALE
*/
Gale::~Gale() {
    if (producer_.joinable()) {
        currClusters_[0] = -1;
        producer_.join();
    }

    #ifdef MEASURE_WAITING
    for (int i = 0; i < threadNumber_; ++i)
        std::cout << "waiting count " << i << ": " << waitingCounts_[i] << ", waiting time: " << waitingTimes_[i] << std::endl;
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
            int numBlocks = clusterNumber_ / (blockSize_ / threadsPerCluster_) + 1;
            int localTriangleNum = triangleNumber_;
            CHECK(cudaMalloc((void **)&dBoundaryTriangles_, sizeof(bool) * localTriangleNum));
            CHECK(cudaMalloc((void **)&dTriangleStarSizes_, sizeof(int) * localTriangleNum));
            CHECK(cudaMalloc((void **)&dTriangleStarMatrix_, sizeof(int) * localTriangleNum * ftColumnNum));
            CHECK(cudaMemset(dBoundaryTriangles_, false, sizeof(bool) * localTriangleNum));
            CHECK(cudaMemset(dTriangleStarSizes_, 0, sizeof(int) * localTriangleNum));
            CHECK(cudaMemset(dTriangleStarMatrix_, -1, sizeof(int) * localTriangleNum * ftColumnNum));

            computeBoundaryTriangles<<<numBlocks, blockSize_>>>(
                dTriangleList_, dTriangleStarMatrix_, 
                dTriangleStarSizes_, dBoundaryTriangles_,
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

            CHECK(cudaFree(dTriangleStarSizes_));
            CHECK(cudaFree(dTriangleStarMatrix_));
            delete[] hBoundaryTriangleList;

            break;
        }

        case RelationType::EFRelation:
        {
            int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
            int localEdgeSum = hEdgeIntervals_[endCluster] - edgeIdOffset;
            CHECK(cudaMalloc((void **)&dEdgeTriangleSizes_, sizeof(int) * localEdgeSum));
            CHECK(cudaMalloc((void **)&dEdgeTriangleMatrix_, sizeof(int) * localEdgeSum * efColumnNum));
            CHECK(cudaMemset(dEdgeTriangleSizes_, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(dEdgeTriangleMatrix_, -1, sizeof(int) * localEdgeSum * efColumnNum));

            computeEdgeTriangles<<<numBlocks, blockSize_>>>(
                dEdgeList_, dTriangleList_,
                dEdgeTriangleMatrix_, dEdgeTriangleSizes_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            hEdgeTriangleSizes_ = new int[localEdgeSum];
            hEdgeTriangleMatrix_ = new int[localEdgeSum * efColumnNum];

            CHECK(cudaMemcpy(hEdgeTriangleSizes_, dEdgeTriangleSizes_, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hEdgeTriangleMatrix_, dEdgeTriangleMatrix_, sizeof(int) * localEdgeSum * efColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(EFMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].edgeTriangles_.empty()) {
                    int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                    std::vector<SimplexId> edgeTriangleData{};
                    std::vector<SimplexId> edgeTriangleOffsets(localEdgeNum + 1);
                    edgeTriangleOffsets[0] = 0;
                    int reid = hEdgeIntervals_[sid - 1] - edgeIdOffset;
                    for (int eid = 0; eid < localEdgeNum; ++eid) {
                        edgeTriangleData.insert(edgeTriangleData.end(), hEdgeTriangleMatrix_ + reid * efColumnNum, hEdgeTriangleMatrix_ + reid * efColumnNum + hEdgeTriangleSizes_[reid]);
                        edgeTriangleOffsets[eid+1] = edgeTriangleOffsets[eid] + hEdgeTriangleSizes_[reid];
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

            CHECK(cudaFree(dEdgeTriangleSizes_));
            CHECK(cudaFree(dEdgeTriangleMatrix_));

            delete[] hEdgeTriangleMatrix_;
            delete[] hEdgeTriangleSizes_;
            break;
        }

        case RelationType::ETRelation:
        {
            int edgeIdOffset = hEdgeIntervals_[clusterId - 1];
            int localEdgeSum = hEdgeIntervals_[endCluster] - edgeIdOffset;
            CHECK(cudaMalloc((void **)&dEdgeStarSizes_, sizeof(int) * localEdgeSum));
            CHECK(cudaMalloc((void **)&dEdgeStarMatrix_, sizeof(int) * localEdgeSum * etColumnNum));
            CHECK(cudaMemset(dEdgeStarSizes_, 0, sizeof(int) * localEdgeSum));
            CHECK(cudaMemset(dEdgeStarMatrix_, -1, sizeof(int) * localEdgeSum * etColumnNum));

            computeEdgeStars<<<numBlocks, blockSize_>>>(
                dEdgeList_, dEdgeStarMatrix_, dEdgeStarSizes_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            hEdgeStarSizes_ = new int[localEdgeSum];
            hEdgeStarMatrix_ = new int[localEdgeSum * etColumnNum];

            CHECK(cudaMemcpy(hEdgeStarSizes_, dEdgeStarSizes_, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hEdgeStarMatrix_, dEdgeStarMatrix_, sizeof(int) * localEdgeSum * etColumnNum, cudaMemcpyDeviceToHost));

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

            CHECK(cudaFree(dEdgeStarSizes_));
            CHECK(cudaFree(dEdgeStarMatrix_));

            delete[] hEdgeStarMatrix_;
            delete[] hEdgeStarSizes_;
            break;
        }

        case RelationType::FTRelation:
        {
            int triangleIdOffset = hTriangleIntervals_[clusterId - 1];
            int localTriangleSum = hTriangleIntervals_[endCluster] - triangleIdOffset;
            CHECK(cudaMalloc((void **)&dTriangleStarSizes_, sizeof(int) * localTriangleSum));
            CHECK(cudaMalloc((void **)&dTriangleStarMatrix_, sizeof(int) * localTriangleSum * ftColumnNum));
            CHECK(cudaMemset(dTriangleStarSizes_, 0, sizeof(int) * localTriangleSum));
            CHECK(cudaMemset(dTriangleStarMatrix_, -1, sizeof(int) * localTriangleSum * ftColumnNum));

            computeTriangleStars<<<numBlocks, blockSize_>>>(
                dTriangleList_, dTriangleStarMatrix_, dTriangleStarSizes_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            hTriangleStarSizes_ = new int[localTriangleSum];
            hTriangleStarMatrix_ = new int[localTriangleSum * ftColumnNum];

            CHECK(cudaMemcpy(hTriangleStarSizes_, dTriangleStarSizes_, sizeof(int) * localTriangleSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hTriangleStarMatrix_, dTriangleStarMatrix_, sizeof(int) * localTriangleSum * ftColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(FTMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].triangleStars_.empty()) {
                    int localTriangleNum = hTriangleIntervals_[sid] - hTriangleIntervals_[sid - 1];
                    std::vector<SimplexId> triangleStarData{};
                    std::vector<SimplexId> triangleStarOffsets(localTriangleNum + 1);
                    triangleStarOffsets[0] = 0;
                    int rfid = hTriangleIntervals_[sid - 1] - triangleIdOffset;
                    for (int fid = 0; fid < localTriangleNum; ++fid) {
                        triangleStarData.insert(triangleStarData.end(), hTriangleStarMatrix_ + rfid * ftColumnNum, hTriangleStarMatrix_ + rfid * ftColumnNum + hTriangleStarSizes_[rfid]);
                        triangleStarOffsets[fid+1] = triangleStarOffsets[fid] + hTriangleStarSizes_[rfid];
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

            CHECK(cudaFree(dTriangleStarSizes_));
            CHECK(cudaFree(dTriangleStarMatrix_));
            delete[] hTriangleStarMatrix_;
            delete[] hTriangleStarSizes_;

            break;
        }

        case RelationType::VERelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexSum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexEdgeSizes_, sizeof(int) * localVertexSum));
            CHECK(cudaMalloc((void **)&dVertexEdgeMatrix_, sizeof(int) * localVertexSum * veColumnNum));
            CHECK(cudaMemset(dVertexEdgeSizes_, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(dVertexEdgeMatrix_, -1, sizeof(int) * localVertexSum * veColumnNum));

            computeVertexEdges<<<numBlocks, blockSize_>>>(
                dEdgeList_, dVertexEdgeMatrix_, dVertexEdgeSizes_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            hVertexEdgeSizes_ = new int[localVertexSum];
            hVertexEdgeMatrix_ = new int[localVertexSum * veColumnNum];

            CHECK(cudaMemcpy(hVertexEdgeSizes_, dVertexEdgeSizes_, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexEdgeMatrix_, dVertexEdgeMatrix_, sizeof(int) * localVertexSum * veColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VEMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexEdges_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexEdgeData{};
                    std::vector<SimplexId> vertexEdgeOffsets(localVertexNum + 1);
                    vertexEdgeOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexEdgeData.insert(vertexEdgeData.end(), hVertexEdgeMatrix_ + rvid * veColumnNum, hVertexEdgeMatrix_ + rvid * veColumnNum + hVertexEdgeSizes_[rvid]);
                        vertexEdgeOffsets[vid+1] = vertexEdgeOffsets[vid] + hVertexEdgeSizes_[rvid];
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

            CHECK(cudaFree(dVertexEdgeSizes_));
            CHECK(cudaFree(dVertexEdgeMatrix_));
            delete[] hVertexEdgeMatrix_;
            delete[] hVertexEdgeSizes_;

            break;
        }

        case RelationType::VFRelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexSum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexTriangleSizes_, sizeof(int) * localVertexSum));
            CHECK(cudaMalloc((void **)&dVertexTriangleMatrix_, sizeof(int) * localVertexSum * vfColumnNum));
            CHECK(cudaMemset(dVertexTriangleSizes_, 0, sizeof(int) * localVertexSum));
            CHECK(cudaMemset(dVertexTriangleMatrix_, -1, sizeof(int) * localVertexSum * vfColumnNum));

            computeVertexTriangles<<<numBlocks, blockSize_>>>(
                dTriangleList_, dVertexTriangleMatrix_, dVertexTriangleSizes_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_, dIndexArray_,
                dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            hVertexTriangleSizes_ = new int[localVertexSum];
            hVertexTriangleMatrix_ = new int[localVertexSum * vfColumnNum];

            CHECK(cudaMemcpy(hVertexTriangleSizes_, dVertexTriangleSizes_, sizeof(int) * localVertexSum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexTriangleMatrix_, dVertexTriangleMatrix_, sizeof(int) * localVertexSum * vfColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VFMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexTriangles_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexTriangleData{};
                    std::vector<SimplexId> vertexTriangleOffsets(localVertexNum + 1);
                    vertexTriangleOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexTriangleData.insert(vertexTriangleData.end(), hVertexTriangleMatrix_ + rvid * vfColumnNum, hVertexTriangleMatrix_ + rvid * vfColumnNum + hVertexTriangleSizes_[rvid]);
                        vertexTriangleOffsets[vid+1] = vertexTriangleOffsets[vid] + hVertexTriangleSizes_[rvid];
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

            CHECK(cudaFree(dVertexTriangleSizes_));
            CHECK(cudaFree(dVertexTriangleMatrix_));
            delete[] hVertexTriangleMatrix_;
            delete[] hVertexTriangleSizes_;

            break;
        }

        case RelationType::VTRelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexStarSizes_, sizeof(int) * localVertexNum));
            CHECK(cudaMalloc((void **)&dVertexStarMatrix_, sizeof(int) * localVertexNum * vtColumnNum));
            CHECK(cudaMemset(dVertexStarSizes_, 0, sizeof(int) * localVertexNum));
            CHECK(cudaMemset(dVertexStarMatrix_, -1, sizeof(int) * localVertexNum * vtColumnNum));

            computeVertexStars<<<numBlocks, blockSize_>>>(
                dVertexStarMatrix_, dVertexStarSizes_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            hVertexStarSizes_ = new int[localVertexNum];
            hVertexStarMatrix_ = new int[localVertexNum * vtColumnNum];

            CHECK(cudaMemcpy(hVertexStarSizes_, dVertexStarSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexStarMatrix_, dVertexStarMatrix_, sizeof(int) * localVertexNum * vtColumnNum, cudaMemcpyDeviceToHost));

            // std::unique_lock<std::mutex> lck(VTMutex_);
            for (int sid = clusterId; sid <= endCluster; ++sid) {
                if (allClusters_[sid].vertexStars_.empty()) {
                    int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                    std::vector<SimplexId> vertexStarData{};
                    std::vector<SimplexId> vertexStarOffsets(localVertexNum + 1);
                    vertexStarOffsets[0] = 0;
                    int rvid = hVertexIntervals_[sid - 1] - vertexIdOffset;
                    for (int vid = 0; vid < localVertexNum; ++vid) {
                        vertexStarData.insert(vertexStarData.end(), hVertexStarMatrix_ + rvid * vtColumnNum, hVertexStarMatrix_ + rvid * vtColumnNum + hVertexStarSizes_[rvid]);
                        vertexStarOffsets[vid+1] = vertexStarOffsets[vid] + hVertexStarSizes_[rvid];
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

            CHECK(cudaFree(dVertexStarSizes_));
            CHECK(cudaFree(dVertexStarMatrix_));
            delete[] hVertexStarMatrix_;
            delete[] hVertexStarSizes_;

            break;
        }

        case RelationType::VVRelation:
        {
            int vertexIdOffset = hVertexIntervals_[clusterId - 1];
            int localVertexNum = hVertexIntervals_[endCluster] - vertexIdOffset;
            CHECK(cudaMalloc((void **)&dVertexNeighborSizes_, sizeof(int) * localVertexNum));
            CHECK(cudaMalloc((void **)&dVertexNeighborMatrix_, sizeof(int) * localVertexNum * vvColumnNum));
            CHECK(cudaMemset(dVertexNeighborSizes_, 0, sizeof(int) * localVertexNum));
            CHECK(cudaMemset(dVertexNeighborMatrix_, -1, sizeof(int) * localVertexNum * vvColumnNum));
            
            computeVertexNeighbors<<<numBlocks, blockSize_>>>(
                dVertexNeighborMatrix_, dVertexNeighborSizes_,
                clusterId, clusterNumber_,
                dConnectivity_, dOffsets_,
                dVertexIntervals_, dCellIntervals_,
                dExternalCells_, dExternalCellOffsets_, 
                cellDimension_, threadsPerCluster_);

            hVertexNeighborSizes_ = new int[localVertexNum];
            hVertexNeighborMatrix_ = new int[localVertexNum * vvColumnNum];

            CHECK(cudaMemcpy(hVertexNeighborSizes_, dVertexNeighborSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexNeighborMatrix_, dVertexNeighborMatrix_, sizeof(int) * localVertexNum * vvColumnNum, cudaMemcpyDeviceToHost));

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

            CHECK(cudaFree(dVertexNeighborSizes_));
            CHECK(cudaFree(dVertexNeighborMatrix_));

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
    std::vector<cudaStream_t> threadStreams(numRelations);
    std::vector<std::vector<SimplexId>> hEndClusters(numRelations);
    std::vector<std::vector<int>> hSimplexOffsets(numRelations);
    std::vector<int *> dClustersIds(numRelations);
    std::vector<int *> dSimplexOffsets(numRelations);

    for (int curStream = 0; curStream < numRelations; ++curStream) {
        // (relation, [(tid, clusterId), ...])
        RelationType &relation = relationOrders[curStream];
        CHECK(cudaStreamCreate(&threadStreams[curStream]));
        // int relationId = static_cast<int>(relation);
        std::vector<std::pair<int,int>> &jobs = requestMap[relation];

        int threadNum = jobs.size();
        int gpuBlocks = blocksPerThread_ * threadNum;
        int gpuThreads = blockSize_ / threadsPerCluster_ * gpuBlocks;

        // launches different kernels on different streams
        switch (relation) {
            case RelationType::BERelation:
            {
                int localEdgeSum = 0;
                int hClusterIds[threadNum];
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localEdgeNum = hEdgeIntervals_[hEndClusters[curStream][i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNum;
                }
                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dBoundaryEdges_, sizeof(bool) * localEdgeSum, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dBoundaryEdges_, false, sizeof(bool) * localEdgeSum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiBoundaryEdges<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dEdgeList_, dTriangleList_,
                    dBoundaryEdges_, dBoundaryTriangles_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream],
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hBoundaryEdges_ = new bool[localEdgeSum];

                CHECK(cudaMemcpyAsync(hBoundaryEdges_, dBoundaryEdges_, sizeof(bool) * localEdgeSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));

                break;
            }

            case RelationType::BVRelation: 
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
                CHECK(cudaMallocAsync((void **)&dBoundaryVertices_, sizeof(bool) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dBoundaryVertices_, false, sizeof(bool) * localVertexSum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiBoundaryVertices<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dTriangleList_, dBoundaryTriangles_, dBoundaryVertices_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream], 
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hBoundaryVertices_ = new bool[localVertexSum];

                CHECK(cudaMemcpyAsync(hBoundaryVertices_, dBoundaryVertices_, sizeof(bool) * localVertexSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));

                break;
            }
            
            case RelationType::EFRelation:
            {
                int localEdgeSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localEdgeNums = hEdgeIntervals_[hEndClusters[curStream][i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNums;
                }
                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dEdgeTriangleSizes_, sizeof(int) * localEdgeSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dEdgeTriangleMatrix_, sizeof(int) * localEdgeSum * efColumnNum, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dEdgeTriangleSizes_, 0, sizeof(int) * localEdgeSum, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dEdgeTriangleMatrix_, -1, sizeof(int) * localEdgeSum * efColumnNum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiEdgeTriangles<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dEdgeList_, dTriangleList_,
                    dEdgeTriangleMatrix_, dEdgeTriangleSizes_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream],
                    dConnectivity_, dOffsets_, dIndexArray_,
                    dVertexIntervals_, dEdgeIntervals_, dTriangleIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hEdgeTriangleSizes_ = new int[localEdgeSum];
                hEdgeTriangleMatrix_ = new int[localEdgeSum * efColumnNum];

                CHECK(cudaMemcpyAsync(hEdgeTriangleSizes_, dEdgeTriangleSizes_, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(hEdgeTriangleMatrix_, dEdgeTriangleMatrix_, sizeof(int) * localEdgeSum * efColumnNum, cudaMemcpyDeviceToHost, threadStreams[curStream]));

                break;
            }

            case RelationType::ETRelation:
            {
                int localEdgeSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localEdgeSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localEdgeNums = hEdgeIntervals_[hEndClusters[curStream][i]] - hEdgeIntervals_[hClusterIds[i] - 1];
                    localEdgeSum += localEdgeNums;
                }
                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dEdgeStarSizes_, sizeof(int) * localEdgeSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dEdgeStarMatrix_, sizeof(int) * localEdgeSum * etColumnNum, threadStreams[curStream]));

                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemset(dEdgeStarSizes_, 0, sizeof(int) * localEdgeSum));
                CHECK(cudaMemset(dEdgeStarMatrix_, -1, sizeof(int) * localEdgeSum * etColumnNum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiEdgeStars<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dEdgeList_, dEdgeStarMatrix_, dEdgeStarSizes_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream], 
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dEdgeIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hEdgeStarSizes_ = new int[localEdgeSum];
                hEdgeStarMatrix_ = new int[localEdgeSum * etColumnNum];

                CHECK(cudaMemcpyAsync(hEdgeStarSizes_, dEdgeStarSizes_, sizeof(int) * localEdgeSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(hEdgeStarMatrix_, dEdgeStarMatrix_, sizeof(int) * localEdgeSum * etColumnNum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                
                break;
            }

            case RelationType::FTRelation:
            {
                int localTriangleSum = 0;
                hSimplexOffsets[curStream].resize(threadNum);
                hEndClusters[curStream].resize(threadNum);
                int hClusterIds[threadNum];
                int maxEndCluster = 0;
                for (int i = 0; i < threadNum; ++i) {
                    hSimplexOffsets[curStream][i] = localTriangleSum;
                    hClusterIds[i] = jobs[i].second;
                    hEndClusters[curStream][i] = std::min(hClusterIds[i] + gpuThreads, clusterNumber_) - 1;
                    maxEndCluster = std::max(hEndClusters[curStream][i], maxEndCluster);
                    int localTriangleNums = hTriangleIntervals_[hEndClusters[curStream][i]] - hTriangleIntervals_[hClusterIds[i] - 1];
                    localTriangleSum += localTriangleNums;
                }

                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dTriangleStarSizes_, sizeof(int) * localTriangleSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dTriangleStarMatrix_, sizeof(int) * localTriangleSum * ftColumnNum, threadStreams[curStream]));

                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemset(dTriangleStarSizes_, 0, sizeof(int) * localTriangleSum));
                CHECK(cudaMemset(dTriangleStarMatrix_, -1, sizeof(int) * localTriangleSum * ftColumnNum));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiTriangleStars<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dTriangleList_, dTriangleStarMatrix_, dTriangleStarSizes_,
                    clusterNumber_, dClustersIds[curStream], dSimplexOffsets[curStream],
                    dConnectivity_, dOffsets_,
                    dVertexIntervals_, dTriangleIntervals_, dCellIntervals_,
                    dExternalCells_, dExternalCellOffsets_, 
                    cellDimension_, threadsPerCluster_);

                hTriangleStarSizes_ = new int[localTriangleSum];
                hTriangleStarMatrix_ = new int[localTriangleSum * ftColumnNum];

                CHECK(cudaMemcpyAsync(hTriangleStarSizes_, dTriangleStarSizes_, sizeof(int) * localTriangleSum, cudaMemcpyDeviceToHost, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(hTriangleStarMatrix_, dTriangleStarMatrix_, sizeof(int) * localTriangleSum * ftColumnNum, cudaMemcpyDeviceToHost, threadStreams[curStream]));

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

                // CHECK(cudaMallocAsync((void **)&dEdgeLists[curStream], sizeof(int) * hEdgeIntervals_[maxEndCluster] * 2, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dClustersIds[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dSimplexOffsets[curStream], sizeof(int) * threadNum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexEdgeSizes_, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexEdgeMatrix_, sizeof(int) * localVertexSum * veColumnNum, threadStreams[curStream]));

                // CHECK(cudaMemcpyAsync(dEdgeLists[curStream], edgeList_.data(), sizeof(SimplexId) * hEdgeIntervals_[maxEndCluster] * 2, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexEdgeSizes_, 0, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexEdgeMatrix_, -1, sizeof(int) * localVertexSum * veColumnNum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiVertexEdges<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dEdgeList_, dVertexEdgeMatrix_, dVertexEdgeSizes_,
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
                // CHECK(cudaMallocAsync((void **)&dTriangleLists[curStream], sizeof(int) * hTriangleIntervals_[maxEndCluster] * 3, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexTriangleSizes_, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMallocAsync((void **)&dVertexTriangleMatrix_, sizeof(int) * localVertexSum * vfColumnNum, threadStreams[curStream]));

                // CHECK(cudaMemcpyAsync(dTriangleLists[curStream], triangleList_.data(), sizeof(SimplexId) * hTriangleIntervals_[maxEndCluster] * 3, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dClustersIds[curStream], hClusterIds, sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemcpyAsync(dSimplexOffsets[curStream], hSimplexOffsets[curStream].data(), sizeof(int) * threadNum, cudaMemcpyHostToDevice, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexTriangleSizes_, 0, sizeof(int) * localVertexSum, threadStreams[curStream]));
                CHECK(cudaMemsetAsync(dVertexTriangleMatrix_, -1, sizeof(int) * localVertexSum * vfColumnNum, threadStreams[curStream]));

                dim3 dimBlock(threadNum, gpuBlocks);
                computeMultiVertexTriangles<<<dimBlock, blockSize_, 0, threadStreams[curStream]>>>(
                    dTriangleList_, dVertexTriangleMatrix_, dVertexTriangleSizes_,
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
            case RelationType::BERelation: 
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClusterId = jobs[i].second;
                    for (int sid = baseClusterId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].boundaryEdges_.empty()) {
                            int localEdgeNum = hEdgeIntervals_[sid] - hEdgeIntervals_[sid - 1];
                            int relativeEid = hEdgeIntervals_[sid - 1] - hEdgeIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            allClusters_[sid].boundaryEdges_.resize(localEdgeNum);
                            for (int eid = 0; eid < localEdgeNum; ++eid) {
                                allClusters_[sid].boundaryEdges_[eid] = hBoundaryEdges_[relativeEid++];
                            }
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dBoundaryEdges_));
                delete[] hBoundaryEdges_;

                break;
            }
            case RelationType::BVRelation:
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

                for (int i = 0; i < threadNum; ++i) {
                    SimplexId tid = jobs[i].first;
                    SimplexId baseClustersId = jobs[i].second;
                    // std::unique_lock<std::mutex> lck(VVMutex_);
                    for (int sid = baseClustersId; sid <= hEndClusters[curStream][i]; ++sid) {
                        if (allClusters_[sid].boundaryVertices_.empty()) {
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid-1];
                            int relativeVid = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClustersId - 1] + hSimplexOffsets[curStream][i];
                            allClusters_[sid].boundaryVertices_.resize(localVertexNum);
                            for (int vid = 0; vid < localVertexNum; ++vid) {
                                allClusters_[sid].boundaryVertices_[vid] = hBoundaryVertices_[relativeVid++];
                            }
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClustersId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dBoundaryVertices_));
                delete[] hBoundaryVertices_;

                break;
            }
            
            case RelationType::EFRelation:
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

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
                                hEdgeTriangleMatrix_ + reid * efColumnNum,
                                hEdgeTriangleMatrix_ + reid * efColumnNum + hEdgeTriangleSizes_[reid]);
                                edgeTriangleOffsets[eid+1] = edgeTriangleOffsets[eid] + hEdgeTriangleSizes_[reid];
                                ++reid;
                            }
                            allClusters_[sid].edgeTriangles_.setData(std::move(edgeTriangleData), std::move(edgeTriangleOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0) 
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dEdgeTriangleSizes_));
                CHECK(cudaFree(dEdgeTriangleMatrix_));
                delete[] hEdgeTriangleMatrix_;
                delete[] hEdgeTriangleSizes_;
                
                break;
            }
            
            case RelationType::ETRelation: 
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }

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
                                hEdgeStarMatrix_ + reid * etColumnNum,
                                hEdgeStarMatrix_ + reid * etColumnNum + hEdgeStarSizes_[reid]);
                                edgeStarOffsets[j+1] = edgeStarOffsets[j] + hEdgeStarSizes_[reid];
                                ++reid;
                            }
                            allClusters_[sid].edgeStars_.setData(std::move(edgeStarData), std::move(edgeStarOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0) 
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dEdgeStarSizes_));
                CHECK(cudaFree(dEdgeStarMatrix_));
                delete[] hEdgeStarMatrix_;
                delete[] hEdgeStarSizes_;

                break;
            }

            case RelationType::FTRelation:
            {
                cudaError_t status = cudaStreamSynchronize(threadStreams[curStream]);
                if (status != cudaSuccess) {
                    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(status),
                            __FILE__, __LINE__);
                    if (abort)
                        exit(status);
                }
                
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
                                hTriangleStarMatrix_ + reid * ftColumnNum,
                                hTriangleStarMatrix_ + reid * ftColumnNum + hTriangleStarSizes_[reid]);
                                triangleStarOffsets[j+1] = triangleStarOffsets[j] + hTriangleStarSizes_[reid];
                                ++reid;
                            }
                            allClusters_[sid].triangleStars_.setData(std::move(triangleStarData), std::move(triangleStarOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
                CHECK(cudaFree(dSimplexOffsets[curStream]));
                CHECK(cudaFree(dTriangleStarSizes_));
                CHECK(cudaFree(dTriangleStarMatrix_));
                delete[] hTriangleStarMatrix_;
                delete[] hTriangleStarSizes_;

                break;
            }

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
                        if (allClusters_[sid].vertexEdges_.empty()) {
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
                            allClusters_[sid].vertexEdges_.setData(std::move(vertexEdgeData), std::move(vertexEdgeOffsets));
                        }
                    }
                    if (baseClusters_[relationId][tid] == 0)
                        baseClusters_[relationId][tid] = baseClusterId;
                    // lck.unlock();
                    sem_post(&relationSems_[tid]);
                }

                CHECK(cudaFree(dClustersIds[curStream]));
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
                        if (allClusters_[sid].vertexTriangles_.empty()) {
                            std::vector<SimplexId> vertexTriangleData{};
                            int rvid = hVertexIntervals_[sid - 1] - hVertexIntervals_[baseClusterId - 1] + hSimplexOffsets[curStream][i];
                            int localVertexNum = hVertexIntervals_[sid] - hVertexIntervals_[sid - 1];
                            std::vector<SimplexId> vertexTriangleOffsets(localVertexNum + 1);
                            vertexTriangleOffsets[0] = 0;
                            for (int vid = 0; vid < localVertexNum; ++vid) {
                                vertexTriangleData.insert(vertexTriangleData.end(), 
                                hVertexTriangleMatrix_ + rvid * vfColumnNum,
                                hVertexTriangleMatrix_ + rvid * vfColumnNum + hVertexTriangleSizes_[rvid]);
                                vertexTriangleOffsets[vid+1] = vertexTriangleOffsets[vid] + hVertexTriangleSizes_[rvid];
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

                CHECK(cudaFree(dClustersIds[curStream]));
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
                        if (allClusters_[sid].vertexStars_.empty()) {
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
                            allClusters_[sid].vertexStars_.setData(std::move(vertexStarData), std::move(vertexStarOffsets));
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
                        if (allClusters_[sid].vertexNeighbors_.empty()) {
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
                            allClusters_[sid].vertexNeighbors_.setData(std::move(vertexNeighborData), std::move(vertexNeighborOffsets));
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
                    leaderCluster = currClusters_[i];
                    if (leaderCluster < 0) return;
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
    
    allClusters_ = std::vector<MeshSegment>(clusterNumber_);
    clusterMutexes_ = std::vector<boost::shared_mutex>(clusterNumber_);
    for (int cid = 1; cid < clusterNumber_; ++cid) {
        allClusters_[cid] = MeshSegment(cid);
    }

    producer_ = std::thread(&Gale::leaderProducer, this);
    printf("[Gale] GPU threads per cluster: %d, thread num: %d, block number: %d, block size: %d, cluster number: %d, cell number: %d, vertex number: %d\n", threadsPerCluster_, threadNumber_, blocksPerThread_, blockSize_, clusterNumber_, cellNumber_, vertexNumber_);

    // copy input mesh arrays to the GPU     
    CHECK(cudaMallocAsync(&dConnectivity_, sizeof(vtkIdType) * (cellNumber_ * 4), 0));
    CHECK(cudaMemcpyAsync(dConnectivity_, hConnectivity_, sizeof(vtkIdType) * (cellNumber_ * 4), cudaMemcpyHostToDevice));
    CHECK(cudaMallocAsync(&dOffsets_, sizeof(vtkIdType) * (cellNumber_+1), 0));
    CHECK(cudaMemcpyAsync(dOffsets_, hOffsets_, sizeof(vtkIdType) * (cellNumber_+1), cudaMemcpyHostToDevice));
    CHECK(cudaMallocAsync(&dIndexArray_, sizeof(int) * vertexNumber_, 0));
    CHECK(cudaMemcpyAsync(dIndexArray_, hIndexArray_, sizeof(int) * vertexNumber_, cudaMemcpyHostToDevice));
}

/**
 * Compute the cell interval array.
*/
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

    CHECK(cudaMemcpyAsync(dVertexIntervals_, hVertexIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice, 0));
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
