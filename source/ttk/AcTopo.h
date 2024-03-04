#pragma once

#include "AbstractTriangulation.h"
#include "Timer.h"
#include <vtkPointSet.h>
#include <atomic>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <omp.h>
#include <semaphore.h>

// Find the global edge/triangle index from the edge/triangle list
#define findEdgeIdx(x, e) (std::lower_bound(edgeList_.begin()+hEdgeIntervals_[x-1], edgeList_.begin()+hEdgeIntervals_[x], e) - edgeList_.begin())
#define findTriangleIdx(x, f) (std::lower_bound(triangleList_.begin()+hTriangleIntervals_[x-1], triangleList_.begin()+hTriangleIntervals_[x], f) - triangleList_.begin())

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// const int removalRatio = 0.5;

namespace ttk {

class AcTopo final : public AbstractTriangulation {
    public: 
        friend class TestTopoRelations;
        AcTopo(int tnum = 1, int tnumPerCluster = 1);
        ~AcTopo();
        void initialize(vtkPointSet *pointSet, int numBlocks = 100, int blockSize = 32);
        
        // int preconditionBoundaryEdgesInternal() override;
        // int preconditionBoundaryTrianglesInternal() override;
        // int preconditionBoundaryVerticesInternal() override;

        // int preconditionCellEdgesInternal() override;
        // int preconditionCellNeighborsInternal() override;
        // int preconditionCellTrianglesInternal() override;

        int preconditionEdgesInternal() override;
        // int preconditionEdgeLinksInternal() override;
        int preconditionEdgeStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            edgeStarList_ = std::vector<std::vector<int>>(edgeNumber_, std::vector<int>{});
            return 0;
        }

        int preconditionEdgeTrianglesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            edgeTriangleList_ = std::vector<std::vector<int>>(edgeNumber_, std::vector<int>{});
            return 0;
        }

        int preconditionTrianglesInternal() override;
        // int preconditionTriangleEdgesInternal() override;
        // int preconditionTriangleLinksInternal() override;
        // int preconditionTriangleStarsInternal() override;

        inline int preconditionVertexEdgesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            vertexEdgeList_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});
            return 0;
        }
        // int preconditionVertexLinksInternal() override;

        inline int preconditionVertexNeighborsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            vertexNeighborList_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});
            return 0;
        }

        inline int preconditionVertexStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            vertexStarList_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});
            return 0;
        }

        inline int preconditionVertexTrianglesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            vertexTriangleList_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});
            return 0;
        }

        inline SimplexId getNumberOfCells() const override {
                return cellNumber_;
        }

        inline SimplexId getNumberOfEdges() const override {
                return edgeNumber_;
        }

        inline SimplexId getNumberOfTriangles() const override {
                return triangleNumber_;
        }

        inline SimplexId getNumberOfVertices() const override {
                return vertexNumber_;
        }

        inline int getCellEdgeInternal(const SimplexId &cellId,
            const int &localEdgeId, SimplexId &edgeId) const override {
            
            std::array<int, 2> edgeIds;
            switch (localEdgeId) {
                case 0:     // (v0, v1) 
                {
                    edgeIds[0] = hConnectivity_[hOffsets_[cellId]];
                    edgeIds[1] = hConnectivity_[hOffsets_[cellId] + 1];
                    SimplexId clusterId = hIndexArray_[edgeIds[0]];
                    edgeId = findEdgeIdx(clusterId, edgeIds);
                    break;
                }
                case 1:     // (v0, v2) 
                {
                    edgeIds[0] = hConnectivity_[hOffsets_[cellId]];
                    edgeIds[1] = hConnectivity_[hOffsets_[cellId] + 2];
                    SimplexId clusterId = hIndexArray_[edgeIds[0]];
                    edgeId = findEdgeIdx(clusterId, edgeIds);
                    break;
                }
                case 2:     // (v0, v3) 
                {
                    edgeIds[0] = hConnectivity_[hOffsets_[cellId]];
                    edgeIds[1] = hConnectivity_[hOffsets_[cellId] + 3];
                    SimplexId clusterId = hIndexArray_[edgeIds[0]];
                    edgeId = findEdgeIdx(clusterId, edgeIds);
                    break;
                }
                case 3:     // (v1, v2) 
                {
                    edgeIds[0] = hConnectivity_[hOffsets_[cellId] + 1];
                    edgeIds[1] = hConnectivity_[hOffsets_[cellId] + 2];
                    SimplexId clusterId = hIndexArray_[edgeIds[0]];
                    edgeId = findEdgeIdx(clusterId, edgeIds);
                    break;
                }
                case 4:     // (v1, v3) 
                {
                    edgeIds[0] = hConnectivity_[hOffsets_[cellId] + 1];
                    edgeIds[1] = hConnectivity_[hOffsets_[cellId] + 3];
                    SimplexId clusterId = hIndexArray_[edgeIds[0]];
                    edgeId = findEdgeIdx(clusterId, edgeIds);
                    break;
                }
                case 5:     // (v2, v3) 
                {
                    edgeIds[0] = hConnectivity_[hOffsets_[cellId] + 2];
                    edgeIds[1] = hConnectivity_[hOffsets_[cellId] + 3];
                    SimplexId clusterId = hIndexArray_[edgeIds[0]];
                    edgeId = findEdgeIdx(clusterId, edgeIds);
                    break;
                }
                default:
                    return -1;
            }
            return 0;
        }

        inline SimplexId getCellEdgeNumberInternal(
            const SimplexId &cellId) const override {
            
            (void) cellId;
            return 6;
        }

        template <std::size_t N>
        inline void convertToVector(
            const std::vector<std::array<SimplexId, N>> &table,
            std::vector<std::vector<SimplexId>> &vec) {
            vec.resize(table.size());
            for(size_t i = 0; i < table.size(); ++i) {
                vec[i] = {table[i].begin(), table[i].end()};
            }
        }

        inline const std::vector<std::vector<SimplexId>> *
            getCellEdgesInternal() override {
            
            convertToVector(tetraEdgeList_, cellEdgeVector_);
            return &cellEdgeVector_;
        }

        inline int getCellTriangleInternal(const SimplexId &cellId,
            const int &localTriangleId, SimplexId &triangleId) const override {
            
            std::array<int, 3> triangleIds;
            switch (localTriangleId) {
                case 0:     // (v0, v1, v2) 
                {
                    triangleIds[0] = hConnectivity_[hOffsets_[cellId]];
                    triangleIds[1] = hConnectivity_[hOffsets_[cellId] + 1];
                    triangleIds[2] = hConnectivity_[hOffsets_[cellId] + 2];
                    SimplexId clusterId = hIndexArray_[triangleIds[0]];
                    triangleId = findTriangleIdx(clusterId, triangleIds);
                    break;
                }
                case 1:     // (v0, v1, v3) 
                {
                    triangleIds[0] = hConnectivity_[hOffsets_[cellId]];
                    triangleIds[1] = hConnectivity_[hOffsets_[cellId] + 1];
                    triangleIds[2] = hConnectivity_[hOffsets_[cellId] + 3];
                    SimplexId clusterId = hIndexArray_[triangleIds[0]];
                    triangleId = findTriangleIdx(clusterId, triangleIds);
                    break;
                }
                case 2:     // (v0, v2, v3) 
                {
                    triangleIds[0] = hConnectivity_[hOffsets_[cellId]];
                    triangleIds[1] = hConnectivity_[hOffsets_[cellId] + 2];
                    triangleIds[2] = hConnectivity_[hOffsets_[cellId] + 3];
                    SimplexId clusterId = hIndexArray_[triangleIds[0]];
                    triangleId = findTriangleIdx(clusterId, triangleIds);
                    break;
                }
                case 3:     // (v1, v2, v3) 
                {
                    triangleIds[0] = hConnectivity_[hOffsets_[cellId] + 1];
                    triangleIds[1] = hConnectivity_[hOffsets_[cellId] + 2];
                    triangleIds[2] = hConnectivity_[hOffsets_[cellId] + 3];
                    SimplexId clusterId = hIndexArray_[triangleIds[0]];
                    triangleId = findTriangleIdx(clusterId, triangleIds);
                    break;
                }
                default:
                    return -1;
            }
            return 0;
        }

        inline SimplexId getCellTriangleNumberInternal(
            const SimplexId &cellId) const override {
            
            return 4;
        }

        inline const std::vector<std::vector<SimplexId>> *
            getCellTrianglesInternal() override {
            
            convertToVector(tetraTriangleList_, cellTriangleVector_);
            return &cellTriangleVector_;
        }

        inline int getCellVertex(const SimplexId &cellId,
            const int &localVertexId, SimplexId &vertexId) const override {
            
            vertexId = hConnectivity_[hOffsets_[cellId] + localVertexId];
            return 0;
        }

        inline SimplexId getCellVertexNumber(
            const SimplexId &cellId) const override {
            
            return hOffsets_[cellId + 1] - hOffsets_[cellId];
        }

        int getDimensionality() const override {
            return cellDimension_;
        }

        inline const std::vector<std::array<SimplexId, 2>> *
            getEdges() override {
            
            return &edgeList_;
        }
        
        inline int getEdgeStar(const SimplexId &edgeId,
            const int &localStarId, SimplexId &starId) const override {
            
            if (edgeStarList_[edgeId].empty()) {
                int tid = omp_get_thread_num();
                timers_[tid].reStart();
                currClusters_[tid] = hIndexArray_[edgeList_[edgeId][0]], currRelations_[tid] = RelationType::ETRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::ETRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            starId = edgeStarList_[edgeId][localStarId];
            return 0;
        }
        
        inline SimplexId getEdgeStarNumber(
            const SimplexId &edgeId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[edgeList_[edgeId][0]], currRelations_[tid] = RelationType::ETRelation;
            if (edgeStarList_[edgeId].empty()) {
                timers_[tid].reStart();
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::ETRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            return edgeStarList_[edgeId].size();
        }
        
        inline const std::vector<std::vector<SimplexId>> *
            getEdgeStars() override {
            
            return &edgeStarList_;
        }
        
        inline int getEdgeTriangleInternal(const SimplexId &edgeId,
            const int &localTriangleId, SimplexId &triangleId) const override {
        
            if (edgeTriangleList_[edgeId].empty()) {
                int tid = omp_get_thread_num();
                timers_[tid].reStart();
                currClusters_[tid] = hIndexArray_[edgeList_[edgeId][0]], currRelations_[tid] = RelationType::EFRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::EFRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            triangleId = edgeTriangleList_[edgeId][localTriangleId];
            return 0;
        }
        
        inline SimplexId getEdgeTriangleNumberInternal(
            const SimplexId &edgeId) const override {

            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[edgeList_[edgeId][0]], currRelations_[tid] = RelationType::EFRelation;
            if (edgeTriangleList_[edgeId].empty()) {
                timers_[tid].reStart();
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::EFRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            return edgeTriangleList_[edgeId].size();
        }
        
        inline const std::vector<std::vector<SimplexId>> *
            getEdgeTrianglesInternal() override {
                
            return &edgeTriangleList_;
        }

        inline int getEdgeVertexInternal(const SimplexId &edgeId,
            const int &localVertexId, SimplexId &vertexId) const override {
            
            if (localVertexId < 2) {
                vertexId = edgeList_[edgeId][localVertexId];
                return 0;
            }
            return -1;
        }
        
        inline int getTriangleEdgeInternal(const SimplexId &triangleId,
            const int &localEdgeId, SimplexId &edgeId) const override {
            
            std::array<SimplexId, 2> edgeIds;
            switch(localEdgeId) {
                case 0:     // (v0, v1)
                {
                    edgeIds[0] = triangleList_[triangleId][0];
                    edgeIds[1] = triangleList_[triangleId][1];
                    edgeId = findEdgeIdx(hIndexArray_[edgeIds[0]], edgeIds);
                    break;
                }
                case 1:     // (v0, v2)
                {
                    edgeIds[0] = triangleList_[triangleId][0];
                    edgeIds[1] = triangleList_[triangleId][2];
                    edgeId = findEdgeIdx(hIndexArray_[edgeIds[0]], edgeIds);
                    break;
                }
                case 2:     // (v1, v2)
                {
                    edgeIds[0] = triangleList_[triangleId][1];
                    edgeIds[1] = triangleList_[triangleId][2];
                    edgeId = findEdgeIdx(hIndexArray_[edgeIds[0]], edgeIds);
                    break;
                }
                default:
                    return -1;
            }
            return 0;
        }
        
        inline SimplexId getTriangleEdgeNumberInternal(
            const SimplexId &triangleId) const override {
            
            (void)triangleId;
            return 3;
        }
        
        inline const std::vector<std::vector<SimplexId>> *
            getTriangleEdgesInternal() override {
            
            if(triangleEdgeVector_.empty()) {

            }
            return &triangleEdgeVector_;
        }

        inline int getTriangleVertexInternal(const SimplexId &triangleId,
            const int &localVertexId, SimplexId &vertexId) const override {

            if (localVertexId < 3) {
                vertexId = triangleList_[triangleId][localVertexId];
                return 0;
            }
            return -1;
        }

        inline int getVertexNeighbor( const SimplexId &vertexId,
            const int &localNeighborId, SimplexId &neighborId) const override {

            if (vertexNeighborList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                timers_[tid].reStart();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VVRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VVRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            neighborId = vertexNeighborList_[vertexId][localNeighborId];
            return 0;
        }

        inline SimplexId getVertexNeighborNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VVRelation;
            if (vertexNeighborList_[vertexId].empty()) {
                timers_[tid].reStart();
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VVRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            return vertexNeighborList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexNeighbors() override {
            // vertexNeighborData_.copyTo(vertexNeighborList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexNeighborList_;
        }

        inline int getVertexEdge(const SimplexId &vertexId,
            const int &localEdgeId, SimplexId &edgeId) const override {

            if (vertexEdgeList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                timers_[tid].reStart();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VERelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VERelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            edgeId = vertexEdgeList_[vertexId][localEdgeId];
            return 0;
        }

        inline SimplexId getVertexEdgeNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VERelation;
            if (vertexEdgeList_[vertexId].empty()) {
                // waitings_[tid] = 1;
                timers_[tid].reStart();
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VERelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            return vertexEdgeList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexEdges() override {
            // TODO: copy the matrix data back to vector of vector
            return &vertexEdgeList_;
        }

        inline int getVertexStar(const SimplexId &vertexId,
            const int &localStarId, SimplexId &starId) const override {
            
            if (vertexStarList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                timers_[tid].reStart();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VTRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VTRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VTMutex_);
            starId = vertexStarList_[vertexId][localStarId];
            return 0;
        }

        inline SimplexId getVertexStarNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VTRelation;
            if (vertexStarList_[vertexId].empty()) {
                // waitings_[tid] = 1;
                timers_[tid].reStart();
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VTRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VTMutex_);
            return vertexStarList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexStars() override {
            // TODO: copy the matrix data back to vector of vector
            return &vertexStarList_;
        }

        inline int getVertexTriangle(const SimplexId &vertexId,
            const int &localTriangleId, SimplexId &triangleId) const override {

            if (vertexTriangleList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                timers_[tid].reStart();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VFRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VFRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            triangleId = vertexTriangleList_[vertexId][localTriangleId];
            return 0;
        }

        inline SimplexId getVertexTriangleNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VFRelation;
            if (vertexTriangleList_[vertexId].empty()) {
                // waitings_[tid] = 1;
                timers_[tid].reStart();
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VFRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            return vertexTriangleList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexTriangles() override {
            // vertexNeighborData_.copyTo(vertexNeighborList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexTriangleList_;
        }


    private:
    
        /* Build the internal edge list in the cluster. */
        int buildInternalEdgeList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 2>>> &clusterEdges, bool buildBoundary = false);

        /* Build the internal triangle list in the cluster. */
        int buildInternalTriangleList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 3>>> &clusterTriangles, bool buildBoundary = false);

        /* Compute the interval array for cells */
        void computeCellIntervals();

        /* Compute the interval array for vertices */
        void computeVertexIntervals();

        /* Assisting function used by the producer thread */
        void assistProcedure();

        /* Compute the array for the given topological relation */
        int computeRelation(const RelationType relation, const SimplexId clusterId, int threadId, bool signal);

        /* Compute the relations for different starting clusters */
        int computeCombinedRelation(const RelationType relation, const std::vector<std::pair<int, int>> requests);

        int cellDimension_;
        int clusterNumber_;
        int vertexNumber_, edgeNumber_, triangleNumber_, cellNumber_;

        // data allocated on host memory
        volatile int numBlocks_, blockSize_;
        int *hIndexArray_, *hExternalCells_, *hExternalCellOffsets_;
        vtkIdType *hConnectivity_, *hOffsets_;
        int *hCellIntervals_, *hTriangleIntervals_, *hEdgeIntervals_, *hVertexIntervals_;

        // data allocated on device memory
        vtkIdType *dConnectivity_, *dOffsets_;
        int *dIndexArray_;
        int *dCellIntervals_, *dTriangleIntervals_, *dEdgeIntervals_, *dVertexIntervals_;
        int *dExternalCells_, *dExternalCellOffsets_;

        // relation matrices
        int *hEdgeStarSizes_, *hEdgeStarMatrix_;
        int *dEdgeStarSizes_, *dEdgeStarMatrix_;
        
        int *hEdgeTriangleSizes_, *hEdgeTriangleMatrix_;
        int *dEdgeTriangleSizes_, *dEdgeTriangleMatrix_;
        
        int *hVertexEdgeSizes_, *hVertexEdgeMatrix_;
        int *dVertexEdgeSizes_, *dVertexEdgeMatrix_;

        int *hVertexNeighborSizes_, *hVertexNeighborMatrix_;
        int *dVertexNeighborSizes_, *dVertexNeighborMatrix_;
        
        int *hVertexStarSizes_, *hVertexStarMatrix_;
        int *dVertexStarSizes_, *dVertexStarMatrix_;

        int *hVertexTriangleSizes_, *hVertexTriangleMatrix_;
        int *dVertexTriangleSizes_, *dVertexTriangleMatrix_;

        // multithreading support
        std::thread producer_;
        int threadNumber_;
        int threadsPerCluster_;
        std::array<std::vector<int>, 25> baseClusters_;
        mutable std::vector<SimplexId> currClusters_;
        mutable std::vector<RelationType> currRelations_;
        mutable std::mutex EFMutex_, ETMutex_;
        mutable std::mutex VEMutex_, VVMutex_, VTMutex_;
        mutable std::vector<std::vector<sem_t>> relationSems_;
        mutable std::vector<Timer> timers_;
        // mutable std::vector<int> waitings_;
        mutable std::vector<double> waitingTimes_;
        mutable std::vector<int> waitingCounts_;
        mutable std::mutex qMutex_;
        mutable std::queue<int> jobQueue;
    };
}
