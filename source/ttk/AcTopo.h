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

#define OUTSCOPE(cluster) (cluster > baseCluster_ + numBlocks_ * blockSize_ || cluster < baseCluster_)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

namespace ttk {

class AcTopo final : public AbstractTriangulation {
    public: 
        friend class TestTopoRelations;
        AcTopo(int tnum = 1, int tnumPerCluster = 1);
        ~AcTopo();
        void initialize(vtkPointSet *pointSet, int numBlocks = 100, int blockSize = 32);

        inline int TTK_TRIANGULATION_INTERNAL(getCellVertex)(
            const SimplexId &cellId,
            const int &localVertexId,
            SimplexId &vertexId) const override {
            vertexId = hConnectivity_[hOffsets_[cellId] + localVertexId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getCellVertexNumber)(
            const SimplexId &cellId) const override {
            return hOffsets_[cellId + 1] - hOffsets_[cellId];
        }


        int TTK_TRIANGULATION_INTERNAL(getDimensionality)() const override {
            return cellDimension_;
        }
        
        // int preconditionBoundaryEdgesInternal() override;
        // int preconditionBoundaryTrianglesInternal() override;
        // int preconditionBoundaryVerticesInternal() override;

        // int preconditionCellEdgesInternal() override;
        // int preconditionCellNeighborsInternal() override;
        // int preconditionCellTrianglesInternal() override;

        int preconditionEdgesInternal() override;
        // int preconditionEdgeLinksInternal() override;
        // int preconditionEdgeStarsInternal() override;
        // int preconditionEdgeTrianglesInternal() override;

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

        inline SimplexId
            TTK_TRIANGULATION_INTERNAL(getNumberOfCells)() const override {
                return cellNumber_;
        }

        inline SimplexId
            TTK_TRIANGULATION_INTERNAL(getNumberOfEdges)() const override {
                return edgeNumber_;
        }

        inline SimplexId
            TTK_TRIANGULATION_INTERNAL(getNumberOfTriangles)() const override {
                return triangleNumber_;
        }

        inline SimplexId
            TTK_TRIANGULATION_INTERNAL(getNumberOfVertices)() const override {
                return vertexNumber_;
        }

        inline int TTK_TRIANGULATION_INTERNAL(getVertexNeighbor)(
            const SimplexId &vertexId,
            const int &localNeighborId,
            SimplexId &neighborId) const override {

            if (vertexNeighborList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VVRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VVRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            neighborId = vertexNeighborList_[vertexId][localNeighborId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexNeighborNumber)(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VVRelation;
            if (vertexNeighborList_[vertexId].empty()) {
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VVRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            return vertexNeighborList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            TTK_TRIANGULATION_INTERNAL(getVertexNeighbors)() override {
            // vertexNeighborData_.copyTo(vertexNeighborList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexNeighborList_;
        }

        inline int TTK_TRIANGULATION_INTERNAL(getVertexEdge)(
            const SimplexId &vertexId,
            const int &localEdgeId,
            SimplexId &edgeId) const override {

            if (vertexEdgeList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VERelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VERelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            edgeId = vertexEdgeList_[vertexId][localEdgeId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexEdgeNumber)(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VERelation;
            if (vertexEdgeList_[vertexId].empty()) {
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VERelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            return vertexEdgeList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            TTK_TRIANGULATION_INTERNAL(getVertexEdges)() override {
            // vertexNeighborData_.copyTo(vertexNeighborList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexEdgeList_;
        }

        inline int TTK_TRIANGULATION_INTERNAL(getVertexStar)(
            const SimplexId &vertexId,
            const int &localStarId,
            SimplexId &starId) const override {
            
            if (vertexStarList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VTRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VTRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VTMutex_);
            starId = vertexStarList_[vertexId][localStarId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexStarNumber)(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VTRelation;
            if (vertexStarList_[vertexId].empty()) {
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VTRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VTMutex_);
            return vertexStarList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            TTK_TRIANGULATION_INTERNAL(getVertexStars)() override {
            // vertexStarData_.copyTo(vertexStarList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexStarList_;
        }

        inline int TTK_TRIANGULATION_INTERNAL(getVertexTriangle)(
            const SimplexId &vertexId,
            const int &localTriangleId,
            SimplexId &triangleId) const override {

            if (vertexTriangleList_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VFRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VFRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            triangleId = vertexTriangleList_[vertexId][localTriangleId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexTriangleNumber)(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VFRelation;
            if (vertexTriangleList_[vertexId].empty()) {
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VFRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            return vertexTriangleList_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            TTK_TRIANGULATION_INTERNAL(getVertexTriangles)() override {
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
