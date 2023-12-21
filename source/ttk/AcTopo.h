#pragma once

#include "AbstractTriangulation.h"
#include "Timer.h"
#include <vtkPointSet.h>
#include <atomic>
#include <mutex>
#include <queue>
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
        AcTopo(int tnum = 1);
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

        // int preconditionEdgesInternal() override;
        // int preconditionEdgeLinksInternal() override;
        // int preconditionEdgeStarsInternal() override;
        // int preconditionEdgeTrianglesInternal() override;

        // int preconditionTrianglesInternal() override;
        // int preconditionTriangleEdgesInternal() override;
        // int preconditionTriangleLinksInternal() override;
        // int preconditionTriangleStarsInternal() override;

        // int preconditionVertexEdgesInternal() override;
        // int preconditionVertexLinksInternal() override;
        int preconditionVertexNeighborsInternal() override;
        int preconditionVertexStarsInternal() override;
        // int preconditionVertexTrianglesInternal() override;

        inline SimplexId
            TTK_TRIANGULATION_INTERNAL(getNumberOfCells)() const override {
                return cellNumber_;
        }

        inline SimplexId
            TTK_TRIANGULATION_INTERNAL(getNumberOfVertices)() const override {
                return vertexNumber_;
        }

        inline int TTK_TRIANGULATION_INTERNAL(getVertexNeighbor)(
            const SimplexId &vertexId,
            const int &localNeighborId,
            SimplexId &neighborId) const override {

            if (vertexNeighbors_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VVRelation;
                // waitings_[tid] = 1;
                // std::unique_lock<std::mutex> qlock(qMutex_);
                // jobQueue.push(tid);
                // qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                computeRelation(currClusters_[tid], currRelations_[tid], tid, false);
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                // sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VVRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            neighborId = vertexNeighbors_[vertexId][localNeighborId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexNeighborNumber)(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VVRelation;
            if (vertexNeighbors_[vertexId].empty()) {
                // waitings_[tid] = 1;
                // std::unique_lock<std::mutex> qlock(qMutex_);
                // jobQueue.push(tid);
                // qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                computeRelation(currClusters_[tid], currRelations_[tid], tid, false);
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                // sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VVRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VVMutex_);
            return vertexNeighbors_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            TTK_TRIANGULATION_INTERNAL(getVertexNeighbors)() override {
            // vertexNeighborData_.copyTo(vertexNeighborList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexNeighborList_;
        }

        inline int TTK_TRIANGULATION_INTERNAL(getVertexStar)(
            const SimplexId &vertexId,
            const int &localStarId,
            SimplexId &starId) const override {
            
            if (vertexStars_[vertexId].empty()) {
                int tid = omp_get_thread_num();
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VTRelation;
                // waitings_[tid] = 1;
                // std::unique_lock<std::mutex> qlock(qMutex_);
                // jobQueue.push(tid);
                // qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                computeRelation(currClusters_[tid], currRelations_[tid], tid, false);
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                // sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VTRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VTMutex_);
            starId = vertexStars_[vertexId][localStarId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexStarNumber)(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VTRelation;
            if (vertexStars_[vertexId].empty()) {
                // waitings_[tid] = 1;
                // std::unique_lock<std::mutex> qlock(qMutex_);
                // jobQueue.push(tid);
                // qlock.unlock();
                waitingCounts_[tid]++;
                timers_[tid].reStart();
                computeRelation(currClusters_[tid], currRelations_[tid], tid, false);
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                // sem_wait(&relationSems_[tid][static_cast<int>(RelationType::VTRelation)]);
                waitingTimes_[tid] += timers_[tid].getElapsedTime();
                // waitings_[tid] = 0;
            }
            // std::lock_guard<std::mutex> lck(VTMutex_);
            return vertexStars_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            TTK_TRIANGULATION_INTERNAL(getVertexStars)() override {
            // vertexStarData_.copyTo(vertexStarList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexStarList_;
        }


    private:
        /* Compute the interval array for cells */
        void computeCellIntervals();

        /* Compute the interval array for vertices */
        void computeVertexIntervals();

        /* Assisting function used by the producer thread */
        void assistProcedure();

        /* Compute the array for the given topological relation */
        int computeRelation(const SimplexId clusterId, const RelationType relation, int threadId, bool signal) const;

        int cellDimension_;
        int clusterNumber_;
        int vertexNumber_, edgeNumber_, triangleNumber_, cellNumber_;

        /* data allocated on host memory */
        volatile int numBlocks_, blockSize_;
        int *hIndexArray_, *hExternalCells_;
        vtkIdType *hConnectivity_, *hOffsets_;
        int *hCellIntervals_, *hTriangleIntervals_, *hEdgeIntervals_, *hVertexIntervals_;

        /* data allocated on device memory */
        vtkIdType *dConnectivity_, *dOffsets_;
        int *dIndexArray_;
        int *dCellIntervals_, *dTriangleIntervals_, *dEdgeIntervals_, *dVertexIntervals_;
        int *dExternalCells_, *dExternalCellOffsets_;

        /* relation matrices */
        mutable std::vector<std::vector<int>> vertexNeighbors_;
        mutable std::vector<std::vector<int>> vertexStars_;
        
        /* multithreading support */
        std::thread producer_;
        int threadNumber_;
        mutable std::array<std::vector<int>, 25> baseClusters_;
        mutable std::vector<SimplexId> currClusters_;
        mutable std::vector<RelationType> currRelations_;
        mutable std::mutex VVMutex_, VTMutex_;
        mutable std::vector<std::vector<sem_t>> relationSems_;
        mutable std::vector<Timer> timers_;
        // mutable std::vector<int> waitings_;
        mutable std::vector<double> waitingTimes_;
        mutable std::vector<int> waitingCounts_;
        mutable std::mutex qMutex_, ioMutex_;
        mutable std::queue<int> jobQueue;
    };
}
