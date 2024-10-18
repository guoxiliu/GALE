#pragma once

#include <AbstractTriangulation.h>
#include <FlatJaggedArray.h>
#include <Timer.h>
#include <vtkPointSet.h>
#include <atomic>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <omp.h>
#include <unordered_map>
#include <queue>
#include <set>
#include <thread>
#include <semaphore.h>

/* Uncomment to track waiting time of consumer threads*/
#define MEASURE_WAITING 1

// Find the global edge/triangle index from the edge/triangle list
// (x, e): x specifies the segment index, e specifies the edge pair
#define findEdgeIdx(x, e) (std::lower_bound(edgeList_.begin()+hEdgeIntervals_[x-1], edgeList_.begin()+hEdgeIntervals_[x], e) - edgeList_.begin())
#define findTriangleIdx(x, f) (std::lower_bound(triangleList_.begin()+hTriangleIntervals_[x-1], triangleList_.begin()+hTriangleIntervals_[x], f) - triangleList_.begin())

// Check if CUDA function calls are correct
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
// the mesh segment class
class MeshSegment {
    private:
    /* components */
    SimplexId nid;
    /* boundary cells */
    std::vector<bool> boundaryVertices_;
    std::vector<bool> boundaryEdges_;
    /* vertex relations */
    FlatJaggedArray vertexEdges_;
    FlatJaggedArray vertexLinks_;
    FlatJaggedArray vertexNeighbors_;
    FlatJaggedArray vertexStars_;
    FlatJaggedArray vertexTriangles_;
    /* edge relations */
    FlatJaggedArray edgeLinks_;
    FlatJaggedArray edgeStars_;
    FlatJaggedArray edgeTriangles_;
    /* triangle relations */
    FlatJaggedArray triangleLinks_;
    FlatJaggedArray triangleStars_;
    /* cell relations */
    FlatJaggedArray cellNeighbors_;

public:
    MeshSegment() {}
    // multiple thread numbers
    MeshSegment(SimplexId id) : nid(id) {
        /* boundary cells */
        boundaryVertices_ = std::vector<bool>();
        boundaryEdges_ = std::vector<bool>();
        /* vertex relations */
        vertexEdges_ = FlatJaggedArray();
        vertexLinks_ = FlatJaggedArray();
        vertexNeighbors_ = FlatJaggedArray();
        vertexStars_ = FlatJaggedArray();
        vertexTriangles_ = FlatJaggedArray();
        /* edge relations */
        edgeLinks_ = FlatJaggedArray();
        edgeStars_ = FlatJaggedArray();
        edgeTriangles_ = FlatJaggedArray();
        /* triangle relations */
        triangleLinks_ = FlatJaggedArray();
        triangleStars_ = FlatJaggedArray();
        /* cell relations */
        cellNeighbors_ = FlatJaggedArray();
    }
    ~MeshSegment() {}

    inline void clear() {
        // keep the edge and triangle lists
        /* boundary cells */
        boundaryVertices_ = std::vector<bool>{};
        boundaryEdges_ = std::vector<bool>{};
        /* vertex relations */
        vertexEdges_ = FlatJaggedArray{};
        vertexLinks_ = FlatJaggedArray{};
        vertexNeighbors_ = FlatJaggedArray{};
        vertexStars_ = FlatJaggedArray{};
        vertexTriangles_ = FlatJaggedArray{};
        /* edge relations */
        // edgeVertex relation can be extracted from internal edge list
        edgeLinks_ = FlatJaggedArray{};
        edgeStars_ = FlatJaggedArray{};
        edgeTriangles_ = FlatJaggedArray{};
        /* triangle relations */
        // triangleVertex relation can be extracted from internal triangle list
        triangleLinks_ = FlatJaggedArray{};
        triangleStars_ = FlatJaggedArray{};
        /* cell relations */
        cellNeighbors_ = FlatJaggedArray{};
    }

    friend class Gale;
};


class Gale final : public AbstractTriangulation {
    public: 
        friend class TestTopoRelations;
        int *hIndexArray_{};
        Gale(int tnum = 1, int tnumPerCluster = 1);
        ~Gale();
        void initialize(vtkDataSet *dataSet, int numBlocks = 100, int blockSize = 32, bool reorder = false);
        
        /**
         * Preconditioning relation functions.
         * These functions are called before the algorithm execution to
         * tell the toplogical data structure which relations will be used 
         * 
        */
        inline int preconditionBoundaryEdgesInternal() override {
            if(cellDimension_ == 2) {
                #ifdef TTK_ENABLE_OPENMP
                #pragma omp parallel for num_threads(threadNumber_)
                #endif
                for(SimplexId clusterId = 1; clusterId < clusterNumber_; ++clusterId) {
                    // buildBoundaryTopCellList(clusterId);
                }
            } else if(cellDimension_ == 3) {
                preconditionBoundaryTriangles();
                preconditionedRelations_.push_back(RelationType::BERelation);
            } else {
                // unsupported dimension
                printErr("Unsupported dimension for boundary precondition.");
                return -1;
            }
            this->printMsg("Preconditioned boundary edges");
            return 0;
        }

        inline int preconditionBoundaryTrianglesInternal() override {
            if(cellDimension_ == 2) {
                return 0;
            } else if(cellDimension_ == 3) {
                boundaryTriangles_ = std::vector<bool>(triangleNumber_, false);
                computeRelation(RelationType::BFRelation, 1, 0, false);
                // use the producer threads to compute boundary triangles
                // #ifdef TTK_ENABLE_OPENMP
                // #pragma omp parallel for num_threads(threadNumber_)
                // #endif
                // for(SimplexId clusterId = 1; clusterId < clusterNumber_; ++clusterId) {
                //     // buildBoundaryTopCellList(clusterId);
                // }
                // printMsg("Boundary triangles preconditioned in " + std::to_string(t.getElapsedTime()) + "s.");
            } else {
                // unsupported dimension
                printErr("Unsupported dimension for boundary precondition.");
                return -1;
            }
            return 0;
        }

        inline int preconditionBoundaryVerticesInternal() override {
            if(cellDimension_ == 2) {
                if(!hasPreconditionedBoundaryEdges_) {
                    preconditionBoundaryEdges();
                }
            } else if(cellDimension_ == 3) {
                if(!hasPreconditionedBoundaryTriangles_) {
                    preconditionBoundaryTriangles();
                }
            }
            preconditionedRelations_.push_back(RelationType::BVRelation);
            return 0;
        }

        int preconditionEdgesInternal() override;
        int preconditionEdgeStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            preconditionedRelations_.push_back(RelationType::ETRelation);
            return 0;
        }

        int preconditionEdgeTrianglesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            preconditionedRelations_.push_back(RelationType::EFRelation);
            return 0;
        }

        int preconditionTrianglesInternal() override;
        int preconditionTriangleStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            preconditionedRelations_.push_back(RelationType::FTRelation);
            return 0;
        }

        inline int preconditionVertexEdgesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            preconditionedRelations_.push_back(RelationType::VERelation);
            return 0;
        }

        inline int preconditionVertexNeighborsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            preconditionedRelations_.push_back(RelationType::VVRelation);
            return 0;
        }

        inline int preconditionVertexStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            preconditionedRelations_.push_back(RelationType::VTRelation);
            return 0;
        }

        inline int preconditionVertexTrianglesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            preconditionedRelations_.push_back(RelationType::VFRelation);
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
            
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[edgeList_[edgeId][0]];
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].edgeStars_.empty()) {
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::ETRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localEdgeId = edgeId - hEdgeIntervals_[segId - 1];
            starId = allClusters_[segId].edgeStars_.get(localEdgeId, localStarId);
            return 0;
        }
        
        inline SimplexId getEdgeStarNumber(
            const SimplexId &edgeId) const override {
            
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[edgeList_[edgeId][0]];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::ETRelation;

            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].edgeStars_.empty()) {
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localEdgeId = edgeId - hEdgeIntervals_[segId - 1];
            return allClusters_[segId].edgeStars_.size(localEdgeId);
        }
        
        inline const std::vector<std::vector<SimplexId>> *
            getEdgeStars() override {

            computeRelation(RelationType::ETRelation, 1, true);
            return &edgeStarList_;
        }
        
        inline int getEdgeTriangleInternal(const SimplexId &edgeId,
            const int &localTriangleId, SimplexId &triangleId) const override {
        
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[edgeList_[edgeId][0]];
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].edgeTriangles_.empty()) {
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = hIndexArray_[edgeList_[edgeId][0]], currRelations_[tid] = RelationType::EFRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localEdgeId = edgeId - hEdgeIntervals_[segId - 1];
            triangleId = allClusters_[segId].edgeTriangles_.get(localEdgeId, localTriangleId);
            return 0;
        }
        
        inline SimplexId getEdgeTriangleNumberInternal(
            const SimplexId &edgeId) const override {

            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[edgeList_[edgeId][0]];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::EFRelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].edgeTriangles_.empty()) {
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localEdgeId = edgeId - hEdgeIntervals_[segId - 1];
            return allClusters_[segId].edgeTriangles_.size(localEdgeId);
        }
        
        inline const std::vector<std::vector<SimplexId>> *
            getEdgeTrianglesInternal() override {
            
            computeRelation(RelationType::EFRelation, 1, true);
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

        inline int getTriangleStar(const SimplexId &triangleId,
            const int &localStarId, SimplexId &starId) const override {

            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[triangleList_[triangleId][0]];
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].triangleStars_.empty()) {
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::FTRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localTriangleId = triangleId - hTriangleIntervals_[segId - 1];
            starId = allClusters_[segId].triangleStars_.get(localTriangleId, localStarId);
            return 0;
        }

        inline SimplexId getTriangleStarNumber(
            const SimplexId &triangleId) const override {

            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[triangleList_[triangleId][0]];
            currClusters_[tid] = hIndexArray_[triangleList_[triangleId][0]], currRelations_[tid] = RelationType::FTRelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].triangleStars_.empty()) {
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localTriangleId = triangleId - hTriangleIntervals_[segId - 1];
            return allClusters_[segId].triangleStars_.size(localTriangleId);
        }

        inline const std::vector<std::vector<SimplexId>> *
            getTriangleStars() override {
            
            computeRelation(RelationType::FTRelation, 1, true);
            return &triangleStarList_;
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

            SimplexId segId = hIndexArray_[vertexId];
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexNeighbors_.empty()) {
                readLock.unlock();
                int tid = omp_get_thread_num();
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::VVRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            neighborId = allClusters_[segId].vertexNeighbors_.get(localVertexId, localNeighborId);
            return 0;
        }

        inline SimplexId getVertexNeighborNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[vertexId];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::VVRelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexNeighbors_.empty()) {
                readLock.unlock();
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;

                // copy the data from GPU to CPU
                SimplexId endClusterId = std::min(segId + precompClusterNumber_, clusterNumber_) - 1;
                int curStream = relationStreamMap_.at(RelationType::VVRelation), pos = gpuThreadPositions_[tid];
                for (SimplexId sid = segId; sid <= endClusterId; ++sid) {
                    if (allClusters_[sid].vertexNeighbors_.empty()) {
                        std::vector<SimplexId> vertexNeighborData{};
                        int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[segId - 1] + relationCompVecs_[curStream].hSimplexOffsets[pos];
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
                        allClusters_[sid].vertexNeighbors_.setData(std::move(vertexNeighborData), std::move(vertexNeighborOffsets));
                    }
                }
                consumerCounters_[curStream].fetch_sub(1);
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            return allClusters_[segId].vertexNeighbors_.size(localVertexId);
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexNeighbors() override {
            // vertexNeighborData_.copyTo(vertexNeighborList_);
            computeRelation(RelationType::VVRelation, 1, true);
            return &vertexNeighborList_;
        }

        inline int getVertexEdge(const SimplexId &vertexId,
            const int &localEdgeId, SimplexId &edgeId) const override {

            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[vertexId];
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexEdges_.empty()) {
                readLock.unlock();
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::VERelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            edgeId = allClusters_[segId].vertexEdges_.get(localVertexId, localEdgeId);
            return 0;
        }

        inline SimplexId getVertexEdgeNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[vertexId];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::VERelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexEdges_.empty()) {
                // waitings_[tid] = 1;
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;

                // copy the data from GPU to CPU
                SimplexId endClusterId = std::min(segId + precompClusterNumber_, clusterNumber_) - 1;
                int curStream = relationStreamMap_.at(RelationType::VERelation), pos = gpuThreadPositions_[tid];
                for (SimplexId sid = segId; sid <= endClusterId; ++sid) {
                    if (allClusters_[sid].vertexEdges_.empty()) {
                        std::vector<SimplexId> vertexEdgeData{};
                        int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[segId - 1] + relationCompVecs_[curStream].hSimplexOffsets[pos];
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
                consumerCounters_[curStream].fetch_sub(1);
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            return allClusters_[segId].vertexEdges_.size(localVertexId);
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexEdges() override {
            
            computeRelation(RelationType::VERelation, 1, true);
            return &vertexEdgeList_;
        }

        inline int getVertexStar(const SimplexId &vertexId,
            const int &localStarId, SimplexId &starId) const override {
            
            SimplexId segId = hIndexArray_[vertexId];
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexStars_.empty()) {
                int tid = omp_get_thread_num();
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::VTRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            starId = allClusters_[segId].vertexStars_.get(localVertexId, localStarId);
            return 0;
        }

        inline SimplexId getVertexStarNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[vertexId];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::VTRelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexStars_.empty()) {
                // waitings_[tid] = 1;
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;

                // copy the data from GPU to CPU
                SimplexId endClusterId = std::min(segId + precompClusterNumber_, clusterNumber_) - 1;
                int curStream = relationStreamMap_.at(RelationType::VTRelation), pos = gpuThreadPositions_[tid];
                for (SimplexId sid = segId; sid <= endClusterId; ++sid) {
                    if (allClusters_[sid].vertexStars_.empty()) {
                        std::vector<SimplexId> vertexStarData{};
                        int diff = hVertexIntervals_[sid - 1] - hVertexIntervals_[segId - 1] + relationCompVecs_[curStream].hSimplexOffsets[pos];
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
                        allClusters_[sid].vertexStars_.setData(std::move(vertexStarData), std::move(vertexStarOffsets));
                    }
                }
                consumerCounters_[curStream].fetch_sub(1);
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            return allClusters_[segId].vertexStars_.size(localVertexId);
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexStars() override {
            
            computeRelation(RelationType::VTRelation, 1, true);
            return &vertexStarList_;
        }

        inline int getVertexTriangle(const SimplexId &vertexId,
            const int &localTriangleId, SimplexId &triangleId) const override {

            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[vertexId];
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexTriangles_.empty()) {
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VFRelation;
                // waitings_[tid] = 1;
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            triangleId = allClusters_[segId].vertexTriangles_.get(localVertexId, localTriangleId);
            return 0;
        }

        inline SimplexId getVertexTriangleNumber(
            const SimplexId &vertexId) const override {
            
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[vertexId];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::VFRelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].vertexTriangles_.empty()) {
                // waitings_[tid] = 1;
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            return allClusters_[segId].vertexTriangles_.size(localVertexId);
        }

        inline const std::vector<std::vector<SimplexId>> *
            getVertexTriangles() override {
            // vertexNeighborData_.copyTo(vertexNeighborList_);
            computeRelation(RelationType::VFRelation, 1, true);
            return &vertexTriangleList_;
        }


        inline bool isEdgeOnBoundary(const SimplexId &edgeId) const override {
            #ifndef TTK_ENABLE_KAMIKAZE
                if((edgeId < 0) || (edgeId >= edgeNumber_))
                    return false;
            #endif

            // if (cellDimension_ == 2) {
            //     return boundaryEdges_[edgeId];
            // }

            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[edgeList_[edgeId][0]];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::BERelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].boundaryEdges_.empty()) {
                // waitings_[tid] = 1;
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localEdgeId = edgeId - hEdgeIntervals_[segId - 1];
            return allClusters_[segId].boundaryEdges_.at(localEdgeId);
        }

        inline bool isEmpty() const override {
            return !vertexNumber_;
        }

        inline bool isTriangleOnBoundary(const SimplexId &triangleId) const override {
            if(cellDimension_ == 2)
                return false;

            #ifndef TTK_ENABLE_KAMIKAZE
                if((triangleId < 0) || (triangleId >= triangleNumber_))
                    return false;
            #endif

            return boundaryTriangles_[triangleId];
        }

        inline bool isVertexOnBoundary(const SimplexId &vertexId) const override {
            #ifndef TTK_ENABLE_KAMIKAZE
                if((vertexId < 0) || (vertexId >= vertexNumber_))
                    return false;
            #endif

            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[vertexId];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::BVRelation;
            boost::shared_lock<boost::shared_mutex> readLock(clusterMutexes_[segId]);
            if (allClusters_[segId].boundaryVertices_.empty()) {
                // waitings_[tid] = 1;
                #ifdef MEASURE_WAITING
                timers_[tid].reStart();
                #endif
                std::unique_lock<std::mutex> qlock(qMutex_);
                jobQueue.push(tid);
                qlock.unlock();
                #ifdef MEASURE_WAITING
                consumerTimes_[tid].push_back(timers_[tid].getElapsedTime());
                sem_wait(&relationSems_[tid]);
                // waitingCounts_[tid][currRelations_[tid]]++;
                // waitingTimes_[tid][currRelations_[tid]] += timers_[tid].getElapsedTime();
                #else
                sem_wait(&relationSems_[tid]);
                #endif
                // waitings_[tid] = 0;
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            return allClusters_[segId].boundaryVertices_.at(localVertexId);
        }


    private:
    
        /* Build the internal edge list in the cluster. */
        int buildInternalEdgeList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 2>>> &clusterEdges);

        /* Build the internal triangle list in the cluster. */
        int buildInternalTriangleList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 3>>> &clusterTriangles);

        /* Compute the interval array for cells */
        void computeCellIntervals();

        /* Compute the interval array for vertices */
        void computeVertexIntervals();

        /* Change the vertex id based on the cluster index */
        int reorderVertices(std::vector<SimplexId> &vertexMap);

        /* Check if the cell array is ordered in a specific way */
        int checkCells();

        /* Change the vertex id in the cell array */
        int reorderCells(const std::vector<SimplexId> &vertexMap);

        /* Function used by the leader producer thread */
        void leaderProducer();

        /* Compute the array for the given topological relation */
        int computeRelation(const RelationType relation, const SimplexId clusterId, bool global = false, int threadId = 0, bool signal = false);

        /* Compute the relations for different starting clusters */
        int computeCombinedRelation(std::map<RelationType, std::vector<std::pair<int,int>>> &requestMap, std::vector<RelationType> &relationOrders);

        bool doublePrecision_;
        int cellDimension_;
        int clusterNumber_;
        int precompClusterNumber_;
        int vertexNumber_, edgeNumber_, triangleNumber_, cellNumber_;
        std::vector<RelationType> preconditionedRelations_{};
        std::vector<cudaStream_t> threadStreams_;
        mutable std::vector<MeshSegment> allClusters_;

        /* Data allocated on the host memory */
        // pointers will be initialized to nullptr directly
        void *pointSet_;
        int blocksPerThread_, blockSize_;
        int *hExternalCells_{}, *hExternalCellOffsets_{};
        int *hConnectivity_{}, *hOffsets_{};
        int *hCellIntervals_{}, *hTriangleIntervals_{}, *hEdgeIntervals_{}, *hVertexIntervals_{};

        /* Data allocated on the device memory */
        int *dConnectivity_{}, *dOffsets_{};
        int *dIndexArray_{};
        int *dCellIntervals_{}, *dTriangleIntervals_{}, *dEdgeIntervals_{}, *dVertexIntervals_{};
        int *dExternalCells_{}, *dExternalCellOffsets_{};
        int *dEdgeList_{}, *dTriangleList_{};

        /* Relation matrices on the host and device */
        bool *hBoundaryTriangles_, *dBoundaryTriangles_;

        struct RelationComputation {
            int *hSizeArray, *hDataArray;
            int *dSizeArray, *dDataArray;
            int *dClusterIds, *hSimplexOffsets, *dSimplexOffsets;
        };
        mutable std::vector<RelationComputation> relationCompVecs_;

        /* Multithreading support */
        std::thread producer_;
        int threadsPerCluster_;
        std::array<std::vector<int>, 25> baseClusters_;
        std::vector<int> gpuThreadPositions_;
        std::unordered_map<RelationType, int> relationStreamMap_;
        mutable std::vector<std::atomic<int>> consumerCounters_;
        mutable std::vector<boost::shared_mutex> clusterMutexes_;
        mutable std::vector<SimplexId> currClusters_;
        mutable std::vector<RelationType> currRelations_;
        mutable std::vector<sem_t> relationSems_;
        mutable std::mutex qMutex_;
        mutable std::queue<int> jobQueue;
        #ifdef MEASURE_WAITING
        mutable std::vector<Timer> timers_;
        mutable std::vector<std::vector<double>> consumerTimes_;
        // mutable std::vector<double> waitingTimes_;
        // mutable std::vector<int> waitingCounts_;
        mutable std::vector<std::unordered_map<RelationType, int>> waitingCounts_;
        mutable std::vector<std::unordered_map<RelationType, double>> waitingTimes_;
        std::vector<int> qSizes_;
        std::vector<float> kernelTimes_;
        std::vector<float> returnTimes_;
        std::vector<double> qTimes_;
        #endif
    };
}
