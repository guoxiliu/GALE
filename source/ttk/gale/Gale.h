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
#include <unordered_set>
#include <queue>
#include <set>
#include <thread>
#include <semaphore.h>

/* Uncomment to track detailed timing of consumer and producer threads*/
// #define MEASURE_TIME 1

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
        int *dVVDataArray_;
        int *dVVIndexArray_;
        std::vector<std::vector<SimplexId>> clusterVertices;
        Gale(int tnum = 1, int tnumPerCluster = 1, int SFCP_ColumnNum_=30);
        ~Gale();
        void initialize(vtkDataSet *dataSet, int numBlocks = 100, int blockSize = 32, bool reorder = false);
        void enableTracking() { startTracking_ = true; }
        // Extensions
        void computeVertexNeighborWithExternals(const SimplexId clusterId);

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
                int curId = preconditionedRelations_.size();
                relationMaps_[RelationType::BERelation] = curId;
                preconditionedRelations_.push_back(RelationType::BERelation);
                #ifdef MEASURE_TIME
                jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
                #else
                jobQueues_.push_back(std::queue<int>{});
                #endif
                producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
                // #ifdef MEASURE_TIME
                // qTimes_.push_back(std::vector<double>{});
                // qSizes_.push_back(std::vector<int>{});
                // kernelTimes_.push_back(std::vector<float>{});
                // returnTimes_.push_back(std::vector<float>{});
                // #endif
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
                computeRelation(RelationType::BFRelation, true, 1, 0, false);
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
            int curId = preconditionedRelations_.size();
            relationMaps_[RelationType::BVRelation] = curId;
            preconditionedRelations_.push_back(RelationType::BVRelation);
            #ifdef MEASURE_TIME
            jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
            #else
            jobQueues_.push_back(std::queue<int>{});
            #endif
            producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
            // #ifdef MEASURE_TIME
            // qTimes_.push_back(std::vector<double>{});
            // qSizes_.push_back(std::vector<int>{});
            // kernelTimes_.push_back(std::vector<float>{});
            // returnTimes_.push_back(std::vector<float>{});
            // #endif
            return 0;
        }

        int preconditionEdgesInternal() override;
        int preconditionEdgeStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            int curId = preconditionedRelations_.size();
            relationMaps_[RelationType::ETRelation] = curId;
            preconditionedRelations_.push_back(RelationType::ETRelation);
            #ifdef MEASURE_TIME
            jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
            #else
            jobQueues_.push_back(std::queue<int>{});
            #endif
            producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
            // #ifdef MEASURE_TIME
            // qTimes_.push_back(std::vector<double>{});
            // qSizes_.push_back(std::vector<int>{});
            // kernelTimes_.push_back(std::vector<float>{});
            // returnTimes_.push_back(std::vector<float>{});
            // #endif
            return 0;
        }

        int preconditionEdgeTrianglesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            int curId = preconditionedRelations_.size();
            relationMaps_[RelationType::EFRelation] = curId;
            preconditionedRelations_.push_back(RelationType::EFRelation);
            #ifdef MEASURE_TIME
            jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
            #else
            jobQueues_.push_back(std::queue<int>{});
            #endif
            producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
            // #ifdef MEASURE_TIME
            // qTimes_.push_back(std::vector<double>{});
            // qSizes_.push_back(std::vector<int>{});
            // kernelTimes_.push_back(std::vector<float>{});
            // returnTimes_.push_back(std::vector<float>{});
            // #endif
            return 0;
        }

        int preconditionTrianglesInternal() override;
        int preconditionTriangleStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            int curId = preconditionedRelations_.size();
            relationMaps_[RelationType::FTRelation] = curId;
            preconditionedRelations_.push_back(RelationType::FTRelation);
            #ifdef MEASURE_TIME
            jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
            #else
            jobQueues_.push_back(std::queue<int>{});
            #endif
            producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
            // #ifdef MEASURE_TIME
            // qTimes_.push_back(std::vector<double>{});
            // qSizes_.push_back(std::vector<int>{});
            // kernelTimes_.push_back(std::vector<float>{});
            // returnTimes_.push_back(std::vector<float>{});
            // #endif
            return 0;
        }

        inline int preconditionVertexEdgesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            int curId = preconditionedRelations_.size();
            relationMaps_[RelationType::VERelation] = curId;
            preconditionedRelations_.push_back(RelationType::VERelation);
            #ifdef MEASURE_TIME
            jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
            #else
            jobQueues_.push_back(std::queue<int>{});
            #endif
            producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
            // #ifdef MEASURE_TIME
            // qTimes_.push_back(std::vector<double>{});
            // qSizes_.push_back(std::vector<int>{});
            // kernelTimes_.push_back(std::vector<float>{});
            // returnTimes_.push_back(std::vector<float>{});
            // #endif
            return 0;
        }
        int preconditionVertexNeighborsInternal() override;

        inline int preconditionVertexStarsInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            int curId = preconditionedRelations_.size();
            relationMaps_[RelationType::VTRelation] = curId;
            preconditionedRelations_.push_back(RelationType::VTRelation);
            #ifdef MEASURE_TIME
            jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
            #else
            jobQueues_.push_back(std::queue<int>{});
            #endif
            producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
            // #ifdef MEASURE_TIME
            // qTimes_.push_back(std::vector<double>{});
            // qSizes_.push_back(std::vector<int>{});
            // kernelTimes_.push_back(std::vector<float>{});
            // returnTimes_.push_back(std::vector<float>{});
            // #endif
            return 0;
        }

        inline int preconditionVertexTrianglesInternal() override {
            if (this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
                std::cerr << "Empty dataset, precondition skipped";
                return 1;
            }
            int curId = preconditionedRelations_.size();
            relationMaps_[RelationType::VFRelation] = curId;
            preconditionedRelations_.push_back(RelationType::VFRelation);
            #ifdef MEASURE_TIME
            jobQueues_.push_back(std::queue<std::pair<int, Timer>>{});
            #else
            jobQueues_.push_back(std::queue<int>{});
            #endif
            producers_.push_back(std::thread(&Gale::leaderProducer, this, curId));
            // #ifdef MEASURE_TIME
            // qTimes_.push_back(std::vector<double>{});
            // qSizes_.push_back(std::vector<int>{});
            // kernelTimes_.push_back(std::vector<float>{});
            // returnTimes_.push_back(std::vector<float>{});
            // #endif
            return 0;
        }

        // Publicizing the internals
        inline int getNumberOfClusters() {
                return clusterNumber_;
        }
        inline int * getDConnectivity() {
                return dConnectivity_;
        }
        inline int * getDOffsets() {
                return dOffsets_;
        }
        inline int * getDVertexIntervals() {
                return dVertexIntervals_;
        }
        inline int * getDCellIntervals() {
                return dCellIntervals_;
        }
        inline int * getDExternalCells() {
                return dExternalCells_;
        }
        inline int * getDExternalCellOffsets() {
                return dExternalCellOffsets_;
        }
        inline int * getHVertexIntervals() {
                return hVertexIntervals_;
        }
        inline int * getHCellIntervals() {
                return hCellIntervals_;
        }
        inline int * getHExternalCells() {
                return hExternalCells_;
        }
        // End publicized internals
        // Additional functions
        void setPartitionAttributes(vtkDataArray *dataArray, vtkIdType nPoints);
        // End additional functions


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

        inline int getCellNeighborInternal(const SimplexId &cellId,
                              const int &localNeighborId,
                              SimplexId &neighborId) const override {
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[hConnectivity_[hOffsets_[cellId]]];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::TTRelation;
            if (allClusters_[segId].cellNeighbors_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                getClusterCellNeighbors(segId);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
            }
            SimplexId localCellId = cellId - hCellIntervals_[segId - 1];
            neighborId = allClusters_[segId].cellNeighbors_.get(localCellId, localNeighborId);
            return 0;
        }

        inline SimplexId getCellNeighborNumberInternal(
            const SimplexId &cellId) const override {
            int tid = omp_get_thread_num();
            SimplexId segId = hIndexArray_[hConnectivity_[hOffsets_[cellId]]];
            currClusters_[tid] = segId, currRelations_[tid] = RelationType::TTRelation;
            if (allClusters_[segId].cellNeighbors_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                getClusterCellNeighbors(segId);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
            }
            SimplexId localCellId = cellId - hCellIntervals_[segId - 1];
            return allClusters_[segId].edgeStars_.size(localCellId);
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::ETRelation;
                int qid = relationMaps_.at(RelationType::ETRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::ETRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = hIndexArray_[edgeList_[edgeId][0]], currRelations_[tid] = RelationType::EFRelation;
                int qid = relationMaps_.at(RelationType::EFRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::EFRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::FTRelation;
                int qid = relationMaps_.at(RelationType::FTRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::FTRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexNeighbors_.empty()) {
                int tid = omp_get_thread_num();
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::VVRelation;
                int qid = relationMaps_.at(RelationType::VVRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexNeighbors_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::VVRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexEdges_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::VERelation;
                int qid = relationMaps_.at(RelationType::VERelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexEdges_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::VERelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexStars_.empty()) {
                int tid = omp_get_thread_num();
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = segId, currRelations_[tid] = RelationType::VTRelation;
                int qid = relationMaps_.at(RelationType::VTRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                // std::cout << "thread " << tid << " waiting on cluster " << currClusters_[tid] << std::endl;
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexStars_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::VTRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexTriangles_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                currClusters_[tid] = hIndexArray_[vertexId], currRelations_[tid] = RelationType::VFRelation;
                int qid = relationMaps_.at(RelationType::VFRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
            if (allClusters_[segId].vertexTriangles_.empty()) {
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::VFRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::BERelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
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
                #ifdef MEASURE_TIME
                timers_[tid].reStart();
                #endif
                int qid = relationMaps_.at(RelationType::BVRelation);
                std::unique_lock<std::mutex> qlock(qMutexes_[qid]);
                #ifdef MEASURE_TIME
                jobQueues_[qid].push(std::make_pair(tid, Timer{}));
                #else
                jobQueues_[qid].push(tid);
                #endif
                qlock.unlock();
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingTimes_[tid][1] += timers_[tid].getElapsedTime();
                }
                #endif
                sem_wait(&relationSems_[tid]);
                #ifdef MEASURE_TIME
                if (startTracking_) {
                    waitingCounts_[tid]++;
                    waitingTimes_[tid][0] += timers_[tid].getElapsedTime();
                }
                #endif
            }
            SimplexId localVertexId = vertexId - hVertexIntervals_[segId - 1];
            return allClusters_[segId].boundaryVertices_.at(localVertexId);
        }

        /* Compute the interval array for vertices */
        // THIS WAS PRIVATE, I'M MAKING IT PUBLIC
        void computeVertexIntervals();

    private:

        /* Build the internal edge list in the cluster. */
        int buildInternalEdgeList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 2>>> &clusterEdges);

        /* Build the internal triangle list in the cluster. */
        int buildInternalTriangleList(const SimplexId &clusterId, std::vector<std::vector<std::array<SimplexId, 3>>> &clusterTriangles);

        /* Compute the interval array for cells */
        void computeCellIntervals();

        /* Change the vertex id based on the cluster index */
        int reorderVertices(std::vector<SimplexId> &vertexMap);

        /* Check if the cell array is ordered in a specific way */
        int checkCells();

        /* Change the vertex id in the cell array */
        int reorderCells(const std::vector<SimplexId> &vertexMap);

        int getClusterVertexStarsVector(const SimplexId &clusterId,
                std::vector<std::vector<SimplexId>> &vertexStarVec) const;

        /* Compute cell neighbors in a cluster */
        int getClusterCellNeighbors(const SimplexId &clusterId) const;

        /* Function used by the leader producer thread */
        void leaderProducer(const int id);

        /* Compute the array for the given topological relation */
        int computeRelation(const RelationType relation, const SimplexId clusterId, bool global = false, int threadId = 0, bool signal = false);

        /* Compute the relations for different starting clusters */
        int computeCombinedRelation(const std::vector<int> &threadIds,
                                    const int producerId);

        bool startTracking_;
        bool doublePrecision_;
        int cellDimension_;
        int clusterNumber_;
        int precompClusterNumber_;
        int vertexNumber_, edgeNumber_, triangleNumber_, cellNumber_;
        std::vector<RelationType> preconditionedRelations_{};
        std::unordered_map<RelationType, int> relationMaps_;
        // std::unordered_map<int, cudaStream_t> threadStreams_;
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
        bool *dBoundaryTriangles_;

        struct RelationComputation {
            int *hSizeArray, *hDataArray;
            int *dSizeArray, *dDataArray;
            int *dClusterIds, *hSimplexOffsets, *dSimplexOffsets;
        };
        std::unordered_map<int, RelationComputation> relationCompVecs_;

        /* Multithreading support */
        std::mutex producerMutex_;
        std::vector<std::thread> producers_{};
        int threadsPerCluster_, SFCP_ColumnNum;
        std::array<std::vector<int>, 25> baseClusters_;
        mutable std::vector<boost::shared_mutex> clusterMutexes_;
        mutable std::vector<SimplexId> currClusters_;
        mutable std::vector<RelationType> currRelations_;
        mutable std::vector<sem_t> relationSems_;
        mutable std::array<std::mutex, 25> qMutexes_;
        #ifdef MEASURE_TIME
        mutable std::vector<Timer> timers_;
        mutable std::vector<std::array<double, 6>> waitingTimes_;
        mutable std::vector<int> waitingCounts_;
        mutable std::vector<std::queue<std::pair<int, Timer>>> jobQueues_;
        #else
        mutable std::vector<std::queue<int>> jobQueues_;
        // mutable std::vector<std::unordered_map<RelationType, int>> waitingCounts_;
        // mutable std::vector<std::unordered_map<RelationType, double>> waitingTimes_;
        // std::vector<std::vector<int>> qSizes_;
        // std::vector<std::vector<float>> kernelTimes_;
        // std::vector<std::vector<float>> returnTimes_;
        // std::vector<std::vector<double>> qTimes_;
        #endif
    };
}
