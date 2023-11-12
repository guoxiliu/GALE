#pragma once

#include "AbstractTriangulation.h"
#include <vtkPointSet.h>
#include <atomic>
#include <mutex>
#include <thread>
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
        AcTopo();
        ~AcTopo();
        void initialize(vtkPointSet *pointSet);


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
            TTK_TRIANGULATION_INTERNAL(getNumberOfVertices)() const override {
                return vertexNumber_;
        }

        inline int TTK_TRIANGULATION_INTERNAL(getVertexNeighbor)(
            const SimplexId &vertexId,
            const int &localNeighborId,
            SimplexId &neighborId) const override {

            if (vertexNeighbors_[vertexId].empty()) {
                waiting_ = true;
                currCluster_ = hIndexArray_[vertexId], currRelation_ = RelationType::VVRelation;
                sem_wait(&VVSem_);
                waiting_ = false;
            }
            std::lock_guard<std::mutex> lck(VVMutex_);
            neighborId = vertexNeighbors_[vertexId][neighborId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexNeighborNumber)(
            const SimplexId &vertexId) const override {
            
            if (vertexNeighbors_[vertexId].empty()) {
                waiting_ = true;
                currCluster_ = hIndexArray_[vertexId], currRelation_ = RelationType::VVRelation;
                sem_wait(&VVSem_);
                waiting_ = false;
            }
            std::lock_guard<std::mutex> lck(VVMutex_);
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
                waiting_ = true;
                currCluster_ = hIndexArray_[vertexId], currRelation_ = RelationType::VTRelation;
                sem_wait(&VTSem_);
                waiting_ = false;
            }
            std::lock_guard<std::mutex> lck(VTMutex_);
            starId = vertexStars_[vertexId][starId];
            return 0;
        }

        inline SimplexId TTK_TRIANGULATION_INTERNAL(getVertexStarNumber)(
            const SimplexId &vertexId) const override {
                return 0;
            
            if (vertexStars_[vertexId].empty()) {
                waiting_ = true;
                currCluster_ = hIndexArray_[vertexId], currRelation_ = RelationType::VTRelation;
                sem_wait(&VTSem_);
                waiting_ = false;
            }
            std::lock_guard<std::mutex> lck(VTMutex_);
            return vertexStars_[vertexId].size();
        }

        inline const std::vector<std::vector<SimplexId>> *
            TTK_TRIANGULATION_INTERNAL(getVertexStars)() override {
            // vertexStarData_.copyTo(vertexStarList_);
            // TODO: copy the matrix data back to vector of vector
            return &vertexStarList_;
        }


    private:
        void computeCellIntervals();
        void computeVertexIntervals();
        void assistProcedure();
        void computeRelation(const SimplexId clusterId, const RelationType relation);

        int cellDimension_;
        int clusterNumber_;
        int vertexNumber_, edgeNumber_, triangleNumber_, cellNumber_;

        // data allocated on host memory
        volatile int numBlocks_, blockSize_;
        int *hIndexArray_, *hExternalCells_;
        vtkIdType *hConnectivity_, *hOffsets_;
        int *hCellIntervals_, *hTriangleIntervals_, *hEdgeIntervals_, *hVertexIntervals_;

        // data allocated on device memory
        vtkIdType *dConnectivity_, *dOffsets_;
        int *dIndexArray_;
        int *dCellIntervals_, *dTriangleIntervals_, *dEdgeIntervals_, *dVertexIntervals_;
        int *dExternalCells_, *dExternalCellOffsets_;

        // relation matrices
        std::vector<std::vector<int>> vertexNeighbors_;
        int *hVertexNeighborSizes_, *hVertexNeighborMatrix_;
        int *dVertexNeighborSizes_, *dVertexNeighborMatrix_;
        
        std::vector<std::vector<int>> vertexStars_;
        int *hVertexStarSizes_, *hVertexStarMatrix_;
        int *dVertexStarSizes_, *dVertexStarMatrix_;

        // multithreading support
        std::thread producer_;
        mutable volatile SimplexId currCluster_, baseCluster_;
        mutable volatile RelationType currRelation_;
        mutable std::mutex VVMutex_, VTMutex_;
        mutable sem_t VVSem_, VTSem_;
        mutable std::atomic<bool> waiting_;
        
    };
}
