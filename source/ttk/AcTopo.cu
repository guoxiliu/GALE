#include "Relations.cuh"
#include "AcTopo.h"
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <iostream>

using namespace ttk;


AcTopo::AcTopo(int tnum) {
    /* initialize the pointers */
    hConnectivity_ = nullptr;
    hOffsets_ = nullptr;
    hIndexArray_ = nullptr;
    hCellIntervals_ = nullptr;
    hTriangleIntervals_ = nullptr;
    hEdgeIntervals_ = nullptr;
    hVertexIntervals_ = nullptr;
    dIndexArray_ = nullptr;
    dCellIntervals_ = nullptr;
    dTriangleIntervals_ = nullptr;
    dEdgeIntervals_ = nullptr;
    dVertexIntervals_ = nullptr;

    /* relation matrices */
    dVertexNeighborSizes_ = nullptr;
    dVertexNeighborMatrix_ = nullptr;
    dVertexStarSizes_ = nullptr;
    dVertexStarMatrix_ = nullptr;

    threadNumber_ = tnum;
    currClusters_ = std::vector<SimplexId>(threadNumber_, 0);
    currRelations_ = std::vector<RelationType>(threadNumber_, RelationType::BlankRelation);
    waitingCounts_ = std::vector<int>(threadNumber_, 0);
    relationSems_ = std::vector<std::vector<sem_t>>(threadNumber_, std::vector<sem_t>(25));
    for (int i = 0; i < 25; ++i) {
        baseClusters_[i] = std::vector<int>(threadNumber_, 0);
    }

    // GPU parameters
    numBlocks_ = 30, blockSize_ = 32;
}


AcTopo::~AcTopo() {
    if (producer_.joinable()) {
        currClusters_[0] = -1;
        producer_.join();
    }

    for (int i = 0; i < threadNumber_; ++i)
        std::cout << "waiting count " << i << ": " << waitingCounts_[i] << std::endl;
    // host memory deallocation
    if (hCellIntervals_ != nullptr) free(hCellIntervals_);
    if (hTriangleIntervals_ != nullptr) free(hTriangleIntervals_);
    if (hEdgeIntervals_ != nullptr) free(hEdgeIntervals_);
    if (hVertexIntervals_ != nullptr) free(hVertexIntervals_);

    // relation matrices on host memory gets freed during the computation

    // device memory deallocation
    if (dIndexArray_ != nullptr) cudaFree(dIndexArray_);
    if (dCellIntervals_ != nullptr) cudaFree(dCellIntervals_);
    if (dTriangleIntervals_ != nullptr) cudaFree(dTriangleIntervals_);
    if (dEdgeIntervals_ != nullptr) cudaFree(dEdgeIntervals_);
    if (dVertexIntervals_ != nullptr) cudaFree(dVertexIntervals_);

    if (dVertexNeighborSizes_ != nullptr) cudaFree(dVertexNeighborSizes_);
    if (dVertexNeighborMatrix_ != nullptr) cudaFree(dVertexNeighborMatrix_);
    if (dVertexStarSizes_ != nullptr) cudaFree(dVertexStarSizes_);
    if (dVertexStarMatrix_ != nullptr) cudaFree(dVertexStarMatrix_);

}

int AcTopo::computeRelation(const SimplexId clusterId, 
                            const RelationType relation, 
                            const int threadId = 0,
                            bool signal = false) {

    // std::cout << "computing cluster " << clusterId << " relation " << static_cast<int>(relation) << std::endl;
    int res = 0;
    
    switch (relation) {
        case RelationType::VVRelation:
        {
            int totalBlocks = numBlocks_ * blockSize_;

            // if (dVertexNeighborSizes_ == nullptr) {
                int vertexIdOffset = hVertexIntervals_[clusterId-1];
                int endCluster = min(clusterId + totalBlocks, clusterNumber_) - 1;
                int localVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[clusterId-1];
                CHECK(cudaMalloc((void **)&dVertexNeighborSizes_, sizeof(int) * localVertexNum));
                CHECK(cudaMalloc((void **)&dVertexNeighborMatrix_, sizeof(int) * localVertexNum * matrixColumns));

                precomputeVertexNeighbors<<<numBlocks_, blockSize_>>>(
                    dVertexNeighborMatrix_, dVertexNeighborSizes_, 
                    clusterId, clusterNumber_, 
                    dConnectivity_, dOffsets_, dIndexArray_, 
                    dVertexIntervals_, dCellIntervals_, 
                    dExternalCells_, dExternalCellOffsets_, cellDimension_);
            // }

            hVertexNeighborSizes_ = new int[localVertexNum];
            hVertexNeighborMatrix_ = new int[localVertexNum * matrixColumns];

            cudaError_t status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                    __FILE__, __LINE__); if (abort) exit(status);
            }

            CHECK(cudaMemcpy(hVertexNeighborSizes_, dVertexNeighborSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexNeighborMatrix_, dVertexNeighborMatrix_, sizeof(int) * localVertexNum * matrixColumns, cudaMemcpyDeviceToHost));

            std::unique_lock<std::mutex> lck(VVMutex_);
            for (int i = 0; i < localVertexNum; ++i) {
                SimplexId vid = vertexIdOffset + i;
                if (!vertexNeighbors_[vid].empty()) continue;
                vertexNeighbors_[vid] = std::vector<int>(hVertexNeighborMatrix_ + i * matrixColumns, hVertexNeighborMatrix_ + i * matrixColumns + hVertexNeighborSizes_[i]);
            }
            baseClusters_[static_cast<int>(RelationType::VVRelation)][threadId] = clusterId;

            lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VVSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId][static_cast<int>(relation)]);
            }

            CHECK(cudaFree(dVertexNeighborSizes_)); dVertexNeighborSizes_ = nullptr;
            CHECK(cudaFree(dVertexNeighborMatrix_)); dVertexNeighborMatrix_ = nullptr;
            
            delete[] hVertexNeighborMatrix_;
            delete[] hVertexNeighborSizes_;
            break;
        }

        case RelationType::VTRelation:
        {
            int totalBlocks = numBlocks_ * blockSize_;

            // if (dVertexStarSizes_ == nullptr) {
                int vertexIdOffset = hVertexIntervals_[clusterId-1];
                int endCluster = min(totalBlocks + clusterId, clusterNumber_) - 1;
                int localVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[clusterId-1];
                baseClusters_[static_cast<int>(RelationType::VTRelation)][threadId] = clusterId;
                CHECK(cudaMalloc((void **)&dVertexStarSizes_, sizeof(int) * localVertexNum));
                CHECK(cudaMalloc((void **)&dVertexStarMatrix_, sizeof(int) * localVertexNum * matrixColumns));

                precomputeVertexStars<<<numBlocks_, blockSize_>>>(
                    dVertexStarMatrix_, dVertexStarSizes_, 
                    clusterId, clusterNumber_, 
                    dConnectivity_, dOffsets_, dIndexArray_, 
                    dVertexIntervals_, dCellIntervals_, 
                    dExternalCells_, dExternalCellOffsets_, cellDimension_);
            // }

            hVertexStarSizes_ = new int[localVertexNum];
            hVertexStarMatrix_ = new int[localVertexNum * matrixColumns];

            cudaError_t status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
                    __FILE__, __LINE__); if (abort) exit(status);
            }

            CHECK(cudaMemcpy(hVertexStarSizes_, dVertexStarSizes_, sizeof(int) * localVertexNum, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hVertexStarMatrix_, dVertexStarMatrix_, sizeof(int) * localVertexNum * matrixColumns, cudaMemcpyDeviceToHost));

            std::unique_lock<std::mutex> lck(VTMutex_);
            for (int i = 0; i < localVertexNum; ++i) {
                SimplexId vid = vertexIdOffset + i;
                if (!vertexStars_[vid].empty()) continue;
                vertexStars_[vid] = std::vector<int>(hVertexStarMatrix_ + i * matrixColumns, hVertexStarMatrix_ + i * matrixColumns + hVertexStarSizes_[i]);
            }
            lck.unlock();
            if (signal) {
                // if (clusterId <= endCluster) sem_post(&VTSems_[threadId]);
                // else res = 1;
                sem_post(&relationSems_[threadId][static_cast<int>(relation)]);
            }

            CHECK(cudaFree(dVertexStarSizes_)); dVertexStarSizes_ = nullptr;
            CHECK(cudaFree(dVertexStarMatrix_)); dVertexStarMatrix_ = nullptr;
            
            delete[] hVertexStarMatrix_;
            delete[] hVertexStarSizes_;
            break;
        }

        default:
            res = -1;
    }

    return res;
}


void AcTopo::assistProcedure() {
    volatile SimplexId leaderCluster = 0;
    volatile RelationType leaderFunc = RelationType::BlankRelation;

    while (true) {
        std::unique_lock<std::mutex> qlock(qMutex_);
        int qSize = jobQueue.size();
        qlock.unlock();

        if (qSize > 0) {
            int qSize = jobQueue.size();
            // handle the waiting jobs in the queue
            for (int i = 0; i < qSize; ++i) {
                int tid = jobQueue.front();
                leaderCluster = currClusters_[tid], leaderFunc = currRelations_[tid];

                // no need to compute if the relation is already there from another thread
                int baseCluster = baseClusters_[static_cast<int>(leaderFunc)][tid];
                if (baseCluster > 0) {
                    SimplexId diff = leaderCluster - baseCluster;
                    // check if the precomputation needs to be performed
                    if (diff > 0 && diff < numBlocks_ * blockSize_) {
                        sem_post(&relationSems_[tid][static_cast<int>(leaderFunc)]);
                        std::unique_lock<std::mutex> qlock(qMutex_);
                        jobQueue.pop();
                        qlock.unlock();
                        continue;
                    }
                }

                int res = computeRelation(leaderCluster, leaderFunc, tid, true);
                if (res == 0) {
                    std::unique_lock<std::mutex> qlock(qMutex_);
                    jobQueue.pop();
                    qlock.unlock();
                }
            }
        }
        else {
            for (int i = 0; i < threadNumber_; ++i) {
                leaderCluster = currClusters_[i], leaderFunc = currRelations_[i];
                if (leaderCluster == -1) return;
                else if (leaderCluster == 0) continue;
                SimplexId diff = leaderCluster - baseClusters_[static_cast<int>(leaderFunc)][i];
                // check if the precomputation needs to be performed
                if (diff < 0 || diff > numBlocks_ * blockSize_ * 0.6) {
                    // std::cout << "precomputing for thread " << i << " cluster = " << leaderCluster << " diff = " << diff << std::endl;
                    computeRelation(leaderCluster, leaderFunc, i, false);
                }
            }
        }
    }
}

void AcTopo::initialize(vtkPointSet *pointSet, int blockNum, int blockSize) {
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
    if(pointDataType != VTK_FLOAT && pointDataType != VTK_DOUBLE) {
        std::cerr << "Unable to initialize 'ttk::Triangulation' for point "
                                        "precision other than 'float' or 'double'." << std::endl;
    }

    void *pointDataArray = points->GetData()->GetVoidPointer(0);
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

    CHECK(cudaMallocAsync(&dConnectivity_, sizeof(vtkIdType) * (cellNumber_ * 4), 0));
    CHECK(cudaMallocAsync(&dOffsets_, sizeof(vtkIdType) * cellNumber_, 0));
    CHECK(cudaMemcpyAsync(dConnectivity_, hConnectivity_, sizeof(vtkIdType) * (cellNumber_ * 4), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(dOffsets_, hOffsets_, sizeof(vtkIdType) * cellNumber_, cudaMemcpyHostToDevice));

    // index array
    hIndexArray_ = nullptr;
    {
        int hasIndexArray = pointSet->GetPointData()->HasArray("_index");
        if (hasIndexArray) {
            hIndexArray_ = (int *) pointSet->GetPointData()->GetAbstractArray("_index")->GetVoidPointer(0);
            for (int i = 0; i < vertexNumber_; ++i) hIndexArray_[i] += 1;
            CHECK(cudaMallocAsync(&dIndexArray_, sizeof(int) * vertexNumber_, 0));
            CHECK(cudaMemcpyAsync(dIndexArray_, hIndexArray_, sizeof(int) * vertexNumber_, cudaMemcpyHostToDevice));
        }
        else {
            std::cerr << "Cannot detect index array! AcTopo must run with a predefined clustering information!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    clusterNumber_ = hIndexArray_[vertexNumber_ - 1] + 1;
    for (int i = 0; i < threadNumber_; ++i) {
        for (int j = 0; j < 25; ++j) {
            sem_init(&relationSems_[i][j], 0, 0);
        }
    }
    if (blockNum > 0) numBlocks_ = blockNum;
    computeCellIntervals();
    computeVertexIntervals();

    // // Compute cluster numbers of each thread
    // int verticesPerThread = vertexNumber_ / threadNumber_;
    // for (int i = 1; i < threadNumber_; ++i) {
    //     std::cout << "thread " << i << ": " << hIndexArray_[verticesPerThread * i] << std::endl;
    // }
    // std::cout << "thread " << threadNumber_ << ": " << clusterNumber_ << std::endl;

    producer_ = std::thread(&AcTopo::assistProcedure, this);
    printf("Block number: %d, Cluster number: %d, cell number: %d, vertex number: %d\n", numBlocks_, clusterNumber_, cellNumber_, vertexNumber_);
}


void AcTopo::computeCellIntervals() {
    std::vector<std::vector<int>> externalCells(clusterNumber_);
    std::vector<int> hExternalCellOffsets(clusterNumber_, 0);
    hCellIntervals_ = (int *) calloc(clusterNumber_, sizeof(int));

    CHECK(cudaMallocAsync(&dCellIntervals_, sizeof(int) * clusterNumber_, 0));
    CHECK(cudaMallocAsync(&dExternalCellOffsets_, sizeof(int) * clusterNumber_, 0));

    for (int cid = 0; cid < cellNumber_; ++cid) {
        int clusterId = hIndexArray_[hConnectivity_[hOffsets_[cid]]];
        for (int j = 1; j <= cellDimension_; ++j) {
            int tmpCluster = hIndexArray_[hConnectivity_[hOffsets_[cid] + j]];
            if (tmpCluster > clusterId) {
                if (externalCells[tmpCluster].empty() || externalCells[tmpCluster].back() != cid) {
                    externalCells[tmpCluster].push_back(cid);
                }
            }
        }
        hCellIntervals_[clusterId]++;
    }

    for (int i = 1; i < clusterNumber_; ++i) {
        hCellIntervals_[i] += hCellIntervals_[i-1];
        hExternalCellOffsets[i] = hExternalCellOffsets[i-1] + externalCells[i].size();
    }
    CHECK(cudaMemcpyAsync(dCellIntervals_, hCellIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyAsync(dExternalCellOffsets_, hExternalCellOffsets.data(), sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));

    // Convert the 2D vector to 1D flat vector
    hExternalCells_ = new int[hExternalCellOffsets[clusterNumber_-1]];
    int *tmp = hExternalCells_;
    CHECK(cudaMalloc((void **) &dExternalCells_, sizeof(int) * hExternalCellOffsets[clusterNumber_-1]));
    for (int i = 1; i < externalCells.size(); ++i) {
        std::copy(externalCells[i].begin(), externalCells[i].end(), tmp);
        tmp += externalCells[i].size();
    }
    CHECK(cudaMemcpy(dExternalCells_, hExternalCells_, sizeof(int) * hExternalCellOffsets[clusterNumber_-1], cudaMemcpyHostToDevice));

    // int blockSize = 32;
    // int numBlocks = vertexNumber_ / blockSize + 1;
    // cudaMalloc(&dCellIntervals_, sizeof(int) * clusterNumber_);
    // cudaMemset(dCellIntervals_, 0, clusterNumber_);

    // countCells<<<numBlocks, blockSize>>>(dCellIntervals_, dConnectivity_, dOffsets_, dIndexArray_, cellNumber_, 3);

    // hCellIntervals_ = (int *) malloc(sizeof(int) * clusterNumber_);
    // cudaMemcpy(hCellIntervals_, dCellIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyDeviceToHost);
}


void AcTopo::computeVertexIntervals() {
    // int blockSize = 64;
    // int numBlocks = vertexNumber_ / blockSize + 1;

    // cudaMalloc(&dVertexIntervals_, sizeof(int) * (clusterNumber_+1));
    // cudaMemset(dVertexIntervals_, 0, clusterNumber_+1);
    // countVertices<<<numBlocks, blockSize>>>(dVertexIntervals_, dIndexArray_, vertexNumber_);

    // hVertexIntervals_ = (int *) malloc(sizeof(int) * (clusterNumber_+1));
    // cudaMemcpy(hVertexIntervals_, dVertexIntervals_, sizeof(int) * (clusterNumber_+1), cudaMemcpyDeviceToHost);
    // hVertexIntervals_[0] = -1;
    // for (int i = 1; i <= clusterNumber_; ++i) hVertexIntervals_[i] = hVertexIntervals_[i-1] + hVertexIntervals_[i];

    CHECK(cudaMallocAsync(&dVertexIntervals_, sizeof(int) * clusterNumber_, 0));

    int curCluster = 1;
    hVertexIntervals_ = (int *) malloc(sizeof(int) * clusterNumber_);
    hVertexIntervals_[0] = 0;
    for (int vid = 0; vid < vertexNumber_; ++vid) {
        int clusterId = hIndexArray_[vid];
        if (curCluster != clusterId) {
            hVertexIntervals_[curCluster] = vid;
            curCluster = clusterId;
        }
    }
    hVertexIntervals_[curCluster] = vertexNumber_;

    CHECK(cudaMemcpyAsync(dVertexIntervals_, hVertexIntervals_, sizeof(int) * clusterNumber_, cudaMemcpyHostToDevice));
}


int AcTopo::preconditionVertexNeighborsInternal() {
    if(this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
        std::cerr << "Empty dataset, precondition skipped";
        return 1;
    }
    vertexNeighbors_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});

    // // let GPU compute vertex neighbor data
    // int endCluster = min(numBlocks_ * blockSize_, clusterNumber_ - 1);
    // const int computedVertexNum = hVertexIntervals_[endCluster] - hVertexIntervals_[0];
    
    // CHECK(cudaMalloc((void **) &dVertexNeighborSizes_, sizeof(int) * computedVertexNum));
    // CHECK(cudaMalloc((void **) &dVertexNeighborMatrix_, sizeof(int) * computedVertexNum * matrixColumns));

    // precomputeVertexNeighbors<<<numBlocks_, blockSize_>>>(
    //     dVertexNeighborMatrix_, dVertexNeighborSizes_, 
    //     1, clusterNumber_, 
    //     dConnectivity_, dOffsets_, dIndexArray_,
    //     dVertexIntervals_, dCellIntervals_, dExternalCells_,
    //     dExternalCellOffsets_, cellDimension_);
    

    // hVertexNeighborSizes_ = new int[computedVertexNum];
    // hVertexNeighborMatrix_ = new int[computedVertexNum * matrixColumns];

    // cudaError_t status = cudaDeviceSynchronize();
    // if (status != cudaSuccess) {
    //     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //         __FILE__, __LINE__); if (abort) exit(status);
    // }

    // CHECK(cudaMemcpy(hVertexNeighborSizes_, dVertexNeighborSizes_, sizeof(int) * computedVertexNum, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(hVertexNeighborMatrix_, dVertexNeighborMatrix_, sizeof(int) * computedVertexNum * matrixColumns, cudaMemcpyDeviceToHost));

    // std::unique_lock<std::mutex> lck(VVMutex_);
    // for (int vid = 0; vid < computedVertexNum; ++vid) {
    //     if (!vertexNeighbors_[vid].empty()) continue;
    //     vertexNeighbors_[vid] = std::vector<int>(hVertexNeighborMatrix_ + vid * matrixColumns, hVertexNeighborMatrix_ + vid * matrixColumns + hVertexNeighborSizes_[vid]);
    // }
    // baseClusters_[static_cast<int>(RelationType::VVRelation)][0] = 1;

    // CHECK(cudaFree(dVertexNeighborSizes_)); dVertexNeighborSizes_ = nullptr;
    // CHECK(cudaFree(dVertexNeighborMatrix_)); dVertexNeighborMatrix_ = nullptr;
    
    // delete[] hVertexNeighborMatrix_;
    // delete[] hVertexNeighborSizes_;

    return 0;
}


int AcTopo::preconditionVertexStarsInternal() {
    if(this->hConnectivity_ == nullptr || this->vertexNumber_ == 0) {
        std::cerr << "Empty dataset, precondition skipped";
        return 1;
    }
    vertexStars_ = std::vector<std::vector<int>>(vertexNumber_, std::vector<int>{});

    // // let GPU compute vertex neighbor data
    // int endCluster = min(numBlocks_ * blockSize_, clusterNumber_ - 1);
    // const int computedVertexNum = hVertexIntervals_[endCluster];

    // CHECK(cudaMalloc((void **) &dVertexStarSizes_, sizeof(int) * computedVertexNum));
    // CHECK(cudaMalloc((void **) &dVertexStarMatrix_, sizeof(int) * computedVertexNum * matrixColumns));

    // precomputeVertexStars<<<numBlocks_, blockSize_>>>(
    //     dVertexStarMatrix_, dVertexStarSizes_, 
    //     1, clusterNumber_, 
    //     dConnectivity_, dOffsets_, dIndexArray_,
    //     dVertexIntervals_, dCellIntervals_, dExternalCells_,
    //     dExternalCellOffsets_, cellDimension_);

    // hVertexStarSizes_ = new int[computedVertexNum];
    // hVertexStarMatrix_ = new int[computedVertexNum * matrixColumns];

    // cudaError_t status = cudaDeviceSynchronize();
    // if (status != cudaSuccess) {
    //     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(status),
    //         __FILE__, __LINE__); if (abort) exit(status);
    // }

    // CHECK(cudaMemcpy(hVertexStarSizes_, dVertexStarSizes_, sizeof(int) * computedVertexNum, cudaMemcpyDeviceToHost));
    // CHECK(cudaMemcpy(hVertexStarMatrix_, dVertexStarMatrix_, sizeof(int) * computedVertexNum * matrixColumns, cudaMemcpyDeviceToHost));

    // std::unique_lock<std::mutex> lck(VTMutex_);
    // for (int vid = 0; vid < computedVertexNum; ++vid) {
    //     if (!vertexStars_[vid].empty()) continue;
    //     vertexStars_[vid] = std::vector<int>(hVertexStarMatrix_ + vid * matrixColumns, hVertexStarMatrix_ + vid * matrixColumns + hVertexStarSizes_[vid]);
    // }
    // lck.unlock();

    // CHECK(cudaFree(dVertexStarSizes_)); dVertexStarSizes_ = nullptr;
    // CHECK(cudaFree(dVertexStarMatrix_)); dVertexStarMatrix_ = nullptr;
    
    // delete[] hVertexStarMatrix_;
    // delete[] hVertexStarSizes_;

    return 0;
}
