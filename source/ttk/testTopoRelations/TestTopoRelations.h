#pragma once

#include <AbstractTriangulation.h>
#include <Gale.h>
#include <MemoryUsage.h>
#include <Timer.h>
#include <iostream>

namespace ttk {
    class TestTopoRelations : virtual public Debug {
    public:
      TestTopoRelations() {}
      ~TestTopoRelations() {}

      int execute(AbstractTriangulation *triangulation) const;
      inline int preconditionTriangulation(AbstractTriangulation *triangulation){
        if(triangulation){
          Timer t;
          // build edges and triangles
          // triangulation->preconditionEdges();
          // triangulation->preconditionTriangles();
          // // boundary relations
          // triangulation->preconditionBoundaryVertices();
          // triangulation->preconditionBoundaryEdges();
          // triangulation->preconditionBoundaryTriangles();
          // vertex related relations
          triangulation->preconditionVertexEdges();
          triangulation->preconditionVertexNeighbors();
          triangulation->preconditionVertexStars();
          triangulation->preconditionVertexTriangles();
          // edge related relations
          // triangulation->preconditionEdgeStars();
          triangulation->preconditionEdgeTriangles();
          // triangle related relations
          // triangulation->preconditionTriangleEdges();
          triangulation->preconditionTriangleStars();
          // cell related relations
          // triangulation->preconditionCellEdges();
          // triangulation->preconditionCellNeighbors();
          // triangulation->preconditionCellTriangles();
          // links
          // triangulation->preconditionVertexLinks();
          // triangulation->preconditionEdgeLinks();
          // triangulation->preconditionTriangleLinks();
          std::cout << "[TestTopoCluster] Time usage for preconditioning: "
                  << t.getElapsedTime() << " s." << std::endl;
        }
        return 0;
      }
    };
    
    int TestTopoRelations::execute(AbstractTriangulation *triangulation) const {

      std::cout << "[TestTopoRelations] Dimension: " << triangulation->getDimensionality() << std::endl;

      SimplexId vertexNumber = triangulation->getNumberOfVertices();
      SimplexId cellNumber = triangulation->getNumberOfCells();
      SimplexId edgeNumber = triangulation->getNumberOfEdges();
      SimplexId triangleNumber = triangulation->getNumberOfTriangles();

      std::cout << "[TestTopoRelations] vertex num: " << vertexNumber;
      std::cout << ", cell num: " << cellNumber;
      std::cout << ", edge num: " << edgeNumber;
      std::cout << ", triangle num: " << triangleNumber;
      std::cout << std::endl;

      // SimplexId verticesPerCell = triangulation->getCellVertexNumber(0);
      // SimplexId edgesPerCell = triangulation->getCellEdgeNumber(0);
      // SimplexId trianglesPerCell = triangulation->getCellTriangleNumber(0);

      Timer t, tot;

      /* Test the number of threads per mesh segment */
      // Gale *gale = dynamic_cast<Gale *>(triangulation);
      // // const SimplexId testVertexNumber = 100000;
      // if (gale != nullptr) {
      //   // int clusterId = gale->hIndexArray_[testVertexNumber];
      //   // gale->numBlocks_ = clusterId / gale->blockSize_ + 1;
      //   // std::cout << "Computing " << gale->numBlocks_ << " blocks" << std::endl;

      //   t.reStart();
      //   gale->computeRelation(RelationType::EFRelation, 1, 0, false);
      //   std::cout << "[TestTopoRelations] Time usage for EF: " << t.getElapsedTime() << " s.\n";
        
      //   t.reStart();
      //   gale->computeRelation(RelationType::ETRelation, 1, 0, false);
      //   std::cout << "[TestTopoRelations] Time usage for ET: " << t.getElapsedTime() << " s.\n";
        
      //   t.reStart();
      //   gale->computeRelation(RelationType::FTRelation, 1, 0, false);
      //   std::cout << "[TestTopoRelations] Time usage for FT: " << t.getElapsedTime() << " s.\n";
        
      //   t.reStart();
      //   gale->computeRelation(RelationType::VVRelation, 1, 0, false);
      //   std::cout << "[TestTopoRelations] Time usage for VV: " << t.getElapsedTime() << " s.\n";

      //   t.reStart();
      //   gale->computeRelation(RelationType::VERelation, 1, 0, false);
      //   std::cout << "[TestTopoRelations] Time usage for VE: " << t.getElapsedTime() << " s.\n";

      //   t.reStart();
      //   gale->computeRelation(RelationType::VFRelation, 1, 0, false);
      //   std::cout << "[TestTopoRelations] Time usage for VF: " << t.getElapsedTime() << " s.\n";
        
      //   t.reStart();
      //   gale->computeRelation(RelationType::VTRelation, 1, 0, false);
      //   std::cout << "[TestTopoRelations] Time usage for VT: " << t.getElapsedTime() << " s.\n";
      // }

      /* Time measurement for getting global relation arrays */
      // t.reStart();
      // triangulation->getEdgeTriangles();
      // std::cout << "[TestTopoRelations] Time usage for EF: " << t.getElapsedTime() << " s.\n";
      
      // t.reStart();
      // triangulation->getEdgeStars();
      // std::cout << "[TestTopoRelations] Time usage for ET: " << t.getElapsedTime() << " s.\n";

      // t.reStart();
      // triangulation->getTriangleStars();
      // std::cout << "[TestTopoRelations] Time usage for FT: " << t.getElapsedTime() << " s.\n";
      
      // t.reStart();
      // triangulation->getVertexNeighbors();
      // std::cout << "[TestTopoRelations] Time usage for VV: " << t.getElapsedTime() << " s.\n";
      
      // t.reStart();
      // triangulation->getVertexEdges();
      // std::cout << "[TestTopoRelations] Time usage for VE: " << t.getElapsedTime() << " s.\n";
      
      // t.reStart();
      // triangulation->getVertexTriangles();
      // std::cout << "[TestTopoRelations] Time usage for VF: " << t.getElapsedTime() << " s.\n";
      
      // t.reStart();
      // triangulation->getVertexStars();
      // std::cout << "[TestTopoRelations] Time usage for VT: " << t.getElapsedTime() << " s.\n";

      // Boundary relations
      // t.reStart();
      // int boundaryVertexNum = 0;
      // #pragma omp parallel for num_threads(threadNumber_)
      // for(SimplexId vid = 0; vid < vertexNumber; vid++) {
      //   // std::cout << "vertex " << vid << ": " << triangulation->isVertexOnBoundary(vid) << std::endl;
      //   if(triangulation->isVertexOnBoundary(vid)) {
      //     #pragma omp critical
      //     boundaryVertexNum++;
      //   }
      // }
      // std::cout << "[TestTopoCluster] Boundary vertex number: " << boundaryVertexNum
      //           << std::endl;

      // int boundaryEdgeNum = 0;
      // // std::mutex ioMutex;
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId eid = 0; eid < edgeNumber; eid++) {
      //   if(triangulation->isEdgeOnBoundary(eid)) {
      //     boundaryEdgeNum++;
      //   }
      //   // std::unique_lock<std::mutex> ioLock(ioMutex);
      //   // std::cout << "edge " << eid << ": " << triangulation->isEdgeOnBoundary(eid) << std::endl;
      //   // ioLock.unlock();
      // }
      // std::cout << "[TestTopoCluster] Boundary edge number: " << boundaryEdgeNum
      //           << std::endl;

      // int boundaryTriangleNum = 0;
      // for(SimplexId fid = 0; fid < triangleNumber; fid++) {
      //   if(triangulation->isTriangleOnBoundary(fid)) {
      //     boundaryTriangleNum++;
      //   }
      // }
      // std::cout << "[TestTopoCluster] Boundary triangle number: "
      //           << boundaryTriangleNum << std::endl;
      // std::cout << "[TestTopoCluster] Time usage for getting boundaries: "
      //           << t.getElapsedTime() << " s." << std::endl;


      // Vertex neighbor relation
      {
        // std::mutex ioMutex;
        t.reStart();
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumber_)
        #endif
        for(SimplexId vid = 0; vid < vertexNumber; vid++) {
          SimplexId neighborNum = triangulation->getVertexNeighborNumber(vid);
          SimplexId neighborId;
          // std::unique_lock<std::mutex> ioLock(ioMutex);
          // std::cout << "vertex " << vid << ": " << neighborNum << " [";
          for(int j = 0; j < neighborNum; j++) {
            triangulation->getVertexNeighbor(vid, j, neighborId);
            // std::cout << neighborId << " ";
          }
          // std::cout << "]" << std::endl;
          // ioLock.unlock();
        }
        std::cout << "[TestTopoRelations] Time usage for VN: " << t.getElapsedTime() << " s." << std::endl;
      }

      /* test vertex edge relationship */
      {
        t.reStart();
        // std::mutex ioMutex;
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumber_)
        #endif
        for(SimplexId vertexId = 0; vertexId < vertexNumber; vertexId++){
          SimplexId edgeNum = triangulation->getVertexEdgeNumber(vertexId);
          // std::unique_lock<std::mutex> ioLock(ioMutex);
          // std::cout << "vertex " << vertexId << ": " << edgeNum << " [ ";
          SimplexId edgeId;
          for(SimplexId j = 0; j < edgeNum; j++){
            int result1 = triangulation->getVertexEdge(vertexId, j, edgeId);
            if(result1){
                std::cerr << "[TestTopoRelations] vertexId " << vertexId << " Something wrong in getVertexEdge()! Error code: " << result1 << "\n";
            }
            // std::cout << edgeId << " ";
          }
          // std::cout << "]" << std::endl;
          // ioLock.unlock();
        }
        std::cout << "[TestTopoRelations] Time usage for VE: " << t.getElapsedTime() << " s.\n";
      }

      /* test vertex triangle relationship */
      {
        t.reStart();
        // std::mutex ioMutex;
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumber_)
        #endif
        for(SimplexId vertexId = 0; vertexId < vertexNumber; vertexId++){
          SimplexId triangleNum = triangulation->getVertexTriangleNumber(vertexId);
          // std::unique_lock<std::mutex> ioLock(ioMutex);
          // std::cout << "vertex " << vertexId << ": " << triangleNum << " [";
          SimplexId triangleId;
          for(SimplexId j = 0; j < triangleNum; j++){
            int result1 = triangulation->getVertexTriangle(vertexId, j, triangleId);
            if(result1){
                std::cerr << "[TestTopoRelations] vertexId " << vertexId << " Something wrong in getVertexTriangle()! Error code: " << result1 << "\n";
            }
            // std::cout << triangleId << " ";
          }
          // std::cout << "]" << std::endl;
          // ioLock.unlock();
        }
        std::cout << "[TestTopoRelations] Time usage for VF: " << t.getElapsedTime() << " s.\n";
      }
      
      // /* test vertex tetrahedron relationship */
      {
        t.reStart();
        // std::mutex ioMutex;
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumber_)
        #endif
        for(SimplexId vertexId = 0; vertexId < vertexNumber; vertexId++){
          SimplexId cellNum = triangulation->getVertexStarNumber(vertexId);
          SimplexId cellId;
          // std::unique_lock<std::mutex> ioLock(ioMutex);
          // std::cout << "vertex " << vertexId << ": " << cellNum << " [";
          for(SimplexId j = 0; j < cellNum; j++){
            int result1 = triangulation->getVertexStar(vertexId, j, cellId);
            if(result1){
                std::cout << "[TestTopoRelations] vertexId " << vertexId << " Something wrong in getVertexStar()! Error code: " << result1 << "\n";
            }
            // std::cout << cellId << " ";
          }
          // std::cout << "]" << std::endl;
          // ioLock.unlock();
        }
        std::cout << "[TestTopoRelations] Time usage for VT: " << t.getElapsedTime() << " s.\n";
      }

      // /* test edge vertex relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId edgeId = 0; edgeId < edgeNumber; edgeId++){
      //   SimplexId vertexId;
      //   std::cout << "edge " << edgeId << ": [";
      //   for(SimplexId j = 0; j < 2; j++){
      //     int result1 = triangulation->getEdgeVertex(edgeId, j, vertexId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] edgeId " << edgeId << " Something wrong in getEdgeVertex()! Error code: " << result1 << "\n";
      //     }
      //     std::cout << vertexId << " ";
      //   }
      //   std::cout << "]" << std::endl;
      // }
      // std::cout << "[TestTopoRelations] Time usage for EV: " << t.getElapsedTime() << " s.\n";


      /* test edge triangle relationship */
      {
        t.reStart();
        // std::mutex ioMutex;
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumber_)
        #endif
        for(SimplexId edgeId = 0; edgeId < edgeNumber; edgeId++){
          SimplexId triangleNum = triangulation->getEdgeTriangleNumber(edgeId);
          // std::unique_lock<std::mutex> ioLock(ioMutex);
          // std::cout << "edge " << edgeId << ": " << triangleNum << "[";
          SimplexId triangleId;
          for(SimplexId j = 0; j < triangleNum; j++){
            int result1 = triangulation->getEdgeTriangle(edgeId, j, triangleId);
            if(result1){
                std::cout << "[TestTopoRelations] edgeId " << edgeId << " Something wrong in getEdgeTriangle()! Error code: " << result1 << "\n";
            }
            // std::cout << triangleId << " ";
          }
          // std::cout << "]" << std::endl;
          // ioLock.unlock();
        }
        std::cout << "[TestTopoRelations] Time usage for EF: " << t.getElapsedTime() << " s.\n";
      }
      
      /* test edge tetrahedron relationship */
      {
        t.reStart();
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumber_)
        #endif
        for(SimplexId edgeId = 0; edgeId < edgeNumber; edgeId++){
          SimplexId cellNum = triangulation->getEdgeStarNumber(edgeId);
          SimplexId cellId;
          // #ifdef TTK_ENABLE_OPENMP
          // std::cout << "edge " << edgeId << ": " << cellNum << " [";
          for(SimplexId j = 0; j < cellNum; j++){
            int result1 = triangulation->getEdgeStar(edgeId, j, cellId);
            if(result1){
                std::cout << "[TestTopoRelations] edgeId " << edgeId << " Something wrong in getEdgeStar()! Error code: " << result1 << "\n";
            }
            // std::cout << cellId << " ";
          }
          // std::cout << "]" << std::endl;
        }
        std::cout << "[TestTopoRelations] Time usage for ET: " << t.getElapsedTime() << " s.\n";
      }

      // /* test triangle vertex relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId triangleId = 0; triangleId < triangleNumber; triangleId++){
      //   SimplexId vertexId;
      //   std::cout << "triangle " << triangleId << ": [";
      //   for(SimplexId j = 0; j < 3; j++){
      //   int result1 = triangulation->getTriangleVertex(triangleId, j, vertexId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] triangleId " << triangleId << " Something wrong in getTriangleVertex()! Error code: " << result1 << "\n";
      //     }
      //     std::cout << vertexId << " ";
      //   }
      //   std::cout << "]" << std::endl;
      // }
      // std::cout << "[TestTopoRelations] Time usage for FV: " << t.getElapsedTime() << " s.\n";

      // /* test triangle edge relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId triangleId = 0; triangleId < triangleNumber; triangleId++){
      //   SimplexId edgeId;
      //   for(SimplexId j = 0; j < 3; j++){
      //     int result1 = triangulation->getTriangleEdge(triangleId, j, edgeId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] triangleId " << triangleId << " Something wrong in getTriangleEdge()! Error code: " << result1 << "\n";
      //     }
      //   }
      // }
      // std::cout << "[TestTopoRelations] Time usage for FE: " << t.getElapsedTime() << " s.\n";

      // /* test triangle tetrahedron relationship */
      {
        t.reStart();
        // std::mutex ioMutex;
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumber_)
        #endif
        for(SimplexId triangleId = 0; triangleId < triangleNumber; triangleId++){
          SimplexId cellNum = triangulation->getTriangleStarNumber(triangleId);
          SimplexId cellId;
          // std::unique_lock<std::mutex> ioLock(ioMutex);
          // std::cout << "triangle " << triangleId << ": " << cellNum << " [";
          for(SimplexId j = 0; j < cellNum; j++){
            int result1 = triangulation->getTriangleStar(triangleId, j, cellId);
            if(result1){
                // std::cout << "[TestTopoRelations] triangleId " << triangleId << " Something wrong in getTriangleStar()! Error code: " << result1 << "\n";
            }
            // std::cout << cellId << " ";
          }
          // std::cout << "]" << std::endl;
          // ioLock.unlock();
        }
        std::cout << "[TestTopoRelations] Time usage for FT: " << t.getElapsedTime() << " s.\n";
      }

      /* test tetrahedron vertex relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId cellId = 0; cellId < cellNumber; cellId++){
      //   SimplexId vertexId;
      //   // std::cout << "cell " << cellId << ": ";
      //   for(SimplexId j = 0; j < verticesPerCell; j++){
      //     int result1 = triangulation->getCellVertex(cellId, j, vertexId);
      //     // std::cout << vertexId << " ";
      //     if(result1){
      //         std::cout << "[TestTopoRelations] cellId " << cellId << " Something wrong in getCellVertex()! Error code: " << result1 << "\n";
      //     }
      //   }
      //   // std::cout << std::endl;
      // }
      // std::cout << "[TestTopoRelations] Time usage for TV: " << t.getElapsedTime() << " s.\n";

      /* test tetrahedron edge relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId cellId = 0; cellId < cellNumber; cellId++){
      //   SimplexId edgeId;
      //   SimplexId edgesPerCell = triangulation->getCellEdgeNumber(cellId);
      //   std::cout << "cell " << cellId << ": " << edgesPerCell << " [";
      //   for(SimplexId j = 0; j < edgesPerCell; j++){
      //     int result1 = triangulation->getCellEdge(cellId, j, edgeId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] cellId " << cellId << " Something wrong in getCellEdge()! Error code: " << result1 << "\n";
      //     }
      //     std::cout << edgeId << " ";
      //   }
      //   std::cout << "]" << std::endl;
      // }
      // std::cout << "[TestTopoRelations] Time usage for TE: " << t.getElapsedTime() << " s.\n";

      /* test tetrahedron triangle relationship */
      // if(triangulation->getDimensionality() == 3){
      //   t.reStart();
      //   // #ifdef TTK_ENABLE_OPENMP
      //   // #pragma omp parallel for num_threads(threadNumber_)
      //   // #endif
      //   for(SimplexId cellId = 0; cellId < cellNumber; cellId++){
      //     SimplexId triangleId;
      //     SimplexId trianglesPerCell = triangulation->getCellTriangleNumber(cellId);
      //     std::cout << "cell " << cellId << ": " << trianglesPerCell << " [";
      //     for(SimplexId j = 0; j < trianglesPerCell; j++){
      //         int result1 = triangulation->getCellTriangle(cellId, j, triangleId);
      //         if(result1){
      //           std::cout << "[TestTopoRelations] cellId " << cellId << " Something wrong in getCellTriangle()! Error code: " << result1 << "\n";
      //         }
      //         std::cout << triangleId << " ";
      //     }
      //     std::cout << "]" << std::endl;
      //   }
      //   std::cout << "[TestTopoRelations] Time usage for TF: " << t.getElapsedTime() << " s.\n";
      // }

      std::cout << "[TestTopoRelations] Data-set (" << vertexNumber
          << " points) processed in "
          << tot.getElapsedTime() << " s." << std::endl;
      
      return 0;
    }
}