#pragma once

#include "AbstractTriangulation.h"
#include "Timer.h"
#include <iostream>

namespace ttk {
    class TestTopoRelations {
    protected:
      int threadNumber_;
    public:
      TestTopoRelations() { threadNumber_ = 10; }
      ~TestTopoRelations() {}

      int execute(AbstractTriangulation *triangulation) const;
      inline void setThreadNumber(int tnum) { threadNumber_ = tnum; }
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
          // triangulation->preconditionVertexEdges();
          triangulation->preconditionVertexNeighbors();
          // triangulation->preconditionVertexStars();
          // triangulation->preconditionVertexTriangles();
          // edge related relations
          // triangulation->preconditionEdgeStars();
          // triangulation->preconditionEdgeTriangles();
          // triangle related relations
          // triangulation->preconditionTriangleEdges();
          // triangulation->preconditionTriangleStars();
          // cell related relations
          // triangulation->preconditionCellEdges();
          // triangulation->preconditionCellNeighbors();
          // triangulation->preconditionCellTriangles();
          // links
          // triangulation->preconditionVertexLinks();
          // triangulation->preconditionEdgeLinks();
          // triangulation->preconditionTriangleLinks();

          std::cout << "[TestTopoRelations] Time usage for preconditioning: " << t.getElapsedTime() << " s.\n";
        }
        return 0;
      }
    };
    
    int TestTopoRelations::execute(AbstractTriangulation *triangulation) const {

      std::cout << "[TestTopoRelations] Dimension: " << triangulation->getDimensionality() << std::endl;

      SimplexId vertexNumber = triangulation->getNumberOfVertices();
      // SimplexId cellNumber = triangulation->getNumberOfCells();
      // SimplexId edgeNumber = triangulation->getNumberOfEdges();
      // SimplexId triangleNumber = triangulation->getNumberOfTriangles();

      // std::cout << "[TestTopoRelations] vertex num: " << vertexNumber << ", edge num: "
      //     << edgeNumber << ", triangle num: " << triangleNumber << ", cell num: " << 
      //     cellNumber << std::endl;

      SimplexId verticesPerCell = triangulation->getCellVertexNumber(0);
      // SimplexId edgesPerCell = triangulation->getCellEdgeNumber(0);
      // SimplexId trianglesPerCell = triangulation->getCellTriangleNumber(0);

      Timer t, tot;

      // t.reStart();
      // triangulation->getVertexEdgeNumber(0);
      // std::cout << "[TestTopoRelations] Time usage for VE: " << t.getElapsedTime() << " s.\n";
      
      // t.reStart();
      // triangulation->getVertexNeighborNumber(0);
      // std::cout << "[TestTopoRelations] Time usage for VV: " << t.getElapsedTime() << " s.\n";

      // t.reStart();
      // triangulation->getVertexStarNumber(0);
      // std::cout << "[TestTopoRelations] Time usage for VT: " << t.getElapsedTime() << " s.\n";

      // Vertex neighbor relation
      t.reStart();
      #pragma omp parallel for num_threads(threadNumber_)
      for(SimplexId vid = 0; vid < vertexNumber; vid++) {
        SimplexId neighborNum = triangulation->getVertexNeighborNumber(vid);
        // std::cout << "vertex " << vid << ": " << neighborNum << " [";
        SimplexId neighborId;
        for(int j = 0; j < neighborNum; j++) {
          triangulation->getVertexNeighbor(vid, j, neighborId);
          // std::cout << neighborId << " ";
        }
        // std::cout << "]" << std::endl;
      }
      std::cout << "[TestTopoRelations] Time usage for VN: " << t.getElapsedTime()
                << " s." << std::endl;

      /* test vertex edge relationship */
      // t.reStart();
      // #ifdef TTK_ENABLE_OPENMP
      // #pragma omp parallel for num_threads(threadNumber_)
      // #endif
      // for(SimplexId vertexId = 0; vertexId < vertexNumber; vertexId++){
      //   SimplexId edgeNum = triangulation->getVertexEdgeNumber(vertexId);
      //   // std::cout << "vertex " << vertexId << ": [";
      //   SimplexId edgeId;
      //   for(SimplexId j = 0; j < edgeNum; j++){
      //     int result1 = triangulation->getVertexEdge(vertexId, j, edgeId);
      //     if(result1){
      //         std::cerr << "[TestTopoRelations] vertexId " << vertexId << " Something wrong in getVertexEdge()! Error code: " << result1 << "\n";
      //     }
      //     // std::cout << edgeId << " ";
      //   }
      //   // std::cout << "]" << std::endl;
      // }
      // std::cout << "[TestTopoRelations] Time usage for VE: " << t.getElapsedTime() << " s.\n";

      /* test vertex triangle relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId vertexId = 0; vertexId < vertexNumber; vertexId++){
      //   SimplexId triangleNum = triangulation->getVertexTriangleNumber(vertexId);
      //   // std::cout << "vertex " << vertexId << ": " << triangleNum << " [";
      //   SimplexId triangleId;
      //   for(SimplexId j = 0; j < triangleNum; j++){
      //     int result1 = triangulation->getVertexTriangle(vertexId, j, triangleId);
      //     if(result1){
      //         std::cerr << "[TestTopoRelations] vertexId " << vertexId << " Something wrong in getVertexTriangle()! Error code: " << result1 << "\n";
      //     }
      //     // std::cout << triangleId << " ";
      //   }
      //   // std::cout << "]" << std::endl;
      // }
      // std::cout << "[TestTopoRelations] Time usage for VF: " << t.getElapsedTime() << " s.\n";
      
      /* test vertex tetrahedron relationship */
      // t.reStart();
      // #pragma omp parallel for num_threads(threadNumber_)
      // for(SimplexId vertexId = 0; vertexId < vertexNumber; vertexId++){
      //   SimplexId cellNum = triangulation->getVertexStarNumber(vertexId);
      //   SimplexId cellId;
      //   for(SimplexId j = 0; j < cellNum; j++){
      //     int result1 = triangulation->getVertexStar(vertexId, j, cellId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] vertexId " << vertexId << " Something wrong in getVertexStar()! Error code: " << result1 << "\n";
      //     }
      //   }
      // }
      // std::cout << "[TestTopoRelations] Time usage for VT: " << t.getElapsedTime() << " s.\n";

      // /* test edge vertex relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId edgeId = 0; edgeId < edgeNumber; edgeId++){
      //   SimplexId vertexId;
      //   for(SimplexId j = 0; j < 2; j++){
      //     int result1 = triangulation->getEdgeVertex(edgeId, j, vertexId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] edgeId " << edgeId << " Something wrong in getEdgeVertex()! Error code: " << result1 << "\n";
      //     }
      //   }
      // }
      // std::cout << "[TestTopoRelations] Time usage for EV: " << t.getElapsedTime() << " s.\n";


      // /* test edge triangle relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId edgeId = 0; edgeId < edgeNumber; edgeId++){
      //   SimplexId triangleNum = triangulation->getEdgeTriangleNumber(edgeId);
      //   SimplexId triangleId;
      //   for(SimplexId j = 0; j < triangleNum; j++){
      //     int result1 = triangulation->getEdgeTriangle(edgeId, j, triangleId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] edgeId " << edgeId << " Something wrong in getEdgeTriangle()! Error code: " << result1 << "\n";
      //     }
      //   }
      // }
      // std::cout << "[TestTopoRelations] Time usage for EF: " << t.getElapsedTime() << " s.\n";
      
      // /* test edge tetrahedron relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId edgeId = 0; edgeId < edgeNumber; edgeId++){
      //   SimplexId cellNum = triangulation->getEdgeStarNumber(edgeId);
      //   SimplexId cellId;
      //   for(SimplexId j = 0; j < cellNum; j++){
      //     int result1 = triangulation->getEdgeStar(edgeId, j, cellId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] edgeId " << edgeId << " Something wrong in getEdgeStar()! Error code: " << result1 << "\n";
      //     }
      //   }
      // }
      // std::cout << "[TestTopoRelations] Time usage for ET: " << t.getElapsedTime() << " s.\n";

      // /* test triangle vertex relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId triangleId = 0; triangleId < triangleNumber; triangleId++){
      //   SimplexId vertexId;
      //   for(SimplexId j = 0; j < 3; j++){
      //   int result1 = triangulation->getTriangleVertex(triangleId, j, vertexId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] triangleId " << triangleId << " Something wrong in getTriangleVertex()! Error code: " << result1 << "\n";
      //     }
      //   }
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
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId triangleId = 0; triangleId < triangleNumber; triangleId++){
      //   SimplexId cellNum = triangulation->getTriangleStarNumber(triangleId);
      //   SimplexId cellId;
      //   for(SimplexId j = 0; j < cellNum; j++){
      //     int result1 = triangulation->getTriangleStar(triangleId, j, cellId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] triangleId " << triangleId << " Something wrong in getTriangleStar()! Error code: " << result1 << "\n";
      //     }
      //   }
      // }
      // std::cout << "[TestTopoRelations] Time usage for FT: " << t.getElapsedTime() << " s.\n";

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

      // /* test tetrahedron edge relationship */
      // t.reStart();
      // // #ifdef TTK_ENABLE_OPENMP
      // // #pragma omp parallel for num_threads(threadNumber_)
      // // #endif
      // for(SimplexId cellId = 0; cellId < cellNumber; cellId++){
      //   SimplexId edgeId;
      //   for(SimplexId j = 0; j < edgesPerCell; j++){
      //     int result1 = triangulation->getCellEdge(cellId, j, edgeId);
      //     if(result1){
      //         std::cout << "[TestTopoRelations] cellId " << cellId << " Something wrong in getCellEdge()! Error code: " << result1 << "\n";
      //     }
      //   }
      // }
      // std::cout << "[TestTopoRelations] Time usage for TE: " << t.getElapsedTime() << " s.\n";

      // /* test tetrahedron triangle relationship */
      // if(triangulation->getDimensionality() == 3){
      //   t.reStart();
      //   // #ifdef TTK_ENABLE_OPENMP
      //   // #pragma omp parallel for num_threads(threadNumber_)
      //   // #endif
      //   for(SimplexId cellId = 0; cellId < cellNumber; cellId++){
      //     SimplexId triangleId;
      //     for(SimplexId j = 0; j < trianglesPerCell; j++){
      //         int result1 = triangulation->getCellTriangle(cellId, j, triangleId);
      //         if(result1){
      //           std::cout << "[TestTopoRelations] cellId " << cellId << " Something wrong in getCellTriangle()! Error code: " << result1 << "\n";
      //         }
      //     }
      //   }
      //   std::cout << "[TestTopoRelations] Time usage for TF: " << t.getElapsedTime() << " s.\n";
      // }

      std::cout << "[TestTopoRelations] Data-set (" << vertexNumber
          << " points) processed in "
          << tot.getElapsedTime() << " s." << std::endl;
      
      return 0;
    }
}