/**
 * Common data types that are used in the source code. 
 * Author: Guoxi Liu <guoxil@clemson.edu>
 *         Guillaume Favelier <guillaume.favelier@lip6.fr>
 * Date: 2023/09/20
 * 
*/

#pragma once

/**
 * @brief Mark function/method parameters that are not used in the
 * function body at all.
 *
 * This is the TTK version of vtkNotUsed. It silences compiler
 * warnings about unused parameters when those parameters are not used
 * in the current function/method body but still necessary in the
 * function/method signature (for instance when implementing a
 * particular API with inheritance).
 *
 * Basically, it removes the name of the parameter at its definition
 * site (similar to commenting the parameter name).
 */
#define ttkNotUsed(x)

namespace ttk {
  using LongSimplexId = long long int;
  using SimplexId = int;

  /// default value for critical index
  enum class CriticalType {
    Local_minimum = 0,
    Saddle1,
    Saddle2,
    Local_maximum,
    Degenerate,
    Regular
  };
  /// number of different critical types
  const int CriticalTypeNumber = 6;


  /// relation names for TopoClsuter
  enum class RelationType {
    BlankRelation, // Empty -- 0
    VVRelation, // vertexNeighbor -- 1
    VTRelation, // vertexStar -- 2
    EVRelation, // edgeVertex -- 3
    ETRelation, // edgeStar -- 4
    FVRelation, // triangleVertex -- 5
    FTRelation, // triangleStar -- 6
    TVRelation, // cellVertex -- 7
    TTRelation, // cellNeighbor -- 8
    // external maps computation
    IEMapComp, // internalEdgeMap -- 9
    IFMapComp, // internalTriangleMap -- 10
    IEFMapComp, // intenralEdgeMap and internalTriangleMap -- 11
    IFBMapComp, // internalTriangleMap and boundaryCell -- 12
    // require external maps
    VERelation, // vertexEdge -- 13
    VFRelation, // vertexTriangle -- 14
    EFRelation, // edgeTriangle -- 15
    FERelation, // triangleEdge -- 16
    TERelation, // cellEdge -- 17
    TFRelation, // cellTriangle -- 18
    VLRelation, // vertexLink -- 19
    ELRelation, // edgeLink -- 20
    FLRelation, // triangleLink -- 21
    BVRelation, // boundaryVertex -- 22
    BERelation, // boundaryEdge -- 23
    BFRelation, // boundaryTriangle -- 24
    RelationNum // total number of relations
  };

}
