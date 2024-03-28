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

/**
 * @brief Force the compiler to use the function/method parameter.
 *
 * Some function/method parameters are used under some #ifdef
 * preprocessor conditions. This macro introduce a no-op statement
 * that uses those parameters, to silence compiler warnings in the
 * #else blocks. It can be inserted anywhere in the function body.
 */
#define TTK_FORCE_USE(x) (void)(x)

namespace ttk {
  /// \brief Identifier type for simplices of any dimension.
  using LongSimplexId = long long int;

  /// \brief Identifier type for simplices of any dimension.
#ifdef TTK_ENABLE_64BIT_IDS
  using SimplexId = long long int;
#else
  using SimplexId = int;
#endif

  /// \brief Identifier type for threads (i.e. with OpenMP).
  using ThreadId = int;

  /// \brief Identifier type for tasks (i.e. with OpenMP).
  using TaskId = int;

  /// default name for mask scalar field
  const char MaskScalarFieldName[] = "ttkMaskScalarField";

  /// default name for vertex scalar field
  const char VertexScalarFieldName[] = "ttkVertexScalarField";

  /// default name for offset scalar field
  const char OffsetScalarFieldName[] = "ttkOffsetScalarField";

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

  // default names for the Morse-Smale complex
  const char MorseSmaleCellDimensionName[] = "CellDimension";
  const char MorseSmaleCellIdName[] = "CellId";
  const char MorseSmaleBoundaryName[] = "IsOnBoundary";
  const char MorseSmaleManifoldSizeName[] = "ManifoldSize";
  const char MorseSmaleSourceIdName[] = "SourceId";
  const char MorseSmaleDestinationIdName[] = "DestinationId";
  const char MorseSmaleSeparatrixIdName[] = "SeparatrixId";
  const char MorseSmaleSeparatrixTypeName[] = "SeparatrixType";
  const char MorseSmaleSeparatrixMaximumName[] = "SeparatrixFunctionMaximum";
  const char MorseSmaleSeparatrixMinimumName[] = "SeparatrixFunctionMinimum";
  const char MorseSmaleSeparatrixDifferenceName[]
    = "SeparatrixFunctionDifference";
  const char MorseSmaleCriticalPointsOnBoundaryName[]
    = "NumberOfCriticalPointsOnBoundary";
  const char MorseSmaleAscendingName[] = "AscendingManifold";
  const char MorseSmaleDescendingName[] = "DescendingManifold";
  const char MorseSmaleManifoldName[] = "MorseSmaleManifold";

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
    // edges and triangles computation
    IEList, // internalEdgeList -- 9
    IFList, // internalTriangleList -- 10
    IBEList, // internalEdgeList and boundaryEdgeList -- 11
    IBFList, // internalTriangleList and boundaryTriangleList -- 12
    IBCList, // internalBoundaryCellList -- 13
    // require external maps
    VERelation, // vertexEdge -- 14
    VFRelation, // vertexTriangle -- 15
    EFRelation, // edgeTriangle -- 16
    FERelation, // triangleEdge -- 17
    TERelation, // cellEdge -- 18
    TFRelation, // cellTriangle -- 19
    VLRelation, // vertexLink -- 20
    ELRelation, // edgeLink -- 21
    FLRelation, // triangleLink -- 22
    BVRelation, // boundaryVertex -- 23
    BERelation, // boundaryEdge -- 24
    BFRelation, // boundaryTriangle -- 25
    RelationNum // total number of relations
  };

}
