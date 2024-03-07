/// \ingroup base
/// \class ttk::Skeleton
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \date August 2015.
///
/// \brief %Skeletonprocessing package.
///
/// %Skeletonis a processing package that handles the 1-, 2, 3-skeletons
/// (vertices, edges, tetrahedra) of a triangulation.
/// \sa Triangulation

#pragma once

// base code includes
#include "DataTypes.h"
#include "CellArray.h"
#include "FlatJaggedArray.h"
#include <algorithm>
#include <array>

namespace ttk {

  class Skeleton {

  public:
    Skeleton() {}

    /**
     *  Functions from ThreeSkeleton class
     */
    /// Compute the list of cell-neighbors of each cell of a triangulation
    /// (unspecified behavior if the input mesh is not a triangulation).
    /// This implementation is fast only if you already have the triangle
    /// stars computed. Otherwise, please use
    /// Skeleton::buildCellNeighborsFromVertices instead.
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param cellNeighbors Output neighbor list. The size of this std::vector
    /// will be equal to the number of cells in the mesh. Each entry will be a
    /// std::vector listing the cell identifiers of the entry's cell's
    /// neighbors.
    /// \param triangleStars Optional list of triangle stars (list of
    /// 3-dimensional cells connected to each triangle). If nullptr, the
    /// function will compute this list anyway and free the related memory
    /// upon return. If not nullptr but pointing to an empty std::vector, the
    /// function will fill this empty std::vector (useful if this list needs
    /// to be used later on by the calling program). If not nullptr but pointing
    /// to a non-empty std::vector, this function will use this std::vector as
    /// internal triangle star list. If this std::vector is not empty but
    /// incorrect, the behavior is unspecified. \return Returns 0 upon success,
    /// negative values otherwise.
    int buildCellNeighborsFromTrianglesThreeSkeleton(const SimplexId &vertexNumber,
                                        const CellArray &cellArray,
                                        FlatJaggedArray &cellNeighbors,
                                        FlatJaggedArray *triangleStars
                                        = nullptr) const;

    /// Compute the list of cell-neighbors of each cell of a triangulation
    /// (unspecified behavior if the input mesh is not a triangulation).
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param cellNeighbors Output neighbor list. The size of this
    /// std::vector will be equal to the number of cells in the mesh. Each entry
    /// will be a std::vector listing the cell identifiers of the entry's cell's
    /// neighbors.
    /// \param vertexStars Optional list of vertex stars (list of
    /// 3-dimensional cells connected to each vertex). If nullptr, the
    /// function will compute this list anyway and free the related memory
    /// upon return. If not nullptr but pointing to an empty std::vector, the
    /// function will fill this empty std::vector (useful if this list needs
    /// to be used later on by the calling program). If not nullptr but pointing
    /// to a non-empty std::vector, this function will use this std::vector as
    /// internal vertex star list. If this std::vector is not empty but
    /// incorrect, the behavior is unspecified.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildCellNeighborsFromVerticesThreeSkeleton(const SimplexId &vertexNumber,
                                       const CellArray &cellArray,
                                       FlatJaggedArray &cellNeighbors,
                                       FlatJaggedArray *vertexStars
                                       = nullptr) const;
    
    /**
     * Functions from TwoSkeleton class 
     */
    /// Compute the list of cell-neighbors of each cell of a 2D triangulation
    /// (unspecified behavior if the input mesh is not a triangulation).
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param cellNeighbors Output neighbor list. The size of this std::vector
    /// will be equal to the number of cells in the mesh. Each entry will be a
    /// std::vector listing the cell identifiers of the entry's cell's
    /// neighbors.
    /// \param edgeStars Optional list of edge stars (list of
    /// 2-dimensional cells connected to each edge). If nullptr, the
    /// function will compute this list anyway and free the related memory
    /// upon return. If not nullptr but pointing to an empty std::vector, the
    /// function will fill this empty std::vector (useful if this list needs
    /// to be used later on by the calling program). If not nullptr but pointing
    /// to a non-empty std::vector, this function will use this std::vector as
    /// internal vertex star list. If this std::vector is not empty but
    /// incorrect, the behavior is unspecified.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildCellNeighborsFromEdgesTwoSkeleton(const CellArray &cellArray,
                                    FlatJaggedArray &cellNeighbors,
                                    const FlatJaggedArray &edgeStars) const;

    /// Compute the list of cell-neighbors of each cell of a 2D triangulation
    /// (unspecified behavior if the input mesh is not a triangulation).
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param cellNeighbors Output neighbor list. The size of this std::vector
    /// will be equal to the number of cells in the mesh. Each entry will be a
    /// std::vector listing the cell identifiers of the entry's cell's
    /// neighbors.
    /// \param vertexStars Optional list of vertex stars (list of
    /// 2-dimensional cells connected to each vertex). If nullptr, the
    /// function will compute this list anyway and free the related memory
    /// upon return. If not nullptr but pointing to an empty std::vector, the
    /// function will fill this empty std::vector (useful if this list needs
    /// to be used later on by the calling program). If not nullptr but pointing
    /// to a non-empty std::vector, this function will use this std::vector as
    /// internal vertex star list. If this std::vector is not empty but
    /// incorrect, the behavior is unspecified.
    ///  \return Returns 0 upon success, negative values otherwise.
    int buildCellNeighborsFromVerticesTwoSkeleton(const SimplexId &vertexNumber,
                                       const CellArray &cellArray,
                                       FlatJaggedArray &cellNeighbors,
                                       FlatJaggedArray *vertexStars
                                       = nullptr) const;

    /// Compute the list of triangles connected to each edge for 3D
    /// triangulations (unspecified behavior if the input mesh is not a
    /// triangulation).
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param edgeTriangleList Output edge triangle list. The size of this
    /// std::vector will be equal to the number of edges in the triangulation.
    /// Each entry will be a std::vector listing the triangle identifiers for
    /// each triangle connected to the entry's edge.
    /// \param edgeList Edge list (list of std::pairs of vertex
    /// identifiers). If this std::vector is not empty but incorrect,
    /// the behavior is unspecified.
    /// \param triangleEdgeList Optional output triangle edge list (list of
    /// std::vectors of edges identifiers per triangle). If nullptr, the
    /// function will compute this list anyway and free the related memory upon
    /// return. If not nullptr but pointing to an empty std::vector, the
    /// function will fill this empty std::vector (useful if this list needs to
    /// be used later on by the calling program). If not nullptr but pointing to
    /// a non-empty std::vector, this function will use this std::vector as
    /// internal edge list. If this std::vector is not empty but incorrect, the
    /// behavior is unspecified.
    int
      buildEdgeTrianglesTwoSkeleton(const SimplexId &vertexNumber,
                         const CellArray &cellArray,
                         FlatJaggedArray &edgeTriangleList,
                         const std::vector<std::array<SimplexId, 2>> &edgeList,
                         std::vector<std::array<SimplexId, 3>> *triangleEdgeList
                         = nullptr) const;

    /// Compute the list of triangles of a triangulation represented by a
    /// vtkUnstructuredGrid object. Unspecified behavior if the input mesh is
    /// not a valid triangulation.
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param triangleList Optional output triangle list (each entry is the
    /// ordered std::vector of the vertex identifiers of the entry's triangle).
    /// \param triangleStars Optional output for triangle tet-adjacency (for
    /// each triangle, list of its adjacent tetrahedra).
    /// \return Returns 0 upon success, negative values otherwise.
    int buildTriangleListTwoSkeleton(
      const SimplexId &vertexNumber,
      const CellArray &cellArray,
      std::vector<std::array<SimplexId, 3>> *triangleList = nullptr,
      FlatJaggedArray *triangleStars = nullptr,
      std::vector<std::array<SimplexId, 4>> *cellTriangleList = nullptr) const;

    /// Compute the list of edges connected to each triangle for 3D
    /// triangulations (unspecified behavior if the input mesh is not a
    /// 3D triangulation).
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param triangleEdgeList Output triangle edge list. The size of this
    /// std::vector will be equal to the number of triangles in the
    /// triangulation. Each entry will be a std::vector listing the edge
    /// identifiers for each edge connected to the entry's triangle.
    /// \param edgeList Edge list (list of std::pairs of vertex
    /// identifiers). If this std::vector is not empty but incorrect,
    /// the behavior is unspecified.
    /// \param vertexEdgeList Optional output vertex edge list (list of edge
    /// identifiers for each vertex). If nullptr, the function will compute this
    /// list anyway and free the related memory upon return. If not nullptr but
    /// pointing to an empty std::vector, the function will fill this empty
    /// std::vector (useful if this list needs to be used later on by the
    /// calling program). If not nullptr but pointing to a non-empty
    /// std::vector, this function will use this std::vector as internal vertex
    /// edge list. If this std::vector is not empty but incorrect, the behavior
    /// is unspecified.
    /// \param triangleList Optional output triangle list (list of std::vectors
    /// of vertex identifiers). If nullptr, the function will compute this list
    /// anyway and free the related memory upon return. If not nullptr but
    /// pointing to an empty std::vector, the function will fill this empty
    /// std::vector (useful if this list needs to be used later on by the
    /// calling program). If not nullptr but pointing to a non-empty
    /// std::vector, this function will use this std::vector as internal
    /// triangle list. If this std::vector is not empty but incorrect, the
    /// behavior is unspecified.
    /// \param triangleStarList Optional output triangle star list (list of
    /// tetrahedron identifiers for each triangle). If nullptr, the function
    /// will compute this list anyway and free the related memory upon return.
    /// If not nullptr but pointing to an empty std::vector, the function will
    /// fill this empty std::vector (useful if this list needs to be used later
    /// on by the calling program). If not nullptr but pointing to a non-empty
    /// std::vector, this function will use this std::vector as internal
    /// triangle star list. If this std::vector is not empty but incorrect, the
    /// behavior is unspecified.
    /// \param cellTriangleList Optional output cell triangle list (list of
    /// triangle identifiers for each tetrahedron). If nullptr, the function
    /// will compute this list anyway and free the related memory upon return.
    /// If not nullptr but pointing to an empty std::vector, the function will
    /// fill this empty std::vector (useful if this list needs to be used later
    /// on by the calling program). If not nullptr but pointing to a non-empty
    /// std::vector, this function will use this std::vector as internal cell
    /// triangle list. If this std::vector is not empty but incorrect, the
    /// behavior is unspecified.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildTriangleEdgeListTwoSkeleton(
      const SimplexId &vertexNumber,
      const CellArray &cellArray,
      std::vector<std::array<SimplexId, 3>> &triangleEdgeList,
      const std::vector<std::array<SimplexId, 2>> &edgeList,
      FlatJaggedArray *vertexEdgeList = nullptr,
      std::vector<std::array<SimplexId, 3>> *triangleList = nullptr,
      FlatJaggedArray *triangleStarList = nullptr,
      std::vector<std::array<SimplexId, 4>> *cellTriangleList = nullptr) const;

    /// Compute the links of triangles in a 3D triangulation.
    /// \param triangleList Input triangle list. The number of entries of this
    /// list is equal to the number of triangles in the triangulation. Each
    /// entry lists the vertex identifiers of the corresponding triangle.
    /// \param triangleStar Input triangle star list. The number of entries of
    /// this list is equal to the number of triangles in the triangulation. Each
    /// entry lists the identifiers of the tetrahedra which are the co-faces of
    /// the corresponding triangle.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param triangleLinks Output triangle link list. The number of entries of
    /// this list is equal to the number of triangles in the triangulation. Each
    /// entry lists the identifiers of the vertices in the link of the
    /// corresponding triangle.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildTriangleLinksTwoSkeleton(
      const std::vector<std::array<SimplexId, 3>> &triangeList,
      const FlatJaggedArray &triangleStars,
      const CellArray &cellArray,
      FlatJaggedArray &triangleLinks) const;

    /// Compute the list of triangles connected to each vertex for 3D
    /// triangulations (unspecified behavior if the input mesh is not a
    /// triangulation).
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param triangleList Input triangle list (list of std::vectors of
    /// vertex identifiers).
    /// \param vertexTriangleList Output vertex triangle list (list of
    /// std::vectors of triangle identifiers).
    int buildVertexTrianglesTwoSkeleton(
      const SimplexId &vertexNumber,
      const std::vector<std::array<SimplexId, 3>> &triangleList,
      FlatJaggedArray &vertexTriangles) const;

    
    /**
     * Functions from OneSkeleton
    */
    /// Compute the link of each edge of a 2D triangulation (unspecified
    /// behavior if the input mesh is not a valid triangulation).
    /// \param edgeList List of edges. The size of this std::vector
    /// should be equal to the number of edges in the triangulation. Each
    /// entry is a std::pair of vertex identifiers.
    /// \param edgeStars List of edge stars. The size of this std::vector
    /// should be equal to the number of edges. Each entry is a std::vector of
    /// triangle identifiers.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param edgeLinks Output edge links. The size of this std::vector will be
    /// equal to the number of edges in the triangulation. Each entry will be a
    /// std::vector listing the vertices in the link of the corresponding
    /// vertex.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildEdgeLinksOneSkeleton(const std::vector<std::array<SimplexId, 2>> &edgeList,
                       const FlatJaggedArray &edgeStars,
                       const CellArray &cellArray,
                       FlatJaggedArray &edgeLinks) const;

    /// Compute the link of each edge of a 3D triangulation (unspecified
    /// behavior if the input mesh is not a valid triangulation).
    /// \param edgeList List of edges. The size of this std::vector
    /// should be equal to the number of edges in the triangulation. Each
    /// entry is a std::pair of vertex identifiers.
    /// \param edgeStars List of edge stars. The size of this std::vector
    /// should be equal to the number of edges. Each entry is a std::vector of
    /// tetrahedron identifiers.
    /// \param cellEdges List of celle edges. The size of this std::vector
    /// should be equal to the number of tetrahedra in the triangulation. Each
    /// entry is a std::vector of edge identifiers.
    /// \param edgeLinks Output edge links. The size of this std::vector
    /// will be equal to the number of edges in the triangulation. Each
    /// entry will be a std::vector listing the vertices in the link of the
    /// corresponding vertex.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildEdgeLinksOneSkeleton(const std::vector<std::array<SimplexId, 2>> &edgeList,
                       const FlatJaggedArray &edgeStars,
                       const std::vector<std::array<SimplexId, 6>> &cellEdges,
                       FlatJaggedArray &edgeLinks) const;

    /// Compute the list of edges of a valid triangulation.
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param edgeList Output edge list (each entry is an ordered
    /// std::array of vertex identifiers).
    /// \param edgeStars Output for edge cell adjacency (for each
    /// edge, a list of adjacent cells)
    /// \param cellEdgeList Output for cell edges: per cell, the list
    /// of its edges identifiers
    /// \return Returns 0 upon success, negative values otherwise.
    template <std::size_t n>
    int
      buildEdgeListOneSkeleton(const SimplexId &vertexNumber,
                    const CellArray &cellArray,
                    std::vector<std::array<SimplexId, 2>> &edgeList,
                    FlatJaggedArray &edgeStars,
                    std::vector<std::array<SimplexId, n>> &cellEdgeList) const;
    
    /**
     * Functions from ZeroSkeleton
    */

    /// Compute the list of edges connected to each vertex of a triangulation.
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param edgeList List of edges. Each entry is represented by the
    /// ordered std::pair of identifiers of the entry's edge's vertices.
    /// \param vertexEdges Output vertex links. The size of this std::vector
    /// will be equal to the number of vertices in the mesh. Each entry will
    /// be a std::vector listing the identifiers of the edges connected to the
    /// entry's vertex.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildVertexEdgesZeroSkeleton(const SimplexId &vertexNumber,
                         const std::vector<std::array<SimplexId, 2>> &edgeList,
                         FlatJaggedArray &vertexEdges) const;

    /// Compute the link of each vertex of a 2D triangulation (unspecified
    /// behavior if the input mesh is not a valid triangulation).
    /// \param vertexStars List of vertex stars. The size of this std::vector
    /// should be equal to the number of vertices in the triangulation. Each
    /// entry is a std::vector listing the identifiers of triangles.
    /// \param cellEdges List of cell edges. The size of this std::vector
    /// should be equal to the number of triangles. Each entry is a std::vector
    /// of identifiers of edges.
    /// \param vertexLinks Output vertex links. The size of this std::vector
    /// will be equal to the number of vertices in the triangulation. Each
    /// entry will be a std::vector listing the edges in the link of the
    /// corresponding vertex.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildVertexLinksZeroSkeleton(const FlatJaggedArray &vertexStars,
                         const std::vector<std::array<SimplexId, 3>> &cellEdges,
                         const std::vector<std::array<SimplexId, 2>> &edgeList,
                         FlatJaggedArray &vertexLinks) const;

    /// Compute the link of each vertex of a 3D triangulation (unspecified
    /// behavior if the input mesh is not a valid triangulation).
    /// \param vertexStars List of vertex stars. The size of this std::vector
    /// should be equal to the number of vertices in the triangulation. Each
    /// entry is a std::vector listing the identifiers of tetrahedra.
    /// \param cellTriangles List of cell triangles. The size of this
    /// std::vector
    /// should be equal to the number of tetrahedra. Each entry is a
    /// std::vector
    /// of identifiers of triangles.
    /// \param vertexLinks Output vertex links. The size of this std::vector
    /// will be equal to the number of vertices in the triangulation. Each
    /// entry will be a std::vector listing the triangles in the link of the
    /// corresponding vertex.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildVertexLinksZeroSkeleton(
      const FlatJaggedArray &vertexStars,
      const std::vector<std::array<SimplexId, 4>> &cellTriangles,
      const std::vector<std::array<SimplexId, 3>> &triangleList,
      FlatJaggedArray &vertexLinks) const;

    /// Compute the list of neighbors of each vertex of a triangulation.
    /// Unspecified behavior if the input mesh is not a valid triangulation).
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices ids
    /// of each cell.
    /// \param vertexNeighbors Output neighbor list. The size of this
    /// std::vector will be equal to the number of vertices in the mesh. Each
    /// entry will be std::vector listing the vertex identifiers of the entry's
    /// vertex' neighbors.
    /// \param edgeList List of edges. If this std::vector is not
    /// empty but incorrect, the behavior is unspecified.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildVertexNeighborsZeroSkeleton(
      const SimplexId &vertexNumber,
      FlatJaggedArray &vertexNeighbors,
      const std::vector<std::array<SimplexId, 2>> &edgeList) const;

    /// Compute the star of each vertex of a triangulation. Unspecified
    /// behavior if the input mesh is not a valid triangulation.
    /// \param vertexNumber Number of vertices in the triangulation.
    /// \param cellArray Cell container allowing to retrieve the vertices
    /// of each cell.
    /// \param vertexStars Output vertex stars. The size of this std::vector
    /// will be equal to the number of vertices in the mesh. Each entry will
    /// be a std::vector listing the identifiers of the maximum-dimensional
    /// cells (3D: tetrahedra, 2D: triangles, etc.) connected to the entry's
    /// vertex.
    /// \return Returns 0 upon success, negative values otherwise.
    int buildVertexStarsZeroSkeleton(const SimplexId &vertexNumber,
                         const CellArray &cellArray,
                         FlatJaggedArray &vertexStars) const;

  };
} // namespace ttk
