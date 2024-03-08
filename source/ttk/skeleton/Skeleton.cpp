#include <Skeleton.h>
#include <Timer.h>
#include <boost/container/small_vector.hpp>

using namespace ttk;


int Skeleton::buildCellNeighborsFromTrianglesThreeSkeleton(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  FlatJaggedArray &cellNeighbors,
  FlatJaggedArray *triangleStars) const {

  auto localTriangleStars = triangleStars;
  FlatJaggedArray defaultTriangleStars{};
  if(!localTriangleStars) {
    localTriangleStars = &defaultTriangleStars;
  }

  if(localTriangleStars->empty()) {
    this->buildTriangleListTwoSkeleton(
      vertexNumber, cellArray, nullptr, localTriangleStars);
  }

  Timer t;

  std::cout << "Building cell neighbors..." << std::endl;

  const SimplexId cellNumber = cellArray.getNbCells();
  const SimplexId triangleNumber = localTriangleStars->subvectorsNumber();
  std::vector<SimplexId> offsets(cellNumber + 1);
  // number of neighbors processed per cell
  std::vector<SimplexId> neighborsId(cellNumber);

  for(SimplexId i = 0; i < triangleNumber; i++) {
    if(localTriangleStars->size(i) == 2) {
      // tetra cells in triangle i's star
      const auto cs0 = localTriangleStars->get(i, 0);
      const auto cs1 = localTriangleStars->get(i, 1);
      offsets[cs0 + 1]++;
      offsets[cs1 + 1]++;
    }
  }

  // compute partial sum of number of neighbors per vertex
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat neighbors vector
  std::vector<SimplexId> neighbors(offsets.back());

  // fill flat neighbors vector using offsets and neighbors count vectors
  for(SimplexId i = 0; i < triangleNumber; i++) {
    if(localTriangleStars->size(i) == 2) {
      // tetra cells in triangle i's star
      const auto cs0 = localTriangleStars->get(i, 0);
      const auto cs1 = localTriangleStars->get(i, 1);
      neighbors[offsets[cs0] + neighborsId[cs0]] = cs1;
      neighborsId[cs0]++;
      neighbors[offsets[cs1] + neighborsId[cs1]] = cs0;
      neighborsId[cs1]++;
    }
  }

  // fill FlatJaggedArray struct
  cellNeighbors.setData(std::move(neighbors), std::move(offsets));

  std::cout << "Built " << cellNumber << " cell neighbors " << t.getElapsedTime() << " s" << std::endl;

  // ethaneDiol.vtu, 8.7Mtets, vger (4coresHT)
  // 1 thread: 9.80 s
  // 4 threads: 14.18157 s

  // ethaneDiol.vtu, 8.7Mtets, richard (8coresHT)
  // 1 thread: 9.6763 s
  // 4 threads: 16.7489 s

  // ethaneDiol.vtu, 8.7Mtets, vger (4coresHT), VTK implementation
  // 1 thread: 14.9023 s
  // 4 threads: 8.58048 s

  // ethaneDiol.vtu, 8.7Mtets, richard (8coresHT), VTK implementation
  // 1 thread: 15.1387  s
  // 8 threads: 5.38286 s

  return 0;
}

int Skeleton::buildCellNeighborsFromVerticesThreeSkeleton(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  FlatJaggedArray &cellNeighbors,
  FlatJaggedArray *vertexStars) const {

  // TODO: ASSUME uniform mesh here!
  if(cellArray.getNbCells() && cellArray.getCellVertexNumber(0) == 3) {
    return this->buildCellNeighborsFromVerticesTwoSkeleton(
      vertexNumber, cellArray, cellNeighbors, vertexStars);
  }

  if(cellArray.getNbCells() && cellArray.getCellVertexNumber(0) <= 2) {
    // 1D
    std::cerr << "buildCellNeighborsFromVertices in 1D:" << std::endl;
    std::cerr << "Not implemented! TODO?!" << std::endl;
    return -1;
  }

  auto localVertexStars = vertexStars;
  FlatJaggedArray defaultVertexStars{};

  if(!localVertexStars) {
    localVertexStars = &defaultVertexStars;
  }

  if(localVertexStars->empty()) {
    this->buildVertexStarsZeroSkeleton(vertexNumber, cellArray, *localVertexStars);
  }

  Timer t;

  std::cout << "Building cell neighnors..." << std::endl;

  const SimplexId cellNumber = cellArray.getNbCells();
  using boost::container::small_vector;
  // for each cell/tetra, a vector of neighbors
  std::vector<small_vector<SimplexId, 4>> neighbors(cellNumber);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(SimplexId cid = 0; cid < cellNumber; cid++) {
    const SimplexId nbVertCell = cellArray.getCellVertexNumber(cid);

    // go triangle by triangle
    for(SimplexId j = 0; j < nbVertCell; j++) {

      SimplexId v0 = cellArray.getCellVertex(cid, j);
      SimplexId v1 = cellArray.getCellVertex(cid, (j + 1) % nbVertCell);
      SimplexId v2 = cellArray.getCellVertex(cid, (j + 2) % nbVertCell);

      // perform an intersection of the 3 (sorted) star lists
      SimplexId pos0 = 0, pos1 = 0, pos2 = 0;
      SimplexId intersection = -1;

      while(pos0 < localVertexStars->size(v0)
            && pos1 < localVertexStars->size(v1)
            && pos2 < localVertexStars->size(v2)) {

        SimplexId biggest = localVertexStars->get(v0, pos0);
        if(localVertexStars->get(v1, pos1) > biggest) {
          biggest = localVertexStars->get(v1, pos1);
        }
        if(localVertexStars->get(v2, pos2) > biggest) {
          biggest = localVertexStars->get(v2, pos2);
        }

        for(SimplexId l = pos0; l < localVertexStars->size(v0); l++) {
          if(localVertexStars->get(v0, l) < biggest) {
            pos0++;
          } else {
            break;
          }
        }
        for(SimplexId l = pos1; l < localVertexStars->size(v1); l++) {
          if(localVertexStars->get(v1, l) < biggest) {
            pos1++;
          } else {
            break;
          }
        }
        for(SimplexId l = pos2; l < localVertexStars->size(v2); l++) {
          if(localVertexStars->get(v2, l) < biggest) {
            pos2++;
          } else {
            break;
          }
        }

        if(pos0 < localVertexStars->size(v0)
           && pos1 < localVertexStars->size(v1)
           && pos2 < localVertexStars->size(v2)) {

          if((localVertexStars->get(v0, pos0)
              == localVertexStars->get(v1, pos1))
             && (localVertexStars->get(v0, pos0)
                 == localVertexStars->get(v2, pos2))) {

            if(localVertexStars->get(v0, pos0) != cid) {
              intersection = localVertexStars->get(v0, pos0);
              break;
            }

            pos0++;
            pos1++;
            pos2++;
          }
        }
      }

      if(intersection != -1) {
        neighbors[cid].emplace_back(intersection);
      }
    }
  }

  // convert to a FlatJaggedArray
  cellNeighbors.fillFrom(neighbors);

  std::cout << "Built " << cellNumber << " cell neighbors" << std::endl;

  // ethaneDiol.vtu, 8.7Mtets, richard (4coresHT)
  // 1 thread: 9.39488 s
  // 2 threads: 5.93132 s [faster than any other implementation]
  // 4 threads: 3.3445 s [~x3, not too bad]

  // ethaneDiol.vtu, 8.7Mtets, hal9000 (12coresHT)
  // 1 thread: 4.4475 s [speedup on VTK ~2.5]
  // 24 threads: 1.4488 s

  // ethaneDiolMedium.vtu, 70Mtets, hal9000 (12coresHT)
  // 1 thread: 29.4951 s [speedup on VTK ~3.3]
  // 24 threads: 12.7058 s (parallel speed up ~3, speed up on VTK: 2)

  // NOTE:
  // when dealing with simple types such as int, it's better to have heap
  // variable (instead of a vector<int> indexed by threadId).
  // it is MUCH MUCH MUCH better
  // especially with write access (cache misses)

  return 0;
}


int Skeleton::buildCellNeighborsFromEdgesTwoSkeleton(
  const CellArray &cellArray,
  FlatJaggedArray &cellNeighbors,
  const FlatJaggedArray &edgeStars) const {

  Timer t;

  std::cout << "Building cell neighbors..." << std::endl;

  const SimplexId cellNumber = cellArray.getNbCells();
  const SimplexId edgeNumber = edgeStars.subvectorsNumber();
  std::vector<SimplexId> offsets(cellNumber + 1);
  // number of neigbhors processed per cell
  std::vector<SimplexId> neighborsId(cellNumber);

  for(SimplexId i = 0; i < edgeNumber; i++) {
    if(edgeStars.size(i) == 2) {
      // tetra cells in edge i's star
      const auto cs0 = edgeStars.get(i, 0);
      const auto cs1 = edgeStars.get(i, 1);
      offsets[cs0 + 1]++;
      offsets[cs1 + 1]++;
    }
  }

  // compute partial sum of number of neighbors per vertex
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat neighbors vector
  std::vector<SimplexId> neighbors(offsets.back());

  // fill flat neighbors vector using offsets and neighbors count vectors
  for(SimplexId i = 0; i < edgeNumber; i++) {
    if(edgeStars.size(i) == 2) {
      // tetra cells in edge i's star
      const auto cs0 = edgeStars.get(i, 0);
      const auto cs1 = edgeStars.get(i, 1);
      neighbors[offsets[cs0] + neighborsId[cs0]] = cs1;
      neighborsId[cs0]++;
      neighbors[offsets[cs1] + neighborsId[cs1]] = cs0;
      neighborsId[cs1]++;
    }
  }

  // fill FlatJaggedArray struct
  cellNeighbors.setData(std::move(neighbors), std::move(offsets));

  std::cout << "Built " << cellNumber << " cell neighbors " << t.getElapsedTime() << " s" << std::endl;

  return 0;
}

int Skeleton::buildCellNeighborsFromVerticesTwoSkeleton(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  FlatJaggedArray &cellNeighbors,
  FlatJaggedArray *vertexStars) const {

  auto localVertexStars = vertexStars;
  FlatJaggedArray defaultVertexStars{};

  if(!localVertexStars) {
    localVertexStars = &defaultVertexStars;
  }

  if(localVertexStars->empty()) {
    this->buildVertexStarsZeroSkeleton(vertexNumber, cellArray, *localVertexStars);
  }

  Timer t;

  std::cout << "Building cell neighbors..." << std::endl;

  const SimplexId cellNumber = cellArray.getNbCells();
  using boost::container::small_vector;
  // for each cell/triangle, a vector of neighbors
  std::vector<small_vector<SimplexId, 3>> neighbors(cellNumber);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(SimplexId cid = 0; cid < cellNumber; cid++) {
    const SimplexId nbVertCell = cellArray.getCellVertexNumber(cid);

    for(SimplexId j = 0; j < nbVertCell; j++) {

      SimplexId v0 = cellArray.getCellVertex(cid, j);
      SimplexId v1 = cellArray.getCellVertex(cid, (j + 1) % nbVertCell);

      // perform an intersection of the 2 sorted star lists
      SimplexId pos0 = 0, pos1 = 0;
      SimplexId intersection = -1;

      while(pos0 < localVertexStars->size(v0)
            && pos1 < localVertexStars->size(v1)) {

        SimplexId biggest = localVertexStars->get(v0, pos0);
        if(localVertexStars->get(v1, pos1) > biggest) {
          biggest = localVertexStars->get(v1, pos1);
        }

        for(SimplexId l = pos0; l < localVertexStars->size(v0); l++) {
          if(localVertexStars->get(v0, l) < biggest) {
            pos0++;
          } else {
            break;
          }
        }
        for(SimplexId l = pos1; l < localVertexStars->size(v1); l++) {
          if(localVertexStars->get(v1, l) < biggest) {
            pos1++;
          } else {
            break;
          }
        }

        if(localVertexStars->get(v0, pos0) == localVertexStars->get(v1, pos1)) {
          if(localVertexStars->get(v0, pos0) != cid) {
            intersection = localVertexStars->get(v0, pos0);
            break;
          }

          pos0++;
          pos1++;
        }
      }

      if(intersection != -1) {
        neighbors[cid].emplace_back(intersection);
      }
    }
  }

  // convert to a FlatJaggedArray
  cellNeighbors.fillFrom(neighbors);

  std::cout << "Built " << cellNumber << " cell neighbors" << t.getElapsedTime() << " s" << std::endl;

  return 0;
}

int Skeleton::buildEdgeTrianglesTwoSkeleton(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  FlatJaggedArray &edgeTriangleList,
  const std::vector<std::array<SimplexId, 2>> &edgeList,
  std::vector<std::array<SimplexId, 3>> *triangleEdgeList) const {

  // check the consistency of the variables -- to adapt
#ifndef TTK_ENABLE_KAMIKAZE
  if(vertexNumber <= 0)
    return -1;
#endif

  auto localTriangleEdgeList = triangleEdgeList;
  std::vector<std::array<SimplexId, 3>> defaultTriangleEdgeList{};
  if(!localTriangleEdgeList) {
    localTriangleEdgeList = &defaultTriangleEdgeList;
  }

  if(localTriangleEdgeList->empty()) {
    buildTriangleEdgeListTwoSkeleton(
      vertexNumber, cellArray, *localTriangleEdgeList, edgeList);
  }

  const auto edgeNumber{edgeList.size()};

  std::vector<SimplexId> offsets(edgeNumber + 1);
  // number of neighbors processed per vertex
  std::vector<SimplexId> trianglesId(edgeNumber);

  Timer t;

  std::cout << "Building edge triangles..." << std::endl;

  // store number of triangles per edge
  for(const auto &te : *localTriangleEdgeList) {
    offsets[te[0] + 1]++;
    offsets[te[1] + 1]++;
    offsets[te[2] + 1]++;
  }

  // compute partial sum of number of triangles per edge
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat edge triangle vector
  std::vector<SimplexId> edgeTriangles(offsets.back());

  // fill flat neighbors vector using offsets and neighbors count vectors
  for(size_t i = 0; i < localTriangleEdgeList->size(); ++i) {
    const auto &te{(*localTriangleEdgeList)[i]};
    edgeTriangles[offsets[te[0]] + trianglesId[te[0]]] = i;
    trianglesId[te[0]]++;
    edgeTriangles[offsets[te[1]] + trianglesId[te[1]]] = i;
    trianglesId[te[1]]++;
    edgeTriangles[offsets[te[2]] + trianglesId[te[2]]] = i;
    trianglesId[te[2]]++;
  }

  // fill FlatJaggedArray struct
  edgeTriangleList.setData(std::move(edgeTriangles), std::move(offsets));

  std::cout << "Built " << edgeNumber << " edge triangles " << t.getElapsedTime() << " s" << std::endl;

  return 0;
}

int Skeleton::buildTriangleListTwoSkeleton(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  std::vector<std::array<SimplexId, 3>> *triangleList,
  FlatJaggedArray *triangleStars,
  std::vector<std::array<SimplexId, 4>> *cellTriangleList) const {

  Timer t;

  // check the consistency of the variables -- to adapt
#ifndef TTK_ENABLE_KAMIKAZE
  if(vertexNumber <= 0)
    return -1;
  if((!triangleList) && (!triangleStars) && (!cellTriangleList)) {
    // we've got nothing to do here.
    return -2;
  }
#endif

  // check parameters consistency (this method is useless in 2D)
  const auto dim = cellArray.getCellVertexNumber(0) - 1;
  if(dim == 2) {
    std::cerr << "Calling buildTriangleList is useless in 2D, skipping..." << std::endl;
    return -1;
  }

  std::cout << "Building triangles..." << std::endl;

  const SimplexId cellNumber = cellArray.getNbCells();

  // we need cellTriangleList to compute triangleStars
  std::vector<std::array<SimplexId, 4>> defaultCellTriangleList{};
  if(triangleStars != nullptr && cellTriangleList == nullptr) {
    cellTriangleList = &defaultCellTriangleList;
  }

  if(cellTriangleList) {
    cellTriangleList->resize(cellNumber, {-1, -1, -1, -1});
  }

  struct TriangleData {
    // the two higher vertices id of the triangle
    std::array<SimplexId, 2> highVerts{};
    // the triangle id
    SimplexId id{-1};
    TriangleData(std::array<SimplexId, 2> hVerts, SimplexId i)
      : highVerts{hVerts}, id{i} {
    }
  };

  using boost::container::small_vector;
  // for each vertex, a vector of TriangleData
  std::vector<small_vector<TriangleData, 8>> triangleTable(vertexNumber);

  SimplexId nTriangles{};

  for(SimplexId cid = 0; cid < cellNumber; cid++) {
    // a tetra cell has 4 faces
    for(size_t j = 0; j < 4; j++) {
      std::array<SimplexId, 3> triangle{};
      for(size_t k = 0; k < 3; k++) {
        // TODO: ASSUME Regular Mesh Here!
        triangle[k] = cellArray.getCellVertex(cid, (j + k) % 4);
      }
      std::sort(triangle.begin(), triangle.end());
      auto &ttable = triangleTable[triangle[0]];

      // check if current triangle already registered in triangleTable
      // via another tetra in its star
      bool found{false};
      for(auto &d : ttable) {
        if(d.highVerts[0] == triangle[1] && d.highVerts[1] == triangle[2]) {
          found = true;
          if(cellTriangleList != nullptr) {
            (*cellTriangleList)[cid][j] = d.id;
          }
          break;
        }
      }
      if(!found) {
        // new triangle added
        ttable.emplace_back(
          TriangleData{{triangle[1], triangle[2]}, nTriangles});
        if(cellTriangleList != nullptr) {
          (*cellTriangleList)[cid][j] = nTriangles;
        }
        nTriangles++;
      }
    }
  }

  // resize vectors to the correct size
  if(triangleList) {
    triangleList->resize(nTriangles);
  }

  // fill data buffers in parallel

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(this->threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId i = 0; i < vertexNumber; ++i) {
    const auto &ttable = triangleTable[i];
    for(const auto &data : ttable) {
      if(triangleList != nullptr) {
        (*triangleList)[data.id] = {i, data.highVerts[0], data.highVerts[1]};
      }
    }
  }


  if(cellTriangleList != nullptr && triangleStars != nullptr) {
    std::vector<SimplexId> offsets(nTriangles + 1);
    // number of cells processed per vertex
    std::vector<SimplexId> starIds(nTriangles);

    // store number of cells per triangle
    for(const auto &c : *cellTriangleList) {
      offsets[c[0] + 1]++;
      offsets[c[1] + 1]++;
      offsets[c[2] + 1]++;
      offsets[c[3] + 1]++;
    }

    // compute partial sum of number of cells per triangle
    for(size_t i = 1; i < offsets.size(); ++i) {
      offsets[i] += offsets[i - 1];
    }

    // allocate flat triangle stars vector
    std::vector<SimplexId> triangleSt(offsets.back());

    // fill flat neighbors vector using offsets and neighbors count vectors
    for(size_t i = 0; i < cellTriangleList->size(); ++i) {
      const auto &ct{(*cellTriangleList)[i]};
      triangleSt[offsets[ct[0]] + starIds[ct[0]]] = i;
      starIds[ct[0]]++;
      triangleSt[offsets[ct[1]] + starIds[ct[1]]] = i;
      starIds[ct[1]]++;
      triangleSt[offsets[ct[2]] + starIds[ct[2]]] = i;
      starIds[ct[2]]++;
      triangleSt[offsets[ct[3]] + starIds[ct[3]]] = i;
      starIds[ct[3]]++;
    }

    // fill FlatJaggedArray struct
    triangleStars->setData(std::move(triangleSt), std::move(offsets));
  }

  std::cout << "Built " << nTriangles << " triangles " << t.getElapsedTime() << " s" << std::endl;
  // ethaneDiolMedium.vtu, 70Mtets, hal9000 (12coresHT)// 1 thread: 58.5631 s//
  // 24 threads: 87.5816 s (~) ethaneDiol.vtu, 8.7Mtets, vger (2coresHT) - no
  // tet adj// 1 thread: 5.7427 s// 4 threads: 9.14764 s (~)
  // ethaneDiol.vtu, 8.7Mtets, vger (2coresHT) - tet adjacency// 1
  // thread: 8.59854 s// 4 threads: 15.837 s
  // (~) ethaneDiol.vtu, 8.7Mtets, vger (2coresHT) - tet adjacency only// 1
  // thread: 6.85897 s// 4 threads: 14.78 s (~)
  return 0;
}

int Skeleton::buildTriangleEdgeListTwoSkeleton(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  std::vector<std::array<SimplexId, 3>> &triangleEdgeList,
  const std::vector<std::array<SimplexId, 2>> &edgeList,
  FlatJaggedArray *vertexEdgeList,
  std::vector<std::array<SimplexId, 3>> *triangleList,
  FlatJaggedArray *triangleStarList,
  std::vector<std::array<SimplexId, 4>> *cellTriangleList) const {

  auto localVertexEdgeList = vertexEdgeList;
  FlatJaggedArray defaultVertexEdgeList{};
  if(!localVertexEdgeList) {
    localVertexEdgeList = &defaultVertexEdgeList;
  }

  if(localVertexEdgeList->empty()) {

    this->buildVertexEdgesZeroSkeleton(
      vertexNumber, edgeList, (*localVertexEdgeList));
  }

  // NOTE:
  // triangleStarList and cellTriangleList: we do not need these guys but we
  // can compute them for free optionally.

  auto localTriangleList = triangleList;
  std::vector<std::array<SimplexId, 3>> defaultTriangleList{};
  if(!localTriangleList) {
    localTriangleList = &defaultTriangleList;
  }
  if(!localTriangleList->size()) {

    buildTriangleListTwoSkeleton(vertexNumber, cellArray, localTriangleList,
                      triangleStarList, cellTriangleList);
  }

  triangleEdgeList.resize(localTriangleList->size());

  Timer tm;

  std::cout << "Building triangle edges..." << std::endl;

  // now for each triangle, grab its vertices, add the edges in the triangle
  // with no duplicate
  // let's do the real stuff
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < localTriangleList->size(); i++) {
    const auto &t = (*localTriangleList)[i];
    const auto beg0 = localVertexEdgeList->get_ptr(t[0], 0);
    const auto end0 = beg0 + localVertexEdgeList->size(t[0]);
    const auto beg1 = localVertexEdgeList->get_ptr(t[1], 0);
    const auto end1 = beg1 + localVertexEdgeList->size(t[1]);
    const auto beg2 = localVertexEdgeList->get_ptr(t[2], 0);
    const auto end2 = beg2 + localVertexEdgeList->size(t[2]);
    std::set_intersection(beg0, end0, beg1, end1, &triangleEdgeList[i][0]);
    std::set_intersection(beg0, end0, beg2, end2, &triangleEdgeList[i][1]);
    std::set_intersection(beg1, end1, beg2, end2, &triangleEdgeList[i][2]);
    std::sort(triangleEdgeList[i].begin(), triangleEdgeList[i].end());
  }

  SimplexId triangleNumber = localTriangleList->size();

  std::cout << "Built " << triangleNumber << " triangle edges " << tm.getElapsedTime() << std::endl;

  return 0;
}

int Skeleton::buildTriangleLinksTwoSkeleton(
  const std::vector<std::array<SimplexId, 3>> &triangleList,
  const FlatJaggedArray &triangleStars,
  const CellArray &cellArray,
  FlatJaggedArray &triangleLinks) const {

#ifndef TTK_ENABLE_KAMIKAZE
  if(triangleList.empty())
    return -1;
  if((triangleStars.empty())
     || (triangleStars.subvectorsNumber() != triangleList.size()))
    return -2;
#endif

  Timer tm;

  // check parameters consistency (this method is useless in 2D)
  const auto dim = cellArray.getCellVertexNumber(0) - 1;
  if(dim == 2) {
    std::cerr << "Calling buildTriangleLinks is useless in 2D, skipping..." << std::endl;
    return -1;
  }

  const SimplexId triangleNumber = triangleList.size();
  std::vector<SimplexId> offsets(triangleNumber + 1);
  // one vertex per star
  std::vector<SimplexId> links(triangleStars.dataSize());

  std::cout << "Building triangle links..." << std::endl;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId i = 0; i < triangleNumber; i++) {

    // copy the triangleStars offsets array
    offsets[i] = triangleStars.offset(i);

    const auto &t = triangleList[i];
    for(SimplexId j = 0; j < triangleStars.size(i); j++) {
      // for each tetra in triangle i's star, get the opposite vertex
      for(size_t k = 0; k < 4; k++) {
        const auto v = cellArray.getCellVertex(triangleStars.get(i, j), k);
        if(v != t[0] && v != t[1] && v != t[2]) {
          links[offsets[i] + j] = v;
          break;
        }
      }
    }
  }

  // don't forget the last offset
  offsets[triangleNumber] = triangleStars.offset(triangleNumber);

  triangleLinks.setData(std::move(links), std::move(offsets));

  std::cout << "Built " << triangleNumber << " triangle links " << tm.getElapsedTime() << std::endl;

  return 0;
}

int Skeleton::buildVertexTrianglesTwoSkeleton(
  const SimplexId &vertexNumber,
  const std::vector<std::array<SimplexId, 3>> &triangleList,
  FlatJaggedArray &vertexTriangles) const {

  Timer tm;

  std::cout << "Building vertex triangles..." << std::endl;

  std::vector<SimplexId> offsets(vertexNumber + 1);
  // number of triangles processed per vertex
  std::vector<SimplexId> trianglesId(vertexNumber);

  // store number of triangles per vertex
  for(const auto &t : triangleList) {
    offsets[t[0] + 1]++;
    offsets[t[1] + 1]++;
    offsets[t[2] + 1]++;
  }

  // compute partial sum of number of neighbors per vertex
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat neighbors vector
  std::vector<SimplexId> data(offsets.back());

  // fill flat data vector using offsets and triangles count vectors
  for(size_t i = 0; i < triangleList.size(); ++i) {
    const auto &t{triangleList[i]};
    data[offsets[t[0]] + trianglesId[t[0]]] = i;
    trianglesId[t[0]]++;
    data[offsets[t[1]] + trianglesId[t[1]]] = i;
    trianglesId[t[1]]++;
    data[offsets[t[2]] + trianglesId[t[2]]] = i;
    trianglesId[t[2]]++;
  }

  // fill FlatJaggedArray struct
  vertexTriangles.setData(std::move(data), std::move(offsets));

  std::cout << "Built " << vertexNumber << " vertex triangles " << tm.getElapsedTime() << std::endl;

  return 0;
}



// 2D cells (triangles)
int Skeleton::buildEdgeLinksOneSkeleton(
  const std::vector<std::array<SimplexId, 2>> &edgeList,
  const FlatJaggedArray &edgeStars,
  const CellArray &cellArray,
  FlatJaggedArray &edgeLinks) const {

#ifndef TTK_ENABLE_KAMIKAZE
  if(edgeList.empty())
    return -1;
  if(edgeStars.subvectorsNumber() != edgeList.size())
    return -2;
#endif

  Timer t;

  const SimplexId edgeNumber = edgeStars.subvectorsNumber();
  std::vector<SimplexId> offsets(edgeNumber + 1);
  // one vertex per star
  std::vector<SimplexId> links(edgeStars.dataSize());

  std::cout << "Building edge links..." << std::endl;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId i = 0; i < edgeNumber; i++) {
    // copy the edgeStars offsets array
    offsets[i] = edgeStars.offset(i);

    for(SimplexId j = 0; j < edgeStars.size(i); j++) {
      // for each cell/triangle in edge i's star, get the opposite vertex
      for(int k = 0; k < 3; k++) {
        const auto v = cellArray.getCellVertex(edgeStars.get(i, j), k);
        if(v != edgeList[i][0] && v != edgeList[i][1]) {
          // edge i does not contain vertex i
          links[offsets[i] + j] = v;
          break;
        }
      }
    }
  }

  // don't forget the last offset
  offsets[edgeNumber] = edgeStars.offset(edgeNumber);

  edgeLinks.setData(std::move(links), std::move(offsets));

  std::cout << "Built " << edgeNumber << " edge links " << t.getElapsedTime() << " s" << std::endl;

  return 0;
}

// 3D cells (tetrahedron)
int Skeleton::buildEdgeLinksOneSkeleton(
  const std::vector<std::array<SimplexId, 2>> &edgeList,
  const FlatJaggedArray &edgeStars,
  const std::vector<std::array<SimplexId, 6>> &cellEdges,
  FlatJaggedArray &edgeLinks) const {

#ifndef TTK_ENABLE_KAMIKAZE
  if(edgeList.empty())
    return -1;
  if((edgeStars.empty()) || (edgeStars.subvectorsNumber() != edgeList.size()))
    return -2;
  if(cellEdges.empty())
    return -3;
#endif

  Timer t;

  const SimplexId edgeNumber = edgeStars.subvectorsNumber();
  std::vector<SimplexId> offsets(edgeNumber + 1);
  // one edge per star
  std::vector<SimplexId> links(edgeStars.dataSize());

  std::cout << "Building edge links..." << std::endl;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId i = 0; i < edgeNumber; i++) {
    // copy the edgeStars offsets array
    offsets[i] = edgeStars.offset(i);

    // current edge vertices
    const auto &e = edgeList[i];
    for(SimplexId j = 0; j < edgeStars.size(i); j++) {
      const auto c = edgeStars.get(i, j);
      for(size_t k = 0; k < cellEdges[c].size(); k++) {
        // cell edge id
        const auto ceid = cellEdges[c][k];
        // cell edge vertices
        const auto &ce = edgeList[ceid];

        if(ce[0] != e[0] && ce[0] != e[1] && ce[1] != e[0] && ce[1] != e[1]) {
          // ce and e have no vertex in common
          links[offsets[i] + j] = ceid;
          break;
        }
      }
    }
  }

  // don't forget the last offset
  offsets[edgeNumber] = edgeStars.offset(edgeNumber);

  edgeLinks.setData(std::move(links), std::move(offsets));

  std::cout << "Built " << edgeNumber << " edge links " << t.getElapsedTime() << " s" << std::endl;

  return 0;
}

template <std::size_t n>
int Skeleton::buildEdgeListOneSkeleton(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  std::vector<std::array<SimplexId, 2>> &edgeList,
  FlatJaggedArray &edgeStars,
  std::vector<std::array<SimplexId, n>> &cellEdgeList) const {

  Timer t;

  // check parameters consistency (we need n to be consistent with the
  // dimensionality of the mesh)
  const size_t dim = cellArray.getCellVertexNumber(0) - 1;
  if(n != dim * (dim + 1) / 2) {
    std::cerr << "Wrong template parameter (" << n << " edges per " << dim << "D cell)" << std::endl;
    std::cerr << "Cannot build edge list" << std::endl;
    return -1;
  }

  std::cout << "Building edges..." << std::endl;

  const SimplexId cellNumber = cellArray.getNbCells();
  cellEdgeList.resize(cellNumber);

  struct EdgeData {
    // the id of the edge higher vertex
    SimplexId highVert{};
    // the edge id
    SimplexId id{};
    EdgeData(SimplexId hv, SimplexId i) : highVert{hv}, id{i} {
    }
  };

  using boost::container::small_vector;
  // for each vertex, a vector of EdgeData
  std::vector<small_vector<EdgeData, 8>> edgeTable(vertexNumber);

  // const int timeBuckets = std::min<ttk::SimplexId>(10, cellNumber);
  SimplexId edgeCount{};

  for(SimplexId cid = 0; cid < cellNumber; cid++) {

    const SimplexId nbVertsInCell = cellArray.getCellVertexNumber(cid);
    // id of edge in cell
    SimplexId ecid{};

    // tet case: {0-1}, {0-2}, {0-3}, {1-2}, {1-3}, {2-3}
    for(SimplexId j = 0; j <= nbVertsInCell - 2; j++) {
      for(SimplexId k = j + 1; k <= nbVertsInCell - 1; k++) {
        // edge processing
        SimplexId v0 = cellArray.getCellVertex(cid, j);
        SimplexId v1 = cellArray.getCellVertex(cid, k);
        if(v0 > v1) {
          std::swap(v0, v1);
        }
        auto &vec = edgeTable[v0];
        const auto pos
          = std::find_if(vec.begin(), vec.end(),
                         [&](const EdgeData &a) { return a.highVert == v1; });
        if(pos == vec.end()) {
          // not found in edgeTable: new edge
          vec.emplace_back(EdgeData{v1, edgeCount});
          cellEdgeList[cid][ecid] = edgeCount;
          edgeCount++;
        } else {
          // found an existing edge
          cellEdgeList[cid][ecid] = pos->id;
        }
        ecid++;
      }
    }
  }

  // allocate & fill edgeList in parallel
  edgeList.resize(edgeCount);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(this->threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId i = 0; i < vertexNumber; ++i) {
    const auto &etable = edgeTable[i];
    for(const auto &data : etable) {
      edgeList[data.id] = {i, data.highVert};
    }
  }

  // return cellEdgeList to get edgeStars
  std::vector<SimplexId> offsets(edgeCount + 1);
  // number of cells processed per edge
  std::vector<SimplexId> starIds(edgeCount);

  // store number of cells per edge
  for(const auto &ce : cellEdgeList) {
    for(const auto eid : ce) {
      offsets[eid + 1]++;
    }
  }

  // compute partial sum of number of cells per edge
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat edge stars vector
  std::vector<SimplexId> edgeSt(offsets.back());

  // fill flat neighbors vector using offsets and neighbors count vectors
  for(size_t i = 0; i < cellEdgeList.size(); ++i) {
    const auto &ce{cellEdgeList[i]};
    for(const auto eid : ce) {
      edgeSt[offsets[eid] + starIds[eid]] = i;
      starIds[eid]++;
    }
  }

  // fill FlatJaggedArray struct
  edgeStars.setData(std::move(edgeSt), std::move(offsets));

  std::cout << "Built " << edgeCount << " edges " << t.getElapsedTime() << " s" << std::endl;

  // ethaneDiolMedium.vtu, 70Mtets, hal9000 (12coresHT)
  // 1 thread: 10.4979 s
  // 24 threads: 12.3994 s [not efficient in parallel]

  return 0;
}

// explicit template instantiation for 1D cells (edges)
template int Skeleton::buildEdgeListOneSkeleton<1>(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  std::vector<std::array<SimplexId, 2>> &edgeList,
  FlatJaggedArray &edgeStars,
  std::vector<std::array<SimplexId, 1>> &cellEdgeList) const;

// explicit template instantiation for 2D cells (triangles)
template int Skeleton::buildEdgeListOneSkeleton<3>(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  std::vector<std::array<SimplexId, 2>> &edgeList,
  FlatJaggedArray &edgeStars,
  std::vector<std::array<SimplexId, 3>> &cellEdgeList) const;

// explicit template instantiation for 3D cells (tetrathedron)
template int Skeleton::buildEdgeListOneSkeleton<6>(
  const SimplexId &vertexNumber,
  const CellArray &cellArray,
  std::vector<std::array<SimplexId, 2>> &edgeList,
  FlatJaggedArray &edgeStars,
  std::vector<std::array<SimplexId, 6>> &cellEdgeList) const;


int Skeleton::buildVertexEdgesZeroSkeleton(
  const SimplexId &vertexNumber,
  const std::vector<std::array<SimplexId, 2>> &edgeList,
  FlatJaggedArray &vertexEdges) const {

  std::vector<SimplexId> offsets(vertexNumber + 1);
  // number of edges processed per vertex
  std::vector<SimplexId> edgesId(vertexNumber);

  Timer t;

  std::cout << "Building vertex edges..." << std::endl;

  // store number of edges per vertex
  for(const auto &e : edgeList) {
    offsets[e[0] + 1]++;
    offsets[e[1] + 1]++;
  }

  // compute partial sum of number of neighbors per vertex
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat neighbors vector
  std::vector<SimplexId> data(offsets.back());

  // fill flat data vector using offsets and edges count vectors
  for(size_t i = 0; i < edgeList.size(); ++i) {
    const auto &e{edgeList[i]};
    data[offsets[e[0]] + edgesId[e[0]]] = i;
    edgesId[e[0]]++;
    data[offsets[e[1]] + edgesId[e[1]]] = i;
    edgesId[e[1]]++;
  }

  // fill FlatJaggedArray struct
  vertexEdges.setData(std::move(data), std::move(offsets));

  std::cout << "Built " << vertexNumber << " vertex edges" << std::endl;

  // ethaneDiolMedium.vtu, 70Mtets, hal9000 (12coresHT)
  // 1 thread: 11.85 s
  // 24 threads: 20.93 s [not efficient]

  return 0;
}

// 2D cells (triangles)
int Skeleton::buildVertexLinksZeroSkeleton(
  const FlatJaggedArray &vertexStars,
  const std::vector<std::array<SimplexId, 3>> &cellEdges,
  const std::vector<std::array<SimplexId, 2>> &edgeList,
  FlatJaggedArray &vertexLinks) const {

#ifndef TTK_ENABLE_KAMIKAZE
  if(vertexStars.empty())
    return -1;
  if(cellEdges.empty())
    return -2;
  if(edgeList.empty())
    return -3;
#endif

  Timer t;

  const SimplexId vertexNumber = vertexStars.subvectorsNumber();
  std::vector<SimplexId> offsets(vertexNumber + 1);
  // one edge per star
  std::vector<SimplexId> links(vertexStars.dataSize());

  std::cout << "Building vertex links..." << std::endl;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId i = 0; i < vertexNumber; i++) {

    // copy the vertexStars offsets array
    offsets[i] = vertexStars.offset(i);

    for(SimplexId j = 0; j < vertexStars.size(i); j++) {
      // for each cell/triangle in vertex i's star, get the opposite edge
      for(size_t k = 0; k < cellEdges[vertexStars.get(i, j)].size(); k++) {
        const auto e = cellEdges[vertexStars.get(i, j)][k];
        if(i != edgeList[e][0] && i != edgeList[e][1]) {
          links[offsets[i] + j] = e;
          break;
        }
      }
    }
  }

  // don't forget the last offset
  offsets[vertexNumber] = vertexStars.offset(vertexNumber);

  vertexLinks.setData(std::move(links), std::move(offsets));

  std::cout << "Built " << vertexNumber << " vertex links " << t.getElapsedTime() << " s" << std::endl;

  return 0;
}

// 3D cells (tetrahedron)
int Skeleton::buildVertexLinksZeroSkeleton(
  const FlatJaggedArray &vertexStars,
  const std::vector<std::array<SimplexId, 4>> &cellTriangles,
  const std::vector<std::array<SimplexId, 3>> &triangleList,
  FlatJaggedArray &vertexLinks) const {

#ifndef TTK_ENABLE_KAMIKAZE
  if(vertexStars.empty())
    return -1;
  if(cellTriangles.empty())
    return -2;
  if(triangleList.empty())
    return -3;
#endif

  Timer tm;

  const SimplexId vertexNumber = vertexStars.subvectorsNumber();
  std::vector<SimplexId> offsets(vertexNumber + 1);
  // one triangle per star
  std::vector<SimplexId> links(vertexStars.dataSize());

  std::cout << "Building vertex links..." << std::endl;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(SimplexId i = 0; i < vertexNumber; i++) {

    // copy the vertexStars offsets array
    offsets[i] = vertexStars.offset(i);

    for(SimplexId j = 0; j < vertexStars.size(i); j++) {
      // for each cell/tetra in vertex i's star, get the opposite triangle
      for(size_t k = 0; k < cellTriangles[vertexStars.get(i, j)].size(); k++) {
        const auto t = cellTriangles[vertexStars.get(i, j)][k];
        if(i != triangleList[t][0] && i != triangleList[t][1]
           && i != triangleList[t][2]) {
          links[offsets[i] + j] = t;
          break;
        }
      }
    }
  }

  // don't forget the last offset
  offsets[vertexNumber] = vertexStars.offset(vertexNumber);

  vertexLinks.setData(std::move(links), std::move(offsets));

  std::cout << "Built " << vertexNumber << " vertex links " << tm.getElapsedTime() << std::endl;

  return 0;
}

int Skeleton::buildVertexNeighborsZeroSkeleton(
  const SimplexId &vertexNumber,
  FlatJaggedArray &vertexNeighbors,
  const std::vector<std::array<SimplexId, 2>> &edgeList) const {

  std::vector<SimplexId> offsets(vertexNumber + 1);
  // number of neighbors processed per vertex
  std::vector<SimplexId> neighborsId(vertexNumber);

  Timer t;

  std::cout << "Building vertex neighbors..." << std::endl;

  // store number of neighbors per vertex
  for(const auto &e : edgeList) {
    offsets[e[0] + 1]++;
    offsets[e[1] + 1]++;
  }

  // compute partial sum of number of neighbors per vertex
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat neighbors vector
  std::vector<SimplexId> neighbors(offsets.back());

  // fill flat neighbors vector using offsets and neighbors count vectors
  for(const auto &e : edgeList) {
    neighbors[offsets[e[0]] + neighborsId[e[0]]] = e[1];
    neighborsId[e[0]]++;
    neighbors[offsets[e[1]] + neighborsId[e[1]]] = e[0];
    neighborsId[e[1]]++;
  }

  // fill FlatJaggedArray struct
  vertexNeighbors.setData(std::move(neighbors), std::move(offsets));

  std::cout << "Built " << vertexNumber << " vertex neighbors " << t.getElapsedTime() << " s" << std::endl;

  // ethaneDiolMedium.vtu, 70Mtets, hal9000 (12coresHT)
  // (only merging step, after edge list creation)
  // 1 thread: 9.16 s
  // 24 threads: 13.21 s [not efficient in parallel]

  return 0;
}

int Skeleton::buildVertexStarsZeroSkeleton(const SimplexId &vertexNumber,
                                   const CellArray &cellArray,
                                   FlatJaggedArray &vertexStars) const {

  Timer t;

  std::cout << "Building vertex stars..." << std::endl;

  std::vector<SimplexId> offsets(vertexNumber + 1);
  // number of cells processed per vertex
  std::vector<SimplexId> cellIds(vertexNumber);

  const auto cellNumber = cellArray.getNbCells();

  // store number of stars per vertex
  for(SimplexId i = 0; i < cellNumber; ++i) {
    const auto nbVertCell = cellArray.getCellVertexNumber(i);
    for(SimplexId j = 0; j < nbVertCell; ++j) {
      offsets[cellArray.getCellVertex(i, j) + 1]++;
    }
  }

  // compute partial sum of number of stars per vertex
  for(size_t i = 1; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1];
  }

  // allocate flat data vector
  std::vector<SimplexId> data(offsets.back());

  // fill flat data vector using offsets and edges count vectors
  for(SimplexId i = 0; i < cellNumber; ++i) {
    const auto nbVertCell = cellArray.getCellVertexNumber(i);
    for(SimplexId j = 0; j < nbVertCell; ++j) {
      const auto v = cellArray.getCellVertex(i, j);
      data[offsets[v] + cellIds[v]] = i;
      cellIds[v]++;
    }
  }

  // fill FlatJaggedArray struct
  vertexStars.setData(std::move(data), std::move(offsets));

  std::cout << "Built " << vertexNumber << " vertex stars " << t.getElapsedTime() << " s" << std::endl;

  // ethaneDiol.vtu, 8.7Mtets, hal9000 (12coresHT)
  // 1 thread: 0.53 s
  // 24 threads: 7.99 s

  return 0;
}
