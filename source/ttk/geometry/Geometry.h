// \ingroup base
/// \class ttk::Geometry
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \date February 2016.
///
/// \brief Minimalist class that handles simple geometric computations
/// (operations on std::vectors, barycentric coordinates, etc.).
///

#pragma once

#include <cmath>

namespace ttk {

    /// Compute the Euclidean distance between two points
    /// \param p0 xyz coordinates of the first input point.
    /// \param p1 xyz coordinates of the second input point.
    /// \param dimension Optional parameter that specifies the dimension of
    /// the point set (by default 3).
    template <typename T>
    T distance(const T *p0, const T *p1, const int &dimension = 3) {
        T distance = 0;

        for(int i = 0; i < dimension; i++) {
            distance += (p0[i] - p1[i]) * (p0[i] - p1[i]);
        }

        return sqrt(distance);
    }
}
