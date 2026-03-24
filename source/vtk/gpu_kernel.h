#ifndef SFCP_KERNELS
#define SFCP_KERNELS

// probably some missing #includes here
#include <cuda_runtime.h> // Device management stuff
#include <vtkType.h> // vtkIdType
#include <cstdio>
#include "class_enum.h"

#define CONSTRAIN_BLOCK 1

__global__ void critPoints(const int * __restrict__ VV, // formerly int
                           const int * __restrict__ VV_index, // formerly int
                           const int * __restrict__ VV_to_global,
                           int * __restrict__ valences,
                           const int points,
                           const int max_VV_guess, // formerly SFCP_ColumnNum
                           const double * __restrict__ scalar_values,
                           const int * __restrict__ partition, // NEW
                           const vtkIdType * __restrict__ vvi, // NEW
                           const vtkIdType * __restrict__ ivvi, // NEW
                           #if CONSTRAIN_BLOCK
                           const int force_block,
                           #endif
                           unsigned int * __restrict__ classes);

#endif
