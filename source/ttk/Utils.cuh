#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <vtkType.h>

#define MOD_VALUE 29

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        if(abort) exit(error);                                                 \
    }                                                                          \
}


#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}


__global__ void add(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void multiply(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

__global__ void test2DArray(int **arr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 10; ++i) 
        arr[tid][i] *= 2;
}

__global__ void printArray(const int *arr, int start, int end) {
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = start; i < end; ++i) {
        printf("%d: %d\n", i, arr[i]);
    }
}

// Get the hash value using the modulo operation
__device__ int getHashPosition(const int *arr, const int length, const int target) {
    int pos = (target % MOD_VALUE) % length;
    while (arr[pos] != target && arr[pos] != -1) {
        pos = (pos + 1) % length;
    }
    return pos;
}


// Binary search an integer in a 1D array [0, end)
__device__ int binarySearch1D(const int arr[], const int end, const int target) {
    int left = 0, right = end-1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    // no exact match found, return negative
    return -(left+1);
}

// Insert an interger at a specific position
__device__ void insertElement1D(int arr[], const int length, const int pos, const int target) {
    for (int i = length; i > pos; --i) {
        arr[i] = arr[i-1];
    }
    arr[pos] = target;
}

// Binary search an integer pair in a 1D array of pairs/triplets (specified by `dim`) [start, end)
template<size_t dim>
__device__ int binarySearch(int *arr, const int start, const int end, const vtkIdType target[dim]) {
    int left = start, right = end - 1;
    // printf("left: %d, right: %d, length: %d, target: (%d, %d)\n", left, right, length, target[0], target[1]);
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int loc = mid * dim;
        for (int i = 0; i < dim; ++i) {
            if (arr[loc + i] == target[i]) {
                if (i == dim - 1) return mid;
            }
            else if (arr[loc + i] < target[i]) {
                left = mid + 1;
                break;
            }
            else {
                right = mid - 1;
                break;
            }
        }
    }
    // no exact match found, return a negative value for future insertion
    return -(left+1);
}

// Insert an interger pair at a specific position
template<size_t dim>
__device__ void insertElement(int *arr, const int start, const int length, const int pos, const vtkIdType target[dim]) {
    for (int i = start + length; i > pos; --i) {
        for (int j = 0; j < dim; ++j) {
            arr[i * dim + j] = arr[(i-1) * dim + j];
        }
    }
    for (int j = 0; j < dim; ++j) {
        arr[pos * dim + j] = target[j];
    }
}
