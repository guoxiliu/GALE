#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

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

__device__ int binarySearch(const int arr[], const int length, const int target) {
    int left = 0, right = length-1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    // no exact match found, return negative
    return -(left+1);
}

__device__ void insertElement(int arr[], const int length, const int pos, const int target) {
    for (int i = length+1; i > pos; --i) {
        arr[i] = arr[i-1];
    }
    arr[pos] = target;
}
