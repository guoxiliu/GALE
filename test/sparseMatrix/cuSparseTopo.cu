/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include "common.h"
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <string.h>           // memset


/**
 * Function to generate a random sparse matrix in CSR format
 */ 
int generateTopoMatrix(int rows, int cols, int factor, 
                        float** csrVal, int** csrOffsets, int** csrColumns) {
    int numNonZeros = factor * rows;
    // the value in the sparse matrix will only be 1
    float *csrValA = (float *) malloc(sizeof(float) * numNonZeros);
    if (csrValA != NULL) {
        for (int i = 0; i < numNonZeros; ++i) csrValA[i] = 1.0f;
    }
    else {
        fprintf(stderr, "Cannot allocated memory for value array. Aborting...\n");
        exit(EXIT_FAILURE);
    }
    // memset(csrValA, 1.0f, numNonZeros * sizeof(float));  // cannot use memset() for float number

    // each row is expected to have `factor` values 
    int *csrOffsetsA = (int *) malloc(sizeof(int) * (rows + 1));
    if (csrOffsetsA != NULL) {
        csrOffsetsA[0] = 0;
        for (int i = 1; i <= rows; ++i) {
            csrOffsetsA[i] = csrOffsetsA[i-1] + factor;
        }
    }
    else {
        fprintf(stderr, "Cannot allocated memory for offset array. Aborting...\n");
        exit(EXIT_FAILURE);
    }

    // column index should be the size of non-zero elements
    int *csrColumnsA = (int *) malloc(sizeof(int) * numNonZeros);
    if (csrColumnsA != NULL) {
        bool generated[cols] = { false };
        for (int i = 0; i < rows; ++i) {
            int candidates[factor] = { 0 };
            for (int j = 0; j < factor; ++j) {
                do {
                    candidates[j] = rand() % cols;
                } while (generated[candidates[j]]);
                generated[candidates[j]] = true;
            }
            for (int j = 0; j < factor; ++j) {
                csrColumnsA[i * factor + j] = candidates[j];
                generated[candidates[j]] = false;
            }
        }
    }
    else {
        fprintf(stderr, "Cannot allocated memory for offset array. Aborting...\n");
        exit(EXIT_FAILURE);
    }

    // // The following function cannot generate unique indices 
    // for (int i = 0; i < numNonZeros; ++i) {
    //     int colIndex = rand() % cols;
    //     csrColumnsA[i] = colIndex;
    // }

    *csrVal = csrValA;
    *csrOffsets = csrOffsetsA;
    *csrColumns = csrColumnsA;

    return numNonZeros;
}

/**
 * Print the partial array
*/

void printPartialArray(void *M, int type, int nrows, int maxRow = -1) {
    // integer array 
    if (type == 1) {
        int *arr = (int *) M;
        printf("[");
        if (maxRow == -1 || maxRow > nrows) {
            maxRow = nrows;
        }
        for (int i = 0; i < maxRow; ++i) {
            if (i == maxRow - 1) {
                printf("%d]\n", arr[i]);
            }
            else {
                printf("%d, ", arr[i]);
            }
        }
    }
    else if (type == 2) {
        float *arr = (float *) M;
        printf("[");
        if (maxRow == -1 || maxRow > nrows) {
            maxRow = nrows;
        }
        for (int i = 0; i < maxRow; ++i) {
            if (i == maxRow - 1) {
                printf("%g]\n", arr[i]);
            }
            else {
                printf("%g, ", arr[i]);
            }
        }
    }
    else {
        fprintf(stderr, "Unsupported data type!\n");
    }
}



int main(int argc, char **argv) {

    srand(9384);

    // Host matrix definition
    const int tetraNumber = 1600000;
    const int edgeNumber = 2100000;
    const int vertexNumber = 340000;
    
    int *hACsrOffsets, *hACsrColumns, *hBCsrOffsets, *hBCsrColumns;
    float *hAValues, *hBValues;
    
    int nnzA = generateTopoMatrix(tetraNumber, vertexNumber, 4, &hAValues, &hACsrOffsets, &hACsrColumns);
    int nnzB = generateTopoMatrix(edgeNumber, vertexNumber, 2, &hBValues, &hBCsrOffsets, &hBCsrColumns);

    // printf("---------- Matrix A ----------\n");
    // printPartialArray(hAValues, 2, nnzA);
    // printPartialArray(hACsrOffsets, 1, tetraNumber+1);
    // printPartialArray(hACsrColumns, 1, nnzA);
    // printf("---------- Matrix B ----------\n");
    // printPartialArray(hBValues, 2, nnzB);
    // printPartialArray(hBCsrOffsets, 1, edgeNumber+1);
    // printPartialArray(hBCsrColumns, 1, nnzB);

    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    
    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B
    int   *dACsrOffsets, *dAColumns, *dBCsrOffsets, *dBColumns,
          *dCCsrOffsets, *dCColumns;
    float *dAValues, *dBValues, *dCValues;
    // allocate A
    CHECK( cudaMalloc((void**) &dACsrOffsets,
                           (tetraNumber + 1) * sizeof(int)) )
    CHECK( cudaMalloc((void**) &dAColumns, nnzA * sizeof(int))   )
    CHECK( cudaMalloc((void**) &dAValues,  nnzA * sizeof(float)) )
    // allocate B
    CHECK( cudaMalloc((void**) &dBCsrOffsets,
                           (edgeNumber + 1) * sizeof(int)) )
    CHECK( cudaMalloc((void**) &dBColumns, nnzB * sizeof(int))   )
    CHECK( cudaMalloc((void**) &dBValues,  nnzB * sizeof(float)) )
    // allocate C offsets
    CHECK( cudaMalloc((void**) &dCCsrOffsets,
                           (tetraNumber + 1) * sizeof(int)) )

    // timer used to measure the execution time
    clock_t t;

    // copy A
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    t = clock();
    CHECK( cudaMemcpy(dACsrOffsets, hACsrOffsets,
                           (tetraNumber + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK( cudaMemcpy(dAColumns, hACsrColumns, nnzA * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK( cudaMemcpy(dAValues, hAValues,
                           nnzA * sizeof(float), cudaMemcpyHostToDevice) )
    // copy B
    CHECK( cudaMemcpy(dBCsrOffsets, hBCsrOffsets,
                           (edgeNumber + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK( cudaMemcpy(dBColumns, hBCsrColumns, nnzB * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK( cudaMemcpy(dBValues, hBValues,
                           nnzB * sizeof(float), cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matBTrans, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, tetraNumber, vertexNumber, nnzA,
                                      dACsrOffsets, dAColumns, dAValues,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, edgeNumber, vertexNumber, nnzB,
                                      dBCsrOffsets, dBColumns, dBValues,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, tetraNumber, edgeNumber, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    //--------------------------------------------------------------------------
    // Transpose matrix B
    cudaEventRecord(start);
    float *dBTransValues;
    int *dBTransOffsets, *dBTransRows;
    CHECK( cudaMalloc((void **) &dBTransOffsets, (vertexNumber + 1) * sizeof(int)));
    CHECK( cudaMalloc((void **) &dBTransRows, nnzB * sizeof(int)));
    CHECK( cudaMalloc((void **) &dBTransValues, nnzB * sizeof(float)));

    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( 
        cusparseCsr2cscEx2_bufferSize(handle, edgeNumber, vertexNumber, nnzB,
                                      dBValues, dBCsrOffsets, dBColumns,
                                      dBTransValues, dBTransOffsets, dBTransRows,
                                      CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
                                      CUSPARSE_CSR2CSC_ALG1, &bufferSize) )
    CHECK(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUSPARSE( cusparseCsr2cscEx2(handle, edgeNumber, vertexNumber, nnzB,
                                        dBValues, dBCsrOffsets, dBColumns,
                                        dBTransValues, dBTransOffsets, dBTransRows,
                                        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
                                        CUSPARSE_CSR2CSC_ALG1, dBuffer) )

    CHECK_CUSPARSE( cusparseCreateCsr(&matBTrans, vertexNumber, edgeNumber, nnzB,
                                      dBTransOffsets, dBTransRows, dBTransValues,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matBTrans, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matBTrans, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matBTrans, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matBTrans, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )
    // get matrix C non-zero entries nnzC
    int64_t numRowsC, numColsC, nnzC;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &numRowsC, &numColsC,
                                         &nnzC) )

    if (numRowsC != tetraNumber) {
        fprintf(stderr, "Number of rows of matrix C does not match, expected %d but got %ld\n", tetraNumber, numRowsC);
    }
    if (numColsC != edgeNumber) {
        fprintf(stderr, "Number of columns of matrix C does not match, expected %d but got %ld\n", edgeNumber, numColsC);
    }

    // allocate matrix C
    CHECK( cudaMalloc((void**) &dCColumns, nnzC * sizeof(int))   )
    CHECK( cudaMalloc((void**) &dCValues,  nnzC * sizeof(float)) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dCValues, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dCCsrOffsets, dCColumns, dCValues) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dCValues

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matBTrans, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )
    cudaEventRecord(stop);
    //--------------------------------------------------------------------------
    // device result check
    printf("Non zero elments of matrix C: %ld\n", nnzC);
    int *hC_csrOffsets_tmp = (int *) malloc(sizeof(int) * (tetraNumber + 1));
    int *hC_columns_tmp = (int *) malloc(sizeof(int) * nnzC);
    float *hC_values_tmp = (float *) malloc(sizeof(int) * nnzC);
    CHECK( cudaMemcpy(hC_csrOffsets_tmp, dCCsrOffsets,
                           (tetraNumber + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK( cudaMemcpy(hC_columns_tmp, dCColumns, nnzC * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK( cudaMemcpy(hC_values_tmp, dCValues, nnzC * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    t = clock() - t;
    float computeTime = 0;
    cudaEventElapsedTime(&computeTime, start, stop);
    printf("Time measured for computation: %.5f s\n", computeTime/1000);
    printf("Total time measured for computation: %.5f s\n", ((double) t / CLOCKS_PER_SEC));
    // printf("---------- Matrix C ----------\n");
    // printPartialArray(hC_values_tmp, 2, nnzC);
    // printPartialArray(hC_csrOffsets_tmp, 1, tetraNumber+1);
    // printPartialArray(hC_columns_tmp, 1, nnzC);

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matBTrans) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    int counter = 0, correct = 6 * tetraNumber;
    for (int i = 0; i < nnzC; ++i) {
        if (hC_values_tmp[i] == 2) counter++;
    }
    if (counter == correct)
        printf("spgemm_example test PASSED\n");
    else {
        printf("spgemm_example test FAILED: wrong result, expected %d but got %d\n", correct, counter);
        return EXIT_FAILURE;
    }

    free(hAValues);
    free(hACsrOffsets);
    free(hACsrColumns);
    free(hBValues);
    free(hBCsrOffsets);
    free(hBCsrColumns);

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK( cudaFree(dBuffer) )
    CHECK( cudaFree(dBuffer1) )
    CHECK( cudaFree(dBuffer2) )
    CHECK( cudaFree(dACsrOffsets) )
    CHECK( cudaFree(dAColumns) )
    CHECK( cudaFree(dAValues) )
    CHECK( cudaFree(dBCsrOffsets) )
    CHECK( cudaFree(dBColumns) )
    CHECK( cudaFree(dBValues) )
    CHECK( cudaFree(dBTransOffsets) )
    CHECK( cudaFree(dBTransRows) )
    CHECK( cudaFree(dBTransValues) )
    CHECK( cudaFree(dCCsrOffsets) )
    CHECK( cudaFree(dCColumns) )
    CHECK( cudaFree(dCValues) )
    return EXIT_SUCCESS;
}