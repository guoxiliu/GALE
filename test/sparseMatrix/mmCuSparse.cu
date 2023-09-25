#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda.h>

int rows = 1024, cols = 1024;

/**
 * Generate random dense matrix A in column-major order, ensuring some
 * elements are zero and the matrix is sparse.
*/
int generateRandomDenseMatrix(float **outA) {
    double rMax = (double) RAND_MAX;
    float *A = (float *) malloc(sizeof(float) * rows * cols);
    int totalNnz = 0;

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            int r = rand();
            float *curr = A + (j * rows + i);
            if (r % 3 > 0) {
                *curr = 0.0f;
            }
            else {
                double dr = (double) r;
                *curr = (dr / rMax) * 100.0;
            }
            if (*curr != 0.0f) totalNnz++;
        }
    }
    *outA = A;
    return totalNnz;
}

void printPartialMatrix(float *M, int nrows, int ncols, int maxRow, int maxCol) {
    int row, col;
    for (row = 0; row < maxRow; ++row) {
        for (col = 0; col < maxCol; ++col) {
            printf("%2.2f ", M[row * ncols + col]);
        }
        printf("...\n");
    }
    printf("...\n");
}

int main(int argc, char **argv) {
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseHandle_t handle = 0;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t Adescr, Bdescr, Cdescr;

    // Generate input
    srand(9384);
    int trueANnz = generateRandomDenseMatrix(&A);
    int trueBNnz = generateRandomDenseMatrix(&B);
    C = (float *)malloc(sizeof(float) * rows * rows);

    printf("A:\n");
    printPartialMatrix(A, rows, cols, 10, 10);
    printf("B:\n");
    printPartialMatrix(B, cols, rows, 10, 10);

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK(cudaMalloc((void **)&dA, sizeof(float) * rows * cols));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * cols * rows));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * rows * rows));

    // Transfer the input vectors and dense matrix A to the device
    CHECK(cudaMemcpy(dA, A, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * cols * rows, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0x00, sizeof(float) * rows * rows));

    // Construct a dense matrix descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateDnMat(&matA, rows, cols, cols, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Create sparse matrix A in CSR format
    int *dAcsrOffsets, *dAColumns;
    float *dAValues;
    CHECK(cudaMalloc((void**) &dAcsrOffsets, (rows + 1) * sizeof(int)));
    CHECK_CUSPARSE(cusparseCreateCsr(&Adescr, rows, cols, 0, dAcsrOffsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // allocate an external buffer if needed
    void* dBufferA    = NULL;
    size_t bufferSizeA = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, matA, Adescr,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSizeA));
    CHECK(cudaMalloc(&dBufferA, bufferSizeA));

    // Update the number of non-zero elements in the sparse matrix A
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, matA, Adescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBufferA));

    int64_t numRowsA, numColsA, totalANnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(Adescr, &numRowsA, &numColsA, &totalANnz));
    if (totalANnz != trueANnz) {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value in matrix A: expected %d but got %ld\n", trueANnz, totalANnz);
        return 1;
    }
    // Allocate device memory to store the sparse CSR representation of A
    CHECK(cudaMalloc((void **) &dAColumns, totalANnz * sizeof(int)));
    CHECK(cudaMalloc((void **) &dAValues, totalANnz * sizeof(float)));
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(Adescr, dAcsrOffsets, dAColumns, dAValues));
    // execute Dense to Sparse conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, matA, Adescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBufferA));

    // device memory deallocation
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dBufferA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA));

    /* Same process for matrix B */
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, cols, rows, rows, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    int *dBcsrOffsets, *dBColumns;
    float *dBValues;
    CHECK(cudaMalloc((void**) &dBcsrOffsets, (cols + 1) * sizeof(int)));
    CHECK_CUSPARSE(cusparseCreateCsr(&Bdescr, cols, rows, 0, dBcsrOffsets, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    void* dBufferB    = NULL;
    size_t bufferSizeB = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, matB, Bdescr,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSizeB));
    CHECK(cudaMalloc(&dBufferB, bufferSizeB));
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, matB, Bdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBufferB));

    // Compute the number of non-zero elements in B
    int64_t numRowsB, numColsB, totalBNnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(Bdescr, &numRowsB, &numColsB, &totalBNnz));
    if (totalBNnz != trueBNnz) {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value in matrix B: expected %d but got %ld\n", trueBNnz, totalBNnz);
        return 1;
    }

    CHECK(cudaMalloc((void **) &dBColumns, totalBNnz * sizeof(int)));
    CHECK(cudaMalloc((void **) &dBValues, totalBNnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(Bdescr, dBcsrOffsets, dBColumns, dBValues));
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, matB, Bdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBufferB));

    CHECK(cudaFree(dB));
    CHECK(cudaFree(dBufferB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));


    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&Cdescr, rows, rows, 0, NULL, NULL, NULL, 
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // perform matrix-matrix multiplication with the CSR-formatted matrix A
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    // ask bufferSize1 bytes for external memory
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, Adescr, Bdescr, &beta, Cdescr,
                                      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL));
    CHECK(cudaMalloc((void**) &dBuffer1, bufferSize1));
    // inspect the matrices A and B to understand the memory requirement for the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, Adescr, Bdescr, &beta, Cdescr,
                                      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1));

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, Adescr, Bdescr, &beta, Cdescr,
                               CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL));
    CHECK(cudaMalloc((void**) &dBuffer2, bufferSize2));

    // compute the intermediate product of A * B
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, Adescr, Bdescr, &beta, Cdescr,
                                           CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2));
    // get matrix C non-zero entries C_nnz1
    int64_t numRowsC, numColsC, totalCNnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(Cdescr, &numRowsC, &numColsC, &totalCNnz));
    // allocate matrix C
    int *dCcsrOffsets, *dCColumns;
    float *dCValues;
    CHECK(cudaMalloc((void**) &dCcsrOffsets, (rows + 1) * sizeof(int)));
    CHECK(cudaMalloc((void**) &dCColumns, totalCNnz * sizeof(int)));
    CHECK(cudaMalloc((void**) &dCValues,  totalCNnz * sizeof(float)));

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(Cdescr, dCcsrOffsets, dCColumns, dCValues))

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, Adescr, Bdescr, &beta, Cdescr,
                            CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    // Conver the sparse matrix C to dense matrix 
    cusparseDnMatDescr_t matC;
    void *dbuffer3 = NULL;
    size_t bufferSize3 = 0;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, rows, rows, rows, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(handle, Cdescr, matC, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize3));
    CHECK(cudaMalloc(&dbuffer3, bufferSize3));
    CHECK_CUSPARSE(cusparseSparseToDense(handle, Cdescr, matC, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, dbuffer3));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUSPARSE(cusparseDestroySpMat(Adescr));
    CHECK_CUSPARSE(cusparseDestroySpMat(Bdescr));
    CHECK_CUSPARSE(cusparseDestroySpMat(Cdescr));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // copy the result vector back to the host
    CHECK(cudaMemcpy(C, dC, sizeof(float) * rows * rows, cudaMemcpyDeviceToHost));

    printf("C:\n");
    printPartialMatrix(C, rows, rows, 10, 10);

    free(A);
    free(B);
    free(C);

    CHECK(cudaFree(dC));
    CHECK(cudaFree(dBuffer1));
    CHECK(cudaFree(dBuffer2));
    CHECK(cudaFree(dAcsrOffsets));
    CHECK(cudaFree(dAColumns));
    CHECK(cudaFree(dAValues));
    CHECK(cudaFree(dBcsrOffsets));
    CHECK(cudaFree(dBColumns));
    CHECK(cudaFree(dBValues));
    CHECK(cudaFree(dCcsrOffsets));
    CHECK(cudaFree(dCColumns));
    CHECK(cudaFree(dCValues));

    return EXIT_SUCCESS;

}