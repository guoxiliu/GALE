import numpy as np
import random
from scipy.sparse import csr_matrix
import time


# Function to generate a random sparse matrix in CSR format
def generateTopoMatrix(rows, cols, factor): 
    numNonZeros = factor * rows;
    # the value in the sparse matrix will only be 1
    csrVal = np.ones(numNonZeros, dtype=int)
    csrOffsets = np.zeros(rows + 1, dtype=int)
    # each row is expected to have `factor` values 
    for i in range(1, rows+1): 
         csrOffsets[i] = csrOffsets[i-1] + factor

    # column index should be the size of non-zero elements
    csrColumns = np.zeros(numNonZeros, dtype=int)
    for i in range(rows): 
        candidates = random.sample(range(cols), factor)
        for j in range(factor):
            csrColumns[i * factor + j] = candidates[j]

    return csrVal, csrOffsets, csrColumns

if __name__ == '__main__':
    tetraNumber = 1600000 * 2
    edgeNumber = 2100000 * 2
    vertexNumber = 340000 * 2

    # matrix a
    # indptr_a = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    # indices_a = np.array([2, 4, 5, 3, 4, 6, 0, 3, 4, 3, 2, 5, 4, 6, 2, 7, 6, 0, 2, 7, 6, 7, 0, 1, 3, 2, 0, 6, 1, 3, 2, 7, 3, 6, 0, 1, 3, 6, 2, 4])
    # data_a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    data_a, indptr_a, indices_a = generateTopoMatrix(tetraNumber, vertexNumber, 4)
    csr_a = csr_matrix((data_a, indices_a, indptr_a), shape=(tetraNumber, vertexNumber))
    # print("---------- Matrix A ----------")
    # print(csr_a.toarray())


    # matrix b
    # indptr_b = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    # indices_b = np.array([3, 4, 4, 1, 4, 0, 3, 7, 4, 7, 1, 0, 7, 0, 1, 2, 2, 0, 6, 0, 4, 0, 5, 7, 7, 0, 1, 3, 3, 6, 7, 4])
    # data_b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    data_b, indptr_b, indices_b = generateTopoMatrix(edgeNumber, vertexNumber, 2)
    csr_b = csr_matrix((data_b, indices_b, indptr_b), shape=(edgeNumber, vertexNumber))

    start = time.time()
    csr_b_trans = csr_b.transpose()
    # print("---------- Matrix B ----------")
    # print(csr_b.toarray())

    # matrix c
    csr_c = csr_a.dot(csr_b_trans)
    end = time.time()
    diff = end - start
    print("Done, time used: %.6f seconds" % diff)

    # # Check if the results are equal
    # c = np.dot(csr_a.toarray(), csr_b.toarray().transpose())
    # print(np.all(c == csr_c.toarray()))            

    # # Print out the matrix C in CSR format
    # print("data:", csr_c.data)
    # print("indptr:", csr_c.indptr)
    # print("indices: ", csr_c.indices)

    # # Print out the matrix C in dense format
    # print("---------- Matrix C ----------")
    # print(c)
