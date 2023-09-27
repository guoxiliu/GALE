import numpy as np
import time

def matrixMultiply(m_size): 
    matrixA = np.random.rand(m_size, m_size)
    matrixB = np.random.rand(m_size, m_size)

    start = time.time()
    nIter = 30
    for i in range(30): 
        matrixC = np.matmul(matrixA, matrixB)
    end = time.time()
    diff = end - start
    print("Matrix size:", m_size)
    print("Done, time = %.6f seconds" % (diff/nIter))

if __name__ == '__main__':
    test_size = 1920
    matrixMultiply(test_size)
