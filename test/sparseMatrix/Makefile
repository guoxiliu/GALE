CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse

all: mmCuSparse

mmCuSparse: mmCuSparse.cu
	nvcc $(INC) mmCuSparse.cu -o mmCuSparse $(LIBS)

clean:
	rm -f mmCuSparse

test:
	@echo "\n==== Sparse matrix-matrix multiplication example ====\n"
	./mmCuSparse

.PHONY: clean all test
