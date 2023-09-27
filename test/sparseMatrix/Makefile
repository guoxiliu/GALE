CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse

all: mmCuSparseDense cuSparseTopo

mmCuSparseDense: mmCuSparseDense.cu
	nvcc $(INC) mmCuSparseDense.cu -O3 -o mmCuSparseDense $(LIBS)

cuSparseTopo: cuSparseTopo.cu
	nvcc $(INC) cuSparseTopo.cu -O3 -o cuSparseTopo $(LIBS)

clean:
	rm -f mmCuSparseDense cuSparseTopo

test:
	# @echo "\n==== Sparse matrix-matrix multiplication example ====\n"
	# ./mmCuSparseDense

	@echo "\n==== Sparse topological matrix example ====\n"
	./cuSparseTopo


.PHONY: clean all test
