CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lpng -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt

train.out: ../data.o train.o
	$(CC) ../data.o train.o $(LDFLAGS) -o $@

../data.o: ../data.c ../data.h
	$(CC) $(CFLAGS) -c ../data.c -o $@

train.o: train.c ../data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o ../data.o
	rm -rf sample_images/

.PHONY: run clean