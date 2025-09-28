CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lpng -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -lcurand

train.out: sim.o ../transformer/gpu/transformer.o ../transformer/attention/gpu/attention.o ../transformer/mlp/gpu/mlp.o ../data.o train.o
	$(CC) sim.o ../transformer/gpu/transformer.o ../transformer/attention/gpu/attention.o ../transformer/mlp/gpu/mlp.o ../data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

sim.o: sim.c sim.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c sim.c -o $@

../transformer/gpu/transformer.o:
	$(MAKE) -C ../transformer/gpu transformer.o

../transformer/attention/gpu/attention.o:
	$(MAKE) -C ../transformer/attention/gpu attention.o

../transformer/mlp/gpu/mlp.o:
	$(MAKE) -C ../transformer/mlp/gpu mlp.o

../data.o: ../data.c ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../data.c -o $@

train.o: train.c sim.h ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

cont: train.out
	@time ./train.out $$(ls -t *_sim.bin 2>/dev/null | head -n1)

clean:
	rm -f *.out *.png *.o ../data.o