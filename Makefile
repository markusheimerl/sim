CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto -ljpeg -lfftw3

data.out: data.c
	$(CC) $(CFLAGS) $< -lcurl -ljansson $(LDFLAGS) -o $@

sigm.out: sigm.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

data: data.out
	@time ./data.out

run: sigm.out
	@time ./sigm.out

clean:
	rm -f *.out *.bin *.jpg