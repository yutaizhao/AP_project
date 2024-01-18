CC=gcc

CFLAGS=-march=native -g3

OFLAGS=-Ofast -fopenmp -fopt-info-all=nbody.gcc.optrpt

all: nbody3D

nbody3D: nbody.c
	$(CC) $(CFLAGS) $(OFLAGS) $< -o $@ -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm -ldl

clean:
	rm -Rf *~ nbody3D *.optrpt

