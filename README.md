# MPI implementation of an ADMM-based solver for the Lasso problem (python)
I am trying to rewrite [this code](http://web.stanford.edu/~boyd/papers/admm/mpi/) on Python. 

I have to change makefile slightly because original one doesn't work on my machine.

```makefile
MPICC=mpicc
CC=gcc
CFLAGS=-Wall -std=c99 
LDFLAGS=-lgsl -lgslcblas -lm

all: lasso

lasso: lasso.o mmio.o
	$(MPICC) $(CFLAGS) lasso.o mmio.o -o lasso $(LDFLAGS)

lasso.o: lasso.c mmio.o
	$(MPICC) $(CFLAGS) -c lasso.c

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c mmio.c

clean:
	rm -vf *.o lasso
```
