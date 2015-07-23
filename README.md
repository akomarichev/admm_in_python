# admm_lasso
I am trying to rewrite the code on Python. Here is a [link.](http://web.stanford.edu/~boyd/papers/admm/mpi/) 

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
