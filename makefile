FILE=dot.c
CC=gcc-12
all: build

build:
	$(CC) $(FILE) -fopenmp -lm
	./a.out

build-optim:
	$(CC) $(FILE) -fopenmp -lm -Ofast -march=native
	./a.out

build-asm:
	$(CC) $(FILE) -fopenmp -lm -m64 -Ofast -march=native
	./a.out
