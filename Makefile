CC=gcc
CFLAGS=-std=c11 -fopenmp
LDFLAGS=

OPTFLAGS=-O2 -Wall -Wpedantic
#OPTFLAGS=-O3 -march=native -mavx -ffast-math
#OPTFLAGS=-O3 -ftree-vectorize -ffast-math -mcpu=cortex-a72

default:
	$(CC) $(OPTFLAGS) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream stream.c $(LDFLAGS)

clean:
	rm -f stream
