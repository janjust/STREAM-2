CC=gcc
MPICC=mpicc

CFLAGS_MPI = -I$(UCC_DIR)/include -I$(UCX_DIR)/include -std=c11 -fopenmp
LDFLAGS_MPI = -L$(UCC_DIR)/lib $(UCC_DIR)/lib/libucc.so $(UCX_DIR)/lib/libucs.so $(UCX_DIR)/lib/libucp.so -Wl,-rpath -Wl,$(UCC_DIR)/lib -Wl,-rpath -Wl,$(UCX_DIR)/lib -lpthread

#OPTFLAGS=-O2 -Wall -Wpedantic
#OPTFLAGS=-O3 -march=native -mavx -ffast-math
OPTFLAGS=-O3 -ftree-vectorize -ffast-math ##-mcpu=cortex-a72
#OPTFLAGS=-O0 -g

stream:
	$(CC) $(OPTFLAGS) -std=c11 -fopenmp -DOPTFLAGS='"$(OPTFLAGS)"' -o stream.exe stream.c

stream-pt:
	$(CC) $(OPTFLAGS) -std=c11 -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-pt.exe stream-pt.c -lpthread

omp-simple:
	$(MPICC) $(OPTFLAGS) $(CFLAGS_MPI) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-pt.exe stream-omp.c $(LDFLAGS_MPI)

pt-simple:
	$(MPICC) $(OPTFLAGS) $(CFLAGS_MPI) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-pt.exe stream-pt-simple.c $(LDFLAGS_MPI)

pt-multi:
	$(MPICC) $(OPTFLAGS) $(CFLAGS_MPI) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-pt.exe stream-pt-multi.c $(LDFLAGS_MPI)

pt-multi-pipe:
	$(MPICC) $(OPTFLAGS) $(CFLAGS_MPI) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-pt.exe stream-pt-multi-pipe.c $(LDFLAGS_MPI)

clean:
	rm -f stream *.exe
