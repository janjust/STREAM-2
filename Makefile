CC=gcc
MPICC=mpicc

CFLAGS = -I$(UCC_DIR)/include -I$(UCX_DIR)/include -std=c11 -fopenmp
LDFLAGS = -L$(UCC_DIR)/lib $(UCC_DIR)/lib/libucc.so $(UCX_DIR)/lib/libucs.so $(UCX_DIR)/lib/libucp.so -Wl,-rpath -Wl,$(UCC_DIR)/lib -Wl,-rpath -Wl,$(UCX_DIR)/lib -lpthread


#OPTFLAGS=-O2 -Wall -Wpedantic
#OPTFLAGS=-O3 -march=native -mavx -ffast-math
OPTFLAGS=-O3 -ftree-vectorize -ffast-math ##-mcpu=cortex-a72
OPTFLAGS-DBG=-O0 -g 

default:
	$(CC) $(OPTFLAGS) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream.exe stream.c $(LDFLAGS)

omp:
	$(MPICC) $(OPTFLAGS) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-omp.exe stream-net-omp.c $(LDFLAGS)

pthread:
	$(MPICC) $(OPTFLAGS) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread.exe stream-net-pthread.c $(LDFLAGS)

pthread-dbg:
	$(MPICC) $(OPTFLAGS-DBG) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread.exe stream-net-pthread.c $(LDFLAGS)

pthread-push:
	$(MPICC) $(OPTFLAGS) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread-push.exe stream-net-pthread-push.c $(LDFLAGS)

pthread-push-dbg:
	$(MPICC) $(OPTFLAGS-DBG) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread-push.exe stream-net-pthread-push.c $(LDFLAGS)


pthread2:
	$(MPICC) $(OPTFLAGS) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread2.exe stream-net-pthread2.c $(LDFLAGS)

pthread2-dbg:
	$(MPICC) $(OPTFLAGS-DBG) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread2.exe stream-net-pthread2.c $(LDFLAGS)

pthread2-overlap:
	$(MPICC) $(OPTFLAGS) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread2-overlap.exe stream-net-pthread2-overlap.c $(LDFLAGS)

pthread2-overlap-dbg:
	$(MPICC) $(OPTFLAGS-DBG) $(CFLAGS) -DOPTFLAGS='"$(OPTFLAGS)"' -o stream-net-pthread2-overlap.exe stream-net-pthread2-overlap.c $(LDFLAGS)


clean:
	rm -f stream *.exe
