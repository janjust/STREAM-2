#!/bin/bash

make stream-pt

export STREAM_THREADS=${1}

perf stat \
    -e L1-dcache-loads,L1-dcache-load-misses \
    -e l1d_cache_rd,l1d_cache_wr,l1d_cache_inval,l1d_cache_wb_victim \
    -e l2d_cache_rd,l2d_cache_refill_rd,l2d_cache_wr,l2d_cache_refill_wr,l2d_cache_wb_clean,l2d_cache_wb_victim,l2d_cache_inval,l2d_cache_wb_victim \
	-e dTLB-load-misses \
    -e mem_access_rd,mem_access_wr \
	/global/scratch/users/tomislavj/dpu-dev/tests/STREAM-2/stream-pt.exe
