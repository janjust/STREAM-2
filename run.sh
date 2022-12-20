#!/bin/bash

rm -f stream-net.log.out
make omp

OMP_OPTS="-x OMP_NUM_THREADS=${2} -x OMP_PROC_BIND=${3} " 

hosts="helios001:1,helios002:1 "
hosts="heliosbf001:1,heliosbf002:1 "

mpirun -np 2 -H ${hosts} --bind-to none -x STREAM_THREADS=${1} ${OMP_OPTS} -x UCX_TLS=rc_x ./stream-net-omp
