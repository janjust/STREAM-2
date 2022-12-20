#!/bin/bash

rm -f stream-net.log.out
make pthread

hosts="helios001:1,helios002:1 "
hosts="heliosbf001:1,heliosbf002:1 "
hosts="thorbf001:1,thorbf002:1 "

mpirun -np 2 -H ${hosts} --bind-to none -x STREAM_THREADS=${1} -x UCX_TLS=rc_x ./stream-net-pthread.exe
