#!/bin/bash

rm -f stream-net.log.out
#make pthread2-overlap-dbg
make pthread2-overlap

hosts="heliosbf001:1,heliosbf002:1 "
#hosts="helios001:1,helios002:1 "

OPS="-x STREAM_NUM_BUFFS=8 -x STREAM_BUFF_SIZE=$((64 * 1024)) "

mpirun -np 2 -H ${hosts} --bind-to none -x STREAM_THREADS=${1} ${OPS} -x UCX_TLS=rc_x ./stream-net-pthread2-overlap.exe

#  1 512k
#  2 256k
#  4 128k
#  8  64k
# 16  32k
# 32  16k
# 64   8k
#128   4k