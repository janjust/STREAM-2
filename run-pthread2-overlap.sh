#!/bin/bash

rm -f stream-net.log.out
#make pthread2-overlap-dbg
make pthread2-overlap

#hosts="heliosbf001:1,heliosbf002:1 "
#hosts="helios001:1,helios002:1 "
hosts="thorbf001:1,thorbf002:1 "

OPTS="-x STREAM_NUM_BUFFS=$2 -x STREAM_BUFF_SIZE=$3 "

set -x
mpirun -np 2 -H ${hosts} --bind-to none -x STREAM_THREADS=${1} ${OPTS} -x UCX_TLS=dc_x ./stream-net-pthread2-overlap.exe

#  1 512k
#  2 256k
#  4 128k
#  8  64k
# 16  32k
# 32  16k
# 64   8k
#128   4k
