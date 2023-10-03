#!/bin/bash

hosts="heliosbf2a002:1,helios002 "

OPTS="-x STREAM_THREADS=$1 -x STREAM_NUM_BUFFS=$2 -x STREAM_BUFF_SIZE=$3 "

set -x
mpirun -np 2 -H ${hosts} --bind-to none ${OPTS} -x UCX_TLS=rc_x -x LD_LIBRARY_PATH ./stream-pt.exe

#  1 512k
#  2 256k
#  4 128k
#  8  64k
# 16  32k
# 32  16k
# 64   8k
#128   4k
