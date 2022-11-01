#!/bin/bash

#make pthread2-overlap-dbg

#hosts="thorbf001:1,thor001:1 -H ${hosts} "
hosts="thorbf003:1,thor004:1 "

OPTS="-x STREAM_NUM_BUFFS=$2 -x STREAM_BUFF_SIZE=$3 "

hname=`hostname`

echo "======================== $hname ===================== "

set -x
mpirun -np 2 -H ${hosts} --bind-to none -mca coll ^hcoll -x STREAM_THREADS=${1} ${OPTS} -x UCX_TLS=dc_x -x LD_LIBRARY_PATH -x PATH /tmp/cross-stream2/STREAM-2/stream-net-pthread2-overlap.exe

#  1 512k
#  2 256k
#  4 128k
#  8  64k
# 16  32k
# 32  16k
# 64   8k
#128   4k

