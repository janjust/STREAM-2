#!/bin/bash

make stream

set -x
export OMP_NUM_THREADS=${1}
export OMP_PROC_BIND=${2}
./stream.exe
