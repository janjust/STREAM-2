#!/bin/bash
rm -rf /tmp/cross-stream2
mkdir /tmp/cross-stream2

cd /tmp/cross-stream2/
cp -r /global/scratch/users/tomislavj/dpu-dev/cross-arch/hpcx/hpcx-v2.12-gcc-MLNX_OFED_LINUX-5-redhat8-cuda11-gdrcopy2-nccl2.12-aarch64.tbz .
tar -xvf hpcx-v2.12-gcc-MLNX_OFED_LINUX-5-redhat8-cuda11-gdrcopy2-nccl2.12-aarch64.tbz
mv hpcx-v2.12-gcc-MLNX_OFED_LINUX-5-redhat8-cuda11-gdrcopy2-nccl2.12-aarch64 hpcx

cp -r /global/scratch/users/tomislavj/dpu-dev/tests/STREAM-2 .

source hpcx/hpcx-mt-init.sh

hpcx_load

cd STREAM-2

cp /global/scratch/users/tomislavj/dpu-dev/cross-arch/Makefile .
cp /global/scratch/users/tomislavj/dpu-dev/cross-arch/source.sh .

make pthread2-overlap

ls
