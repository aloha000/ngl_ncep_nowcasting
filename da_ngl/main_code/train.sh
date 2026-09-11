#!/bin/bash
# GNSS/FuXi -> ERA5 assimilation training (FSDP, one process per visible GPU).
#
# Usage (run from anywhere; the script cd's to its own directory):
#   nohup bash main_code/train.sh > logs/fuxi_da.log 2>&1 &
#
# Override the visible GPUs / port by exporting them *before* nohup:
#   CUDA_VISIBLE_DEVICES=0,1,2 MASTER_PORT=22336 nohup bash main_code/train.sh > logs/fuxi_da.log 2>&1 &
set -e
cd "$(dirname "$0")"

# defaults: three GPUs, port 22336
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export MASTER_PORT=${MASTER_PORT:-22336}

echo "[train.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  MASTER_PORT=${MASTER_PORT}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "[train.sh] torch sees ${NGPU} GPU(s)"
if [ "${NGPU}" -lt 1 ]; then
    echo "[train.sh] ERROR: no visible CUDA device; check CUDA_VISIBLE_DEVICES / driver" >&2
    exit 1
fi

python train_FSDP.py --configs "${CONFIGS:-configs}" --master_port "${MASTER_PORT}"
