#!/bin/bash
# GNSS/FuXi -> ERA5 assimilation training (FSDP, one process per visible GPU).
#
# Usage (run from anywhere; the script cd's to its own directory):
#   nohup bash main_code/train.sh > logs/fuxi_da.log 2>&1 &
#
# Override the visible GPUs / port by exporting them *before* nohup:
#   CUDA_VISIBLE_DEVICES=0,1,2 MASTER_PORT=22336 nohup bash main_code/train.sh > /dev/null 2>&1 &
#
# Progress goes to da_ngl/logs/{model_id}_{模型配置}.log; extra arguments
# (--model_id / --exp_tag / --arch_tag) are forwarded to train_FSDP.py, e.g.
#   bash main_code/train.sh --model_id exp3 --exp_tag obs6h_era5tp
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

python train_FSDP.py --configs "${CONFIGS:-configs}" --master_port "${MASTER_PORT}" "$@"

# ---------------------------------------------------------------------------
# Chained evaluation + plots on the run that just finished.  One GPU is enough
# and the same arguments are forwarded (--model_id / --exp_tag / --arch_tag /
# --set ...), so `plot_results.py` resolves exactly the folder the training
# wrote to (checkpoint = the val-best one).  Knobs:
#   SKIP_EVAL=1            skip the evaluation
#   EVAL_SPLIT=val|test    which split to evaluate (default test)
#   EVAL_GPUS=0            which visible GPU to use for the evaluation
# ---------------------------------------------------------------------------
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo "[train.sh] training done -> evaluating the val-best checkpoint (CUDA_VISIBLE_DEVICES=${EVAL_GPUS:-0})"
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS:-0}" python plot_results.py \
        --configs "${CONFIGS:-configs}" "$@" --split "${EVAL_SPLIT:-test}"
    echo "[train.sh] evaluation done -> $(ls -dt "${PWD}"/work_dir/results/*/ 2>/dev/null | head -1)"
fi
echo "[train.sh] all done"
