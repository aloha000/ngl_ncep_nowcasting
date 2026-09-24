#!/bin/bash
# 训练入口。所有参数原样转发给 train_FSDP.py：
#   CUDA_VISIBLE_DEVICES=0,1,2 MASTER_PORT=22336 nohup bash train.sh \
#       --model_id lead24h_haloarea_obszero --set zero_obs=true > /tmp/run.log 2>&1 &
#
# 产物落在 {work_dir}/{model_id}/ 下（work_dir / model_id 可在 configs.py 里改，
# 或用 --set work_dir=... --set model_id=... 覆盖）。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export MASTER_PORT=${MASTER_PORT:-22336}
echo "[train.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  MASTER_PORT=${MASTER_PORT}"
python train_FSDP.py --configs "${CONFIGS:-configs}" --master_port "${MASTER_PORT}" "$@"
