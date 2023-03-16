#!/bin/bash -e
export PYTHONPATH=$(pwd)
export NCCL_P2P_DISABLE=1

mpiexec -n 8 \
  python3 scripts/image_train.py \
    --data_dir=/data/bzj_trash/images/green \
    --log_interval=100 \
    --image_size=512 \
    --batch_size=1 \
    --save_interval=10000 \
    --log_dir=log/train/$(date +%s) --resume_checkpoint=log/train/model240000.pt

python3 scripts/image_sample.py \
  --num_samples=1 \
  --image_size=512 \
  --batch_size=1 \
  --model_path=log/train/model240000.pt \
  --log_dir=log/sample/$(date +%s)
