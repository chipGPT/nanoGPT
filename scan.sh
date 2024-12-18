#!/bin/bash

n_layer="10"
for i in `seq 1 10`; do
  python3 train.py \
  --device=cpu \
  --dataset="shakespeare_char" \
  --n_layer="$n_layer" \
  --n_embd=16 \
  --n_head=2 \
  --max_iters=500 \
  --compile \
  --big_block_after_layer "$i" \
  --out_dir "${n_layer}_${i}"
done
