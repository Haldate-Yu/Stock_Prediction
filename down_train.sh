#!/bin/bash


for lr in 1e-2 1e-3 5e-4; do
  for wd in 0; do
    for pad_type in timely; do
        python main.py \
          --epochs 300 \
          --lr $lr \
          --weight_decay $wd \
          --padding_type $pad_type \
          --sample_type down
    done
  done
done
