#!/bin/bash

python run_instance_ROME_ZsRE.py \
    --editing_method Rome \
    --hparams_dir ../hparams/ROME/ \
    --data_dir ./data/ZsRE/benchmark_ZsRE_ZsRE-test-all.json \
    --ds_size 10 \
    --metrics_save_dir ./output/ \
    --datatype ZsRE \
    --train_data_path data/ZsRE/train.json \




