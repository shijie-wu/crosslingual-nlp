#!/bin/bash

seed=${1:-42}
task="langid"
model=${2:-bert-base-multilingual-cased} # or xlm-roberta-base or xlm-mlm-100-1280
feature=${3:-6}

ep=10
bs=32
lr=1e-3

root_dir="${ROOT_DIR:-/bigdata}"
data_path="$root_dir"/dataset/langid
save_path="$root_dir"/checkpoints/crosslingual-nlp

python src/train.py \
    --seed "$seed" \
    --task $task \
    --data_dir "$data_path" \
    --trn_langs all \
    --val_langs all \
    --tst_langs all \
    --pretrain "$model" \
    --freeze_layer 12 --feature_layer "$feature" --projector meanpool \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --schedule reduceOnPlateau \
    --gpus 1 \
    --default_save_path "$save_path"/$task/"$model" \
    --exp_name feature_"$feature"
