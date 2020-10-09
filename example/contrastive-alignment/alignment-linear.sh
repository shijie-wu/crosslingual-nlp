#!/bin/bash
lngs=${2:-"en-ar"} # en-de en-es en-fr en-hi en-ru en-vi en-zh"

bs=128
lr=1e-4

sim=linear
subset=1

orth=0.01 # 0.01 is better than 0 or 0.001

model=${1:-xlm-roberta-base} # or bert-base-multilingual-cased

data=${3:-opus}

root_dir="${ROOT_DIR:-/bigdata}"
data_path="$root_dir"/dataset/bitext/clean/"$data"
save_path="$root_dir"/checkpoints/alignment/"$data"
cache_path="$root_dir"/cache/clnlp

python src/train.py \
    --task alignment \
    --data_dir "$data_path" \
    --trn_langs "$lngs" \
    --val_langs "$lngs" \
    --cache_dataset True \
    --cache_path "$cache_path" \
    --max_trn_len 96 \
    --max_tst_len 96 \
    --pretrain "$model" \
    --batch_size $bs \
    --learning_rate $lr \
    --adam_beta2 0.999 \
    --schedule linear \
    --max_steps 20000 --warmup_steps 1 --val_check_interval 100 \
    --input_dropout 0.0 \
    --freeze_layer 12 \
    --aligner_sim $sim \
    --aligner_orthogonal $orth \
    --patience 999999 \
    --gpus 1 \
    --precision 16 \
    --subset_ratio $subset \
    --default_save_path "$save_path"/"$(echo "$lngs" | tr ' ' ',')"-subset$subset/"$model"-sim_"$sim" \
    --exp_name linear-orth$orth
