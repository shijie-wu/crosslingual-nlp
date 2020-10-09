#!/bin/bash
lngs=(en-ar en-de en-es en-fr en-hi en-ru en-vi en-zh)

bs=128
lr=1e-4

sim=l2
l2_param_coeff=1.0
l2_src_coeff=0.0
subset=1

model=${1:-xlm-roberta-base} # or bert-base-multilingual-cased

data=${2:-opus}

root_dir="${ROOT_DIR:-/bigdata}"
data_path="$root_dir"/dataset/bitext/clean/"$data"
save_path="$root_dir"/checkpoints/alignment/"$data"
cache_path="$root_dir"/cache/clnlp

python src/train.py \
    --task alignment \
    --data_dir "$data_path" \
    --trn_langs "${lngs[@]}" \
    --val_langs "${lngs[@]}" \
    --cache_dataset True \
    --cache_path "$cache_path" \
    --max_trn_len 96 \
    --max_tst_len 96 \
    --pretrain "$model" \
    --batch_size $bs \
    --learning_rate $lr \
    --adam_beta2 0.999 \
    --schedule linear \
    --max_steps 100000 --warmup_steps 4000 --val_check_interval 1000 \
    --input_dropout 0.0 \
    --aligner_sim $sim \
    --aligner_l2_param_coeff $l2_param_coeff \
    --aligner_l2_src_coeff $l2_src_coeff \
    --mix_sampling True \
    --patience 999999 \
    --gpus 1 \
    --precision 16 \
    --subset_ratio $subset \
    --default_save_path "$save_path"/"$(echo "${lngs[@]}" | tr ' ' ',')"-subset$subset/"$model"-sim_$sim \
    --exp_name l2src_$l2_src_coeff-l2param_$l2_param_coeff
