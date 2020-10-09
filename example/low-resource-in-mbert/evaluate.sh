#!/bin/bash

seed=${1:-42}
# or xlm-roberta-base/large
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"ner-wiki"}
lang=${4:-"en"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

root_dir="${ROOT_DIR:-/bigdata}"
save_path="$root_dir"/checkpoints/crosslingual-nlp

case "$task" in
"ner-wiki")
    data_path="$root_dir"/dataset/ner-wiki
    ;;
"udpos" | "parsing")
    data_path="$root_dir"/dataset/universaldependencies/ud-treebanks-v2.3
    ;;
*)
    echo Unsupported task "$task"
    exit 1
    ;;
esac

for bs in 16 32; do
    for lr in 2e-5 3e-5 5e-5; do
        python src/train.py \
            --seed "$seed" \
            --task "$task" \
            --data_dir "$data_path" \
            --trn_langs "$lang" \
            --val_langs "$lang" \
            --tst_langs "$lang" \
            --pretrain "$model" \
            --batch_size $bs \
            --learning_rate $lr \
            --max_steps 10000 --warmup_steps 1000 --val_check_interval 200 \
            --gpus 1 \
            --default_save_path "$save_path"/"$task"/monolingual/"$lang"/"$model_name" \
            --exp_name bs$bs-lr$lr-step10k
    done
done
