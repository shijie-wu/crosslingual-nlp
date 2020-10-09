#!/bin/bash

seed=${1:-42}
# or xlm-roberta-base/large
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"ner-conll"}
freeze=${4:-"-1"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

root_dir="${ROOT_DIR:-/bigdata}"
save_path="$root_dir"/checkpoints/crosslingual-nlp

case "$task" in
"mldoc")
    src="en"
    tgt=(de en es fr it ja ru zh)
    data_path="$root_dir"/dataset/"$task"
    ;;
"xnli")
    src="en"
    tgt=(ar bg de el en es fr hi ru sw th tr ur vi zh)
    data_path="$root_dir"/dataset/"$task"
    ;;
"ner-conll")
    src="en"
    tgt=(de en es nl zh)
    data_path="$root_dir"/dataset/"$task"
    ;;
"udpos")
    src="English"
    tgt=(Bulgarian Danish German English Spanish Persian Hungarian Italian Dutch Polish Portuguese Romanian Slovak Slovenian Swedish)
    data_path="$root_dir"/dataset/universaldependencies/ud-treebanks-v1.4
    ;;
"parsing")
    src="en"
    tgt=(ar bg ca cs da de en es et "fi" fr he hi hr id it ja ko la lv nl no pl pt ro ru sk sl sv uk zh)
    data_path="$root_dir"/dataset/universaldependencies/ud-treebanks-v2.2
    ;;
*)
    echo Unsupported task "$task"
    exit 1
    ;;
esac

for bs in 16 32; do
    for lr in 2e-5 3e-5 5e-5; do
        for ep in 3 4; do
            python src/train.py \
                --seed "$seed" \
                --task "$task" \
                --data_dir "$data_path" \
                --trn_langs $src \
                --val_langs $src \
                --tst_langs "${tgt[@]}" \
                --pretrain "$model" \
                --batch_size $bs \
                --learning_rate $lr \
                --max_epochs $ep \
                --warmup_portion 0.1 \
                --gpus 1 \
                --freeze_layer "$freeze" \
                --default_save_path "$save_path"/"$task"/0-shot-finetune-freeze"$freeze"/"$model_name" \
                --exp_name bs$bs-lr$lr-ep$ep
        done
    done
done
