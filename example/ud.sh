#!/bin/bash
seed=${1:-42}
model=${2:-bert-base-multilingual-cased} # or xlm-roberta-base/large or xlm-mlm-100-1280

src="English-EWT"            # training & dev languages, could be more than 1 and decoupled
tgt=(English-EWT French-GSD) # test languages
task=${3:-"parsing"}         # change it to "udpos" to run ud pos tagging instead

ep=5
bs=32
lr=2e-5

root_dir="${ROOT_DIR:-/bigdata}"
data_path="$root_dir"/dataset/universaldependencies/ud-treebanks-v2.6
save_path="$root_dir"/checkpoints/crosslingual-nlp

# train 10k steps, wramup 1k steps, validate every 200 steps
# --max_steps 10000 --warmup_steps 1000 --val_check_interval 200

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
    --val_check_interval 0.25 \
    --gpus 1 \
    --default_save_path "$save_path"/"$task"/0-shot/"$model" \
    --exp_name bs$bs-lr$lr-ep$ep
