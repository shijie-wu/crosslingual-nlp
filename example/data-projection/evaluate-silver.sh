#!/bin/bash

seed=${1:-42}
# or xlm-roberta-large
encoder=${2:-"bert-base-multilingual-cased"}
task=${3:-"ner-wiki"}
system=${4:-"helsinki_opus.mbert_l8"}
lng=${5:-"ar"}

root_dir="${ROOT_DIR:-/bigdata}"
save_path="$root_dir"/checkpoints/clnlp/en-$lng.silver.$system
encoder_name=$(echo "$encoder" | cut -d/ -f2)

case "$task" in
"ner-wiki")
    case "$lng" in
    "zh") lng_name="zh-word" ;;
    *) lng_name="$lng" ;;
    esac
    src=("$lng.from_en.$system" en)
    tgt=("$lng_name" en)
    data_path="$root_dir"/data/"$task"
    ;;
"udpos" | "parsing")
    case "$lng" in
    "ar") lng_name="Arabic-PADT" ;;
    "de") lng_name="German-GSD" ;;
    "en") lng_name="English-EWT" ;;
    "es") lng_name="Spanish-GSD" ;;
    "fr") lng_name="French-GSD" ;;
    "hi") lng_name="Hindi-HDTB" ;;
    "ru") lng_name="Russian-GSD" ;;
    "vi") lng_name="Vietnamese-VTB" ;;
    "zh") lng_name="Chinese-GSD" ;;
    *) lng_name="$lng" ;;
    esac
    src=("$lng.from_en.$system" English-EWT)
    tgt=("$lng_name" English-EWT)
    data_path="$root_dir"/data/universaldependencies/ud-treebanks-v2.7
    ;;
*)
    echo Unsupported task "$task"
    exit 1
    ;;
esac

ep=5
bs=32
lr=2e-5
val_check_interval=0.25

python src/train.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs "${src[@]}" \
    --val_langs "${src[@]}" \
    --tst_langs "${tgt[@]}" \
    --pretrain "$encoder" \
    --batch_size $bs \
    --eval_batch_size 16 \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --val_check_interval $val_check_interval \
    --gpus 1 \
    --precision 16 \
    --default_save_path "$save_path/$task/$encoder_name" \
    --exp_name "bs$bs-lr$lr-ep$ep-fp16/seed-$seed"
