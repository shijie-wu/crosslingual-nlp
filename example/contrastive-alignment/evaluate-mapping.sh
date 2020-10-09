#!/bin/bash

seed=${1:-42}
# or xlm-roberta-base/large
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"xnli"}
mode=${4:-"finetune"}
mapping=${5:-linear-orth0.01}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

root_dir="${ROOT_DIR:-/bigdata}"
save_path="$root_dir"/checkpoints/crosslingual-nlp

case "$task" in
"xnli")
    src="en"
    tgt=(ar bg de el en es fr hi ru sw th tr ur vi zh)
    data_path="$root_dir"/dataset/"$task"
    ;;
"ner-wiki")
    src="en"
    tgt=(ar de en es fr hi ru vi zh-word)
    data_path="$root_dir"/dataset/"$task"
    ;;
"udpos" | "parsing")
    src="English-EWT"
    tgt=(Arabic-PADT German-GSD English-EWT Spanish-GSD French-GSD Hindi-HDTB Russian-GSD Vietnamese-VTB Chinese-GSD)
    data_path="$root_dir"/dataset/universaldependencies/ud-treebanks-v2.6
    ;;
*)
    echo Unsupported task "$task"
    exit 1
    ;;
esac

case "$mode" in
"feature")
    ep=20
    bs=128
    if [ "$task" = "xnli" ] || [ "$task" = "ner-wiki" ]; then
        lr=1e-4
    else
        lr=1e-3
    fi
    val_check_interval=1
    if [ "$task" = "ner-wiki" ]; then
        extra_args=(--projector transformer --projector_trm_num_layers 4 --freeze_layer 12 --weighted_feature True --tagger_use_crf True)
    else
        extra_args=(--projector transformer --projector_trm_num_layers 4 --freeze_layer 12 --weighted_feature True)
    fi
    ;;
*)
    echo Unsupported task "$mode"
    exit 1
    ;;
esac

for ckpt in "$save_path"/"$task"/0-shot-feature/"$model_name"/bs"$bs"-lr"$lr"-ep"$ep"/*/ckpts/*.ckpt; do
    echo "$ckpt"
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
        --val_check_interval $val_check_interval \
        --gpus 1 \
        --default_save_path "$save_path"/"$task"/0-shot-feature-map-"$mapping"/"$model_name" \
        --do_train False \
        --mapping mapping/"$mapping"/"$model_name".pth \
        --resume_from_checkpoint "$ckpt" \
        --exp_name bs$bs-lr$lr-ep$ep "${extra_args[@]}"
done
