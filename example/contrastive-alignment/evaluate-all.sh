#!/bin/bash
mode=${1:-"finetune"}                      # or "feature"
model=${2:-"bert-base-multilingual-cased"} # or xlm-roberta-base, xlm-roberta-large, /path/to/your/model

script_path=example/contrastive-alignment

for seed in 12 156 78 37 8; do # predefined seeds

    bash $script_path/evaluate.sh "$seed" "$model" xnli "$mode"
    bash $script_path/evaluate.sh "$seed" "$model" ner-wiki "$mode"
    bash $script_path/evaluate.sh "$seed" "$model" udpos "$mode"
    bash $script_path/evaluate.sh "$seed" "$model" parsing "$mode"

done
