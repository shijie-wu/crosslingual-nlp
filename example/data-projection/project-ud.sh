#!/bin/bash
src="en"

split=${1:-"train"}
tgt=${2:-"ar"}
encoder=${3:-"bert-base-multilingual-cased"}
align_layer=${4:-8}
align_system=${5:-"mbert_l8"}
mt_system="helsinki_opus"

max_len=500
#temp dir where outputs are saved in, after each step
DIR="intermediary/ud2.7"
#path to where final projection file will be saved
FINAL_DIR="projection/ud2.7"
mkdir -p $DIR $FINAL_DIR

if [ -f "$DIR/$src.$split.text" ]; then
    echo "$DIR/$src.$split.text exists."
else
    python scripts/extract-text.py \
        --task ud \
        --path /bigdata/dataset/universaldependencies/ud-treebanks-v2.7 \
        --lang $src \
        --split "$split" \
        >"$DIR/$src.$split.text"
fi

if [ -f "$DIR/$src.to_$tgt.$mt_system.$split.text" ]; then
    echo "$DIR/$src.to_$tgt.$mt_system.$split.text exists."
else
    python scripts/translate.py \
        --infile "$DIR/$src.$split.text" \
        --model_name "Helsinki-NLP/opus-mt-$src-$tgt" \
        --src $src \
        --tgt "$tgt" \
        >"$DIR/$src.to_$tgt.$mt_system.$split.text"
fi

if [ -f "$DIR/$src.and_$tgt.$mt_system.$split.text" ]; then
    echo "$DIR/$src.and_$tgt.$mt_system.$split.text exists."
else
    python scripts/bitext-concat.py \
        --src_fp "$DIR/$src.$split.text" \
        --tgt_fp "$DIR/$src.to_$tgt.$mt_system.$split.text" \
        >"$DIR/$src.and_$tgt.$mt_system.$split.text"
fi

if [ -f "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.align" ]; then
    echo "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.align exists."
else
    python scripts/awesome-align.py \
        --data_file "$DIR/$src.and_$tgt.$mt_system.$split.text" \
        --align_layer "$align_layer" \
        --model_name_or_path "$encoder" \
        --max_len $max_len \
        --output_file "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.align"
fi

python scripts/project-label.py \
    --task ud \
    --path /bigdata/dataset/universaldependencies/ud-treebanks-v2.7 \
    --lang $src \
    --split "$split" \
    --bitext "$DIR/$src.and_$tgt.$mt_system.$split.text" \
    --alignment "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.align" \
    --output_path $FINAL_DIR \
    --name "$tgt.from_$src.$mt_system.$align_system"
