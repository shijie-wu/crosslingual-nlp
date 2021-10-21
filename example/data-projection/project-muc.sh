#!/bin/bash
src="en"

split=${1:-"dev"}
tgt=${2:-"ar"}
encoder=${3:-"bert-base-multilingual-cased"}
align_layer=${4:-8}
align_system=${5:-"mbert_l8"}
mt_system="helsinki_opus"

max_len=500
#data from https://github.com/xinyadu/doc_event_role/tree/master/data/processed
DATA_DIR=""
#temp dir where outputs are saved in, after each step
DIR="intermediary/muc"
#path to where final projection file will be saved
FINAL_DIR="projection/muc"
mkdir -p $DIR $FINAL_DIR

if [ -f "$DIR/$src.$split.text" ]; then
    echo "$DIR/$src.$split.text exists."
else
    python scripts/extract-text.py \
        --task muc \
        --path "$DATA_DIR" \
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

python scripts/muc_convert_to_doc.py \
    --path "$DATA_DIR" \
    --lang $src \
    --split "$split" \
    --bitext "$DIR/$src.and_$tgt.$mt_system.$split.text" \
    --alignment "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.align" \
    --output_bitext "$DIR/$src.and_$tgt.$mt_system.$split.doc" \
    --output_alignment "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.doc.align"

python scripts/project-label.py \
    --task muc \
    --path "$DATA_DIR" \
    --lang $src \
    --split "$split" \
    --bitext "$DIR/$src.and_$tgt.$mt_system.$split.doc" \
    --alignment "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.doc.align" \
    --output_path $FINAL_DIR \
    --name "$tgt.from_$src.$mt_system.$align_system"
