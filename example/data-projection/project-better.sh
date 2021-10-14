#!/bin/bash
src="en"

split=${1:-"analysis"}
tgt=${2:-"ar"}
encoder=${3:-"bert-base-multilingual-cased"}
align_layer=${4:-8}
align_system=${5:-"mbert_l8"}
subtask=${6:-"abstract"}
mt_system="helsinki_opus"

max_len=500
#better data dir
DATA_DIR=""
#temp dir where outputs are saved in, after each step
DIR="intermediary/better-$subtask"
#path to where final projection file will be saved
FINAL_DIR="projection/better-$subtask"
mkdir -p "$DIR" "$FINAL_DIR"

if [ -f "$DIR/$split.tok" ]; then
    echo "$DIR/$split.tok exists."
else
    mkdir -p "$DIR"
    python scripts/tokenize_en.py \
        --bpjson "$DATA_DIR/$subtask-8d-inclusive.$split.update2.bp.json" \
        --output "$DIR/$split.tok"
fi

if [ -f "$DIR/$src.$split.text" ]; then
    echo "$DIR/$src.$split.text exists."
else
    python scripts/extract-text.py \
        --task "better-$subtask" \
        --path "$DIR" \
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
    --task "better-$subtask" \
    --path "$DIR" \
    --lang $src \
    --split "$split" \
    --bitext "$DIR/$src.and_$tgt.$mt_system.$split.text" \
    --alignment "$DIR/$src.and_$tgt.$mt_system.$align_system.$split.align" \
    --output_path "$DIR" \
    --name "$tgt.from_$src.$mt_system.$align_system"

python scripts/filter_bpjson.py \
    --input "$DIR/silver-temp-$split.json" \
    --outputdir "$DIR"

python scripts/tokenize_en.py \
    --bpjson "$DIR/silver-temp-$split.valid.bp.json" \
    --output "$FINAL_DIR/$subtask.$split.silver"
