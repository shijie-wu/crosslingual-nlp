#!/bin/bash
task=${1:-ud}
split=${2:-train}
lang=${3:-ar}
encoder=${4:-bert-base-multilingual-cased}

root_dir="${ROOT_DIR:-/bigdata}"
model_path="$root_dir"/checkpoints/clnlp/en.0shot

batch_size=8
encoder_long_name=$(echo "$encoder" | cut -d/ -f2)
case "$encoder" in
"bert-base-multilingual-cased") encoder_short_name=mbert ;;
"xlm-roberta-large") encoder_short_name=xlmrl ;;
"lanwuwei/GigaBERT-v4-Arabic-and-English") encoder_short_name=giga4 ;;
"jhu-clsp/roberta-large-eng-ara-128k") encoder_short_name=L128K ;;
*)
    echo ERROR
    exit
    ;;
esac
name="$lang.from_en.helsinki_opus.self_$encoder_short_name"

if [ "$task" = "ner" ]; then
    path="projection/ner-wiki/$name"
    mkdir -p "$path"
    python src/predict.py \
        --filepath "intermediary/ner-wiki/en.to_$lang.helsinki_opus.$split.text" \
        --encoder "$encoder" \
        --lang "$lang" \
        --task "$task" \
        --batch_size $batch_size \
        --tagger_path "$model_path/ner-wiki/$encoder_long_name/**/*.ckpt" \
        --output_file "$path/$split"
elif [ "$task" = "ud" ]; then
    path="projection/ud2.7/UD_$name"
    mkdir -p "$path"
    python src/predict.py \
        --filepath "intermediary/ud2.7/en.to_$lang.helsinki_opus.$split.text" \
        --encoder "$encoder" \
        --lang "$lang" \
        --task "$task" \
        --batch_size $batch_size \
        --parser_path "$model_path/parsing/$encoder_long_name/**/*.ckpt" \
        --tagger_path "$model_path/udpos/$encoder_long_name/**/*.ckpt" \
        --output_file "$path/$name-ud-$split.conllu"
elif [ "$task" = "udpos" ]; then
    path="projection/ud2.7-pos/UD_$name"
    mkdir -p "$path"
    python src/predict.py \
        --filepath "intermediary/ud2.7/en.to_$lang.helsinki_opus.$split.text" \
        --encoder "$encoder" \
        --lang "$lang" \
        --task "$task" \
        --batch_size $batch_size \
        --tagger_path "$model_path/udpos/$encoder_long_name/**/*.ckpt" \
        --output_file "$path/$name-ud-$split.conllu"
else
    echo ERROR
fi
