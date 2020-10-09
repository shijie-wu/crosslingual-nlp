#!/bin/bash
src=en
tgt=${1:-zh}
pair=en-"$tgt"
root_dir="${ROOT_DIR:-/bigdata}"
data_path="$root_dir"/dataset/bitext

outdir="$data_path"/clean/opus
mkdir -p "$outdir"

declare -A bitexts=(
    ["en-ar"]="opus.ar-en"
    ["en-de"]="opus.de-en"
    ["en-es"]="opus.en-es"
    ["en-fr"]="opus.en-fr"
    ["en-hi"]="opus.en-hi"
    ["en-ru"]="opus.en-ru"
    ["en-vi"]="opus.en-vi"
    ["en-zh"]="opus.en-zh"
)

for mode in train dev test; do
    wget -P "$data_path"/opus-100 -c -nc http://data.statmt.org/opus-100-corpus/v1.0/supervised/"$pair"/"${bitexts[$pair]}"-$mode.$src
    wget -P "$data_path"/opus-100 -c -nc http://data.statmt.org/opus-100-corpus/v1.0/supervised/"$pair"/"${bitexts[$pair]}"-$mode."$tgt"
done

inp="$data_path"/opus-100/"${bitexts[$pair]}"
out="$outdir"/$src-"$tgt"

# nb_trn=$(wc -l "$inp"-train.$src | cut -d' ' -f1)
nb_val=$(wc -l "$inp"-dev.$src | cut -d' ' -f1)

if [ -f "$out".val.align ]; then
    echo file exists
    exit
fi

# concat
if [ ! -f "$out".$src ]; then
    cat "$inp"-dev.$src "$inp"-train.$src >"$out".$src
fi
if [ ! -f "$out"."$tgt" ]; then
    cat "$inp"-dev."$tgt" "$inp"-train."$tgt" >"$out"."$tgt"
fi

# tokenize
if [ ! -f "$out".$src.tok ]; then
    ./tools/tokenize.sh $src <"$out".$src >"$out".$src.tok
fi
if [ ! -f "$out"."$tgt".tok ]; then
    ./tools/tokenize.sh "$tgt" <"$out"."$tgt" >"$out"."$tgt".tok
fi

# concat
if [ ! -f "$out".pair ]; then
    python scripts/bitext-concat.py "$out".$src.tok "$out"."$tgt".tok >"$out".pair
fi

# align
if [ ! -f "$out".fwd.align ]; then
    "$data_path"/fast_align/build/fast_align -i "$out".pair -d -o -v >"$out".fwd.align
fi
if [ ! -f "$out".bwd.align ]; then
    "$data_path"/fast_align/build/fast_align -i "$out".pair -d -o -v -r >"$out".bwd.align
fi
if [ ! -f "$out".align ]; then
    "$data_path"/fast_align/build/atools \
        -i "$out".fwd.align \
        -j "$out".bwd.align \
        -c grow-diag-final-and >"$out".align
fi

if [ ! -f "$out".val.align ]; then
    python scripts/bitext-split.py "$out".pair "$out".align "$nb_val" "$out"
fi

if [ -f "$out".align ]; then
    rm -rf "$out".$src* "$out"."$tgt"* "$out".pair
    rm -rf "$out".bwd.align "$out".fwd.align "$out".align
fi
