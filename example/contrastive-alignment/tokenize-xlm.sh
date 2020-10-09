#!/bin/bash
src=en
tgt=${1:-zh}
pair=en-"$tgt"
root_dir="${ROOT_DIR:-/bigdata}"
data_path="$root_dir"/dataset/bitext

outdir="$data_path"/clean/xlm
mkdir -p "$outdir"

declare -A bitexts=(
    ["en-ar"]="MultiUN.ar-en"
    ["en-de"]="EUbookshop.de-en"
    ["en-es"]="MultiUN.en-es"
    ["en-fr"]="MultiUN.en-fr"
    ["en-hi"]="IITB.en-hi"
    ["en-ru"]="MultiUN.en-ru"
    ["en-vi"]="OpenSubtitles.en-vi"
    ["en-zh"]="MultiUN.en-zh"
)

inp="$data_path"/XLM/data/para/${bitexts[$pair]}
out="$outdir"/$pair

if [ ! -f "$inp".$src ]; then
    echo bitext "$inp".$src not found
    exit
fi

# nb_trn=1000000
nb_val=2000
nb_all=1002000
seed=42

if [ -f "$out".val.align ]; then
    echo file exists
    exit
fi

# subset
get_seeded_random() {
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
        </dev/zero 2>/dev/null
}

shuf --random-source=<(get_seeded_random 42) "$inp".$src | head -n $nb_all >"$out".$src
shuf --random-source=<(get_seeded_random 42) "$inp"."$tgt" | head -n $nb_all >"$out"."$tgt"

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
    python scripts/bitext-split.py "$out".pair "$out".align $nb_val "$out"
fi

if [ -f "$out".align ]; then
    rm -rf "$out".$src* "$out"."$tgt"* "$out".pair
    rm -rf "$out".bwd.align "$out".fwd.align "$out".align
fi
