#!/bin/bash
file=$1
fast_align -i "$file.text" -d -o -v >"$file.forward.align"
fast_align -i "$file.text" -d -o -v -r >"$file.reverse.align"
atools -i "$file.forward.align" -j "$file.reverse.align" -c grow-diag-final-and >"$file.align"
