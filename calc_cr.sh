#!/usr/bin/env bash

# Usage: ./calc_cr.sh logfile.txt

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

FILE="$1"

total_original=0
total_compressed=0

while read -r line; do
    if [[ "$line" =~ Original\ data\ size:\ ([0-9]+)\ bytes ]]; then
        size="${BASH_REMATCH[1]}"
        total_original=$((total_original + size))
    elif [[ "$line" =~ Total\ compressed\ size:\ ([0-9]+)\ bytes ]]; then
        size="${BASH_REMATCH[1]}"
        total_compressed=$((total_compressed + size))
    fi
done < "$FILE"

if [[ "$total_compressed" -eq 0 ]]; then
    echo "Error: total compressed size is zero"
    exit 1
fi


cr=$(awk "BEGIN { printf \"%.6f\", $total_original / $total_compressed }")

echo "Total Original Size   : $total_original bytes"
echo "Total Compressed Size : $total_compressed bytes"
echo "Compression Ratio (CR): $cr"

