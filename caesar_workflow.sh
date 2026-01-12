#!/bin/bash
OPERATOR="CAESAR"
echo ""
echo "Step 2: Compressing all OG files with error bounds..."
for file in *OG.bp; do
    if [ -d "$file" ]; then  # Changed from -f to -d
        echo "  Compressing $file..."
        python3 compress.py --operator $OPERATOR --input $file --quiet
    fi
done
echo ""
echo "Step 3: Validating all compressed files..."
for file in *OG.bp; do
    if [ -d "$file" ]; then  # Changed from -f to -d
        basename=$(basename $file OG.bp)
        echo "  Validating $file..."
        python3 decomrpess.py --original $file --compressed $file --operator $OPERATOR --output ${basename}_validation.txt --quiet
    fi
done
echo ""
echo "=========================================="
echo "SUMMARY OF RESULTS"
echo "=========================================="
for vfile in *_validation.txt; do
    if [ -f "$vfile" ]; then
        echo ""
        echo "File: $vfile"
        if grep -q " OVERALL: PASSED" $vfile; then
            echo "   PASSED"
        else
            echo "   FAILED"
        fi
    fi
done
