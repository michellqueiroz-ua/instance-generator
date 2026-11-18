#!/bin/bash
# Find the instances of a batch that contain incomplete requests (a request whose location is empty) and
# generate them again. Run on a login node.
#   bash regenerate_incomplete.sh <batch>            list them only (changes nothing)
#   bash regenerate_incomplete.sh <batch> --redo     move their csv/json to <work>/incomplete_backup and resubmit
# Each resubmitted instance is one single-core job; MEM_GB (default 4) and WALL_H (default 1) as in generate_batch.sh.
BATCH=$(printf "%02d" "$1")
MODE=$2
WORK=${ISA_WORK:-/work3/$USER/isa}
NET="$WORK/Capital Region of Denmark"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BACKUP="$WORK/incomplete_backup/batch_$BATCH"
found=0
for csv in "$NET"/csv_format/ISA_b${BATCH}_i*_1.csv; do
    # field 4 is the origin; an incomplete request has nothing after its reaction time
    bad=$(awk -F, 'NR>1 && $4=="" {c++} END {print c+0}' "$csv")
    [ "$bad" -eq 0 ] && continue
    name=$(basename "$csv" _1.csv)
    number=$((10#${name##*_i}))
    found=$((found + 1))
    echo "$name: $bad incomplete of $(($(wc -l < "$csv") - 1)) requests"
    if [ "$MODE" = "--redo" ]; then
        mkdir -p "$BACKUP"
        mv "$csv" "$BACKUP/" && mv "$NET/json_format/${name}_1.json" "$BACKUP/" 2>/dev/null
        bash "$HERE/generate_batch.sh" "$1" "$number-$number" | tail -1
    fi
done
echo "$found instances of batch $BATCH have incomplete requests"
