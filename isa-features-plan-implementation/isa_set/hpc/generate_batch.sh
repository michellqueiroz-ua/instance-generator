#!/bin/bash
# Generate one batch of ISA instances on the DTU HPC: one single-core LSF job per config.
# Run on a login node after make_batch.py:
#   bash generate_batch.sh <batch> [first-last]    e.g.  bash generate_batch.sh 1      bash generate_batch.sh 1 1-5
# Every job runs generate_one.sh with its config path as the argument. It reserves 1 core and MEM_GB
# (default 8) for that core. Size MEM_GB and WALL_H from the "Max Memory" and "Run time" of earlier logs.
# Output: <work>/Capital Region of Denmark/csv_format/<instance>_1.csv; logs <work>/logs/batch_XX/gen_<instance>.out
set -e
BATCH=$(printf "%02d" "$1")
WORK=${ISA_WORK:-/work3/$USER/isa}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIST="$WORK/configs/batch_$BATCH/list.txt"
[ -s "$LIST" ] || { echo "no $LIST: run make_batch.py --batch $1 first"; exit 1; }
N=$(grep -c . "$LIST" || true)
RANGE=${2:-1-$N}
FIRST=${RANGE%-*}
LAST=${RANGE#*-}
MEM=${MEM_GB:-8}
HOURS=${WALL_H:-6}
mkdir -p "$WORK/logs/batch_$BATCH"
count=0
for i in $(seq "$FIRST" "$LAST"); do
    cfg=$(sed -n "${i}p" "$LIST")
    [ -n "$cfg" ] || { echo "line $i of $LIST is empty" >&2; exit 1; }
    name=$(basename "$cfg" .json)
    echo "$name  <-  $cfg"
    bsub -J "isa_gen_$name" -q hpc -n 1 -R "rusage[mem=${MEM}GB]" -M ${MEM}GB -W ${HOURS}:00 \
         -o "$WORK/logs/batch_$BATCH/gen_$name.out" -e "$WORK/logs/batch_$BATCH/gen_$name.err" \
         "bash '$HERE/generate_one.sh' '$cfg'"
    count=$((count + 1))
done
echo "submitted $count jobs of batch $BATCH: 1 core and ${MEM} GB each, ${HOURS} h limit"
