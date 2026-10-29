#!/bin/bash
# Generate one batch of ISA instances on the DTU HPC: an LSF job array, one config per element.
# Run on a login node after make_batch.py:
#   bash generate_batch.sh <batch> [elements]      e.g.  bash generate_batch.sh 1   or   bash generate_batch.sh 0 1-1
# Each element is single-threaded Python, so it reserves 1 core and MEM_GB (default 8) for that core.
# Size MEM_GB and WALL_H from the "Max Memory" and "Run time" of the test batch logs.
# Output: <work>/Capital Region of Denmark/csv_format/<instance>_1.csv; logs in <work>/logs/batch_XX/
set -e
BATCH=$(printf "%02d" "$1")
WORK=${ISA_WORK:-/work3/$USER/isa}
LIST="$WORK/configs/batch_$BATCH/list.txt"
[ -s "$LIST" ] || { echo "no $LIST: run make_batch.py --batch $1 first"; exit 1; }
N=$(wc -l < "$LIST")
RANGE=${2:-1-$N}
MEM=${MEM_GB:-8}
HOURS=${WALL_H:-6}
mkdir -p "$WORK/logs/batch_$BATCH" "$WORK/generation_features"
# The generation hook writes feature rows; one file per instance so array elements never share a file.
bsub -J "isa_gen_b${BATCH}[$RANGE]" -q hpc -n 1 -R "rusage[mem=${MEM}GB]" -M ${MEM}GB -W ${HOURS}:00 \
     -o "$WORK/logs/batch_$BATCH/gen_%I.out" -e "$WORK/logs/batch_$BATCH/gen_%I.err" \
     "module load python3/3.11.13; source \$HOME/venvs/reqreate/bin/activate; cd '$WORK'; \
      cfg=\$(sed -n \"\${LSB_JOBINDEX}p\" '$LIST'); name=\$(basename \"\$cfg\" .json); \
      ISA_FEATURES_CSV='$WORK/generation_features/'\$name.csv python -m REQreate.cli generate \"\$cfg\" --skip-existing"
echo "submitted batch $BATCH, elements $RANGE of $N: 1 core and ${MEM} GB each, ${HOURS} h limit"
