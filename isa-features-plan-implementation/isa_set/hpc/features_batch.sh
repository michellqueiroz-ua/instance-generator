#!/bin/bash
# Compute the ISA features of one batch (single core, single thread).
#   bsub -J isa_feat -q hpc -n 1 -R "rusage[mem=4GB]" -M 4GB -W 3:00 -o $WORK/logs/features_batch_01.out bash features_batch.sh 1
# Arguments after the batch number are passed on, e.g. "1 --first 1 --last 5" for a test.
# Size memory and wall time from the "peak memory" and "Run time" lines of the first log.
WORK=${ISA_WORK:-/work3/$USER/isa}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
module load python3/3.11.13
source "$HOME/venvs/reqreate/bin/activate" || exit 4
cd "$WORK" || exit 5
export PYTHONUNBUFFERED=1
BATCH=$1
shift
python "$HERE/../features_batch.py" --batch "$BATCH" --work "$WORK" "$@"
