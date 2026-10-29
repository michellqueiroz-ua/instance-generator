#!/bin/bash
# Generate one ISA instance (one REQreate config) on the DTU HPC. Submitted by generate_batch.sh.
#   bash generate_one.sh <config.json>
# Single-threaded: REQreate runs sequentially (no Ray on the cluster), so each job reserves one core.
# Needs the venv ~/venvs/reqreate (python3/3.11.13) and the cached network under $ISA_WORK.
CFG=$1
if [ -z "$CFG" ] || [ ! -s "$CFG" ]; then
    echo "generate_one.sh: config '$CFG' is missing or empty" >&2
    exit 3
fi
WORK=${ISA_WORK:-/work3/$USER/isa}
module load python3/3.11.13
source "$HOME/venvs/reqreate/bin/activate" || exit 4
cd "$WORK" || exit 5
name=$(basename "$CFG" .json)
mkdir -p generation_features
export PYTHONUNBUFFERED=1   # progress lines reach the job log while it runs, not in blocks at the end
echo "generating $name from $CFG"
# The generation hook writes the feature row of the instance; one file per instance.
ISA_FEATURES_CSV="$WORK/generation_features/$name.csv" python -m REQreate.cli generate "$CFG" --skip-existing
