#!/bin/bash
# Time the phases of one small generation run, to find what makes generating an instance slow.
#   bsub -J isa_prof -q hpc -n 1 -R "rusage[mem=8GB]" -M 8GB -W 1:00 -o $WORK/logs/profile.out bash profile_one.sh <config.json> [requests]
# Takes a real config, cuts it to a few requests and without the station matrix, runs it under cProfile
# and prints the 40 most expensive calls (cumulative time). Single-threaded, one core.
CFG=$1
N=${2:-6}
if [ -z "$CFG" ] || [ ! -s "$CFG" ]; then
    echo "profile_one.sh: config '$CFG' is missing or empty" >&2
    exit 3
fi
WORK=${ISA_WORK:-/work3/$USER/isa}
module load python3/3.11.13
source "$HOME/venvs/reqreate/bin/activate" || exit 4
cd "$WORK" || exit 5
export PYTHONUNBUFFERED=1
mkdir -p configs/profile
SMALL="$WORK/configs/profile/ISA_prof.json"
python - "$CFG" "$N" "$SMALL" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
config["requests"] = int(sys.argv[2])
config.pop("travel_time_matrix", None)
json.dump(config, open(sys.argv[3], "w"), indent=2)
PY
echo "profiling $N requests from $CFG"
date
python -m cProfile -o "$WORK/profile.out" -m REQreate.cli generate "$SMALL"
date
python -c "import pstats; pstats.Stats('$WORK/profile.out').sort_stats('cumulative').print_stats(40)"
