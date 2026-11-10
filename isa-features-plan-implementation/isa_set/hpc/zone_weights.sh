#!/bin/bash
# Count homes and attractions per zone of the ISA network (one-off, before generating batch 2). Run on a login node:
#   bash ~/instance-generator/isa-features-plan-implementation/isa_set/hpc/zone_weights.sh
#
# One single-core job: 50 OpenStreetMap queries (25 strips x residential and attraction), one strip at a time;
# a restart continues from the strips already cached. Almost all of the wall time is waiting for Overpass, so
# CPU utilisation will be low whatever is reserved; 1 core is the minimum. Memory is a guess (one strip of
# buildings, a few hundred MB, plus Python and osmnx); read "Maximum resident" in the log and tighten it.
# Writes "$ISA_WORK/Capital Region of Denmark/csv/Capital Region of Denmark.zone_weights.csv".
WORK=${ISA_WORK:-/work3/$USER/isa}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK/logs"
bsub -J isa_zone_weights -q hpc -n 1 -R "rusage[mem=4GB]" -M 4GB -W 12:00 \
     -o "$WORK/logs/zone_weights.out" -e "$WORK/logs/zone_weights.err" \
     "module load python3/3.11.13; source $HOME/venvs/reqreate/bin/activate; cd '$WORK'; export PYTHONUNBUFFERED=1; \
      /usr/bin/time -v python '$HERE/../zone_weights.py' --work '$WORK'"
echo "submitted: 1 core, 4 GB, 12 h limit; follow with: tail -f '$WORK/logs/zone_weights.out'"
