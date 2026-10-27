#!/bin/bash
# Build the ISA network once on the DTU HPC: Capital Region of Denmark without Bornholm.
# Run on a login node:
#   bash ~/instance-generator/isa-features-plan-implementation/isa_set/hpc/build_network.sh
#
# Writes $ISA_WORK/Capital Region of Denmark/ (default ISA_WORK=/work3/$USER/isa), which every
# instance of the ISA set loads from cache. Log: $ISA_WORK/logs/network.out
# Needs the venv from the setup steps (~/venvs/reqreate, python3/3.11.13).
set -e
WORK=${ISA_WORK:-/work3/$USER/isa}
PLACE="Capital Region of Denmark"
EXCLUDE="Bornholm Regional Municipality, Denmark"
mkdir -p "$WORK/logs"
# 4 cores x 16 GB: the walk graph of a region is large; downloads and station paths take hours.
bsub -J isa_network -q hpc -n 4 -R "span[hosts=1]" -R "rusage[mem=16GB]" -M 16GB -W 48:00 \
     -o "$WORK/logs/network.out" -e "$WORK/logs/network.err" \
     "module load python3/3.11.13; source $HOME/venvs/reqreate/bin/activate; cd '$WORK'; \
      python -m REQreate.cli network '$PLACE' --exclude '$EXCLUDE'"
echo "submitted; follow with: tail -f '$WORK/logs/network.out'"
