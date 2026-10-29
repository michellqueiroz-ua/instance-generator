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
# The build is single-threaded Python (no Ray on the cluster): 1 core. Measured on 2026-10-27
# (4-core run): Max Memory 4.4 GB, 3.3 h wall, of which most was waiting for Overpass retries.
bsub -J isa_network -q hpc -n 1 -R "rusage[mem=6GB]" -M 6GB -W 12:00 \
     -o "$WORK/logs/network.out" -e "$WORK/logs/network.err" \
     "module load python3/3.11.13; source $HOME/venvs/reqreate/bin/activate; cd '$WORK'; \
      python -m REQreate.cli network '$PLACE' --exclude '$EXCLUDE'"
echo "submitted; follow with: tail -f '$WORK/logs/network.out'"
