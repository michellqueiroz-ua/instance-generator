#!/bin/bash
# Write the station-to-station travel-time matrix of the ISA network once (single core).
#   bsub -J isa_matrix -q hpc -n 1 -R "rusage[mem=4GB]" -M 4GB -W 4:00 -o $WORK/logs/station_matrix.out bash station_matrix.sh
# One shortest-path search per station on the drive graph, chunked, one core. Size memory and wall time
# from the "Max Memory" and "Run time" of the log after the first run.
WORK=${ISA_WORK:-/work3/$USER/isa}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
module load python3/3.11.13
source "$HOME/venvs/reqreate/bin/activate" || exit 4
cd "$WORK" || exit 5
export PYTHONUNBUFFERED=1
python "$HERE/../station_matrix.py" --work "$WORK"
