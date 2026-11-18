#!/bin/bash
# Submit the rows of jobs.csv as single-core LSF jobs, PER_JOB consecutive rows per job (run one after the other
# by run_rows.sh, so the reserved core stays busy and the queue is not flooded with 10-second jobs). Login node.
#   SOLVER=<dir with the built a.out> bash submit.sh [first-last]     e.g.  bash submit.sh 277-288   (a test)
# MEM_GB (default 1), WALL_H (default 1) and PER_JOB (default 12 = one instance: 3 fleets x (3 SA + 1 GR)) are per job.
# Batches 2-3: 3 fleets per instance, PER_JOB=12 (one instance). Batches >= 4 (design.json draws_per_instance 10): 40 rows per instance, hpc_submit_next_batch.sh passes
# PER_JOB=10 MEM_GB=2 WALL_H=3 (400 jobs for 100 instances; a row takes 202 s on average, 868 s at most). Nothing in this script depends on the number of fleets.
# Measured on the largest instance (995 requests): peak 280 MB, 42 s per run. Finished runs are skipped.
# SA variants (make_solver_jobs.py --sa-variants CAP): their rows come AFTER ALL base rows. Batch 4 (100 instances x 10 fleets x 4 base rows = 4000):
#   variant rows are 4001.., 3 rows (SA seeds 1-3) per fleet and cap, in the order sorted fleet key, cap, seed;
#   --variants-keys KEY keeps only that fleet (probe: rows 4001-4003). Example probe: SOLVER=... PER_JOB=1 WALL_H=6 bash submit.sh 4001-4003
#   (batches 2-3: 12 base rows per instance, variants after row 1200, 27 per instance with caps 1000,2000,10000: PER_JOB=27 WALL_H=3).
# This file is copied to <work>/solver/batch_XX/submit.sh by make_solver_jobs.py.
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE=$(pwd)
[ -x "$SOLVER/a.out" ] || { echo "set SOLVER to the folder with the built a.out"; exit 1; }
mkdir -p results logs
MEM=${MEM_GB:-1}
HOURS=${WALL_H:-1}
PER=${PER_JOB:-12}
N=$(tail -n +2 jobs.csv | grep -c . || true)
RANGE=${1:-1-$N}
FIRST=${RANGE%-*}
LAST=${RANGE#*-}
start=$FIRST
count=0
while [ "$start" -le "$LAST" ]; do
    end=$((start + PER - 1)); [ "$end" -gt "$LAST" ] && end=$LAST
    bsub -J "isa_rows_${start}_${end}" -q hpc -n 1 -R "span[hosts=1]" -R "rusage[mem=${MEM}GB]" -M ${MEM}GB -W ${HOURS}:00 \
         -o "$HERE/logs/rows_${start}_${end}.out" -e "$HERE/logs/rows_${start}_${end}.err" \
         "SOLVER='$SOLVER' bash '$HERE/run_rows.sh' $start $end" > /dev/null
    count=$((count + 1))
    start=$((end + 1))
done
echo "submitted $count jobs: rows $FIRST-$LAST, 1 core and ${MEM} GB each, ${HOURS} h limit"
