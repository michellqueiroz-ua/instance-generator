#!/bin/bash
# Run rows FIRST..LAST of jobs.csv one after the other on this core (called by submit.sh inside a job).
#   SOLVER=<dir with a.out> bash run_rows.sh <first> <last>
# Per run: logs/<name>.out (solver output), logs/<name>.err (errors and GNU time: "Maximum resident set size").
# This file is copied to <work>/solver/batch_XX/run_rows.sh by make_solver_jobs.py, which fills the two __ values.
TRAVEL_TIME='__TRAVEL_TIME__'
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE=$(pwd)
FIRST=$1
LAST=$2
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v"
n=0
tail -n +2 jobs.csv | while IFS=';' read -r key instance algo seed total n1 n2 depot; do
    n=$((n+1)); [ "$n" -lt "$FIRST" ] && continue; [ "$n" -gt "$LAST" ] && break
    key=$(echo $key); algo=$(echo $algo); seed=$(echo $seed); total=$(echo $total)
    n1=$(echo $n1); n2=$(echo $n2); depot=$(echo $depot); instance=$(echo $instance)
    name="${key}_${algo}_s${seed}"
    [ -s "results/$name.txt" ] && continue
    greedy=""; [ "$algo" = "GR" ] && greedy="--greedy_only"
    OMP_NUM_THREADS=1 $TIMEV "$SOLVER/a.out" --filename_requests 3 "$instance" --seed $seed \
        --filename_travel_time "$TRAVEL_TIME" --output_file "$HERE/results/$name.txt" \
        --number_depots 1 --depot $depot --type_vehicles 2 --capacity_vehicles 4 8 __SA_ARGS__ \
        --number_vehicles1 $n1 --number_vehicles2 $n2 --number_vehicles3 0 --total_requests $total $greedy \
        > "logs/$name.out" 2> "logs/$name.err"
    echo "$name exit $?"
done
