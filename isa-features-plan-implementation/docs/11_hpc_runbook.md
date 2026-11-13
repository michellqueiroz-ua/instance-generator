# 11. HPC runbook (DTU HPC, LSF) for the ISA study

Commands as they were run for batch 2 (2026-11). Replace the batch number. One command per block; the user pastes the output and the
assistant checks it step by step. Never run solver or generation jobs on the Windows PC.

## Environment

- Login node `hpclogin1` (prompt `hpclogin1(micqu)`), queue `hpc`, LSF (`bsub`, `bjobs`). Work directory `/work3/micqu/isa` (set `ISA_WORK` to change it).
- The repository clone is `~/instance-generator`, branch `isa-features`. Local flow (Windows PowerShell): commit on `isa-merge-041`, then
  `git push origin isa-merge-041:isa-features`, then on the HPC `git -C ~/instance-generator pull`.
- Python: the login node default is Python 3.9 **without pandas**. Use the venv: `module load python3/3.11.13; source ~/venvs/reqreate/bin/activate`
  (jobs do the same inside the `bsub` command). `pyarrow` is **not** installed yet (needed only for the GTFS features: `pip install pyarrow`).
- Solver: built from `ODBRP_P/MDHODBRP-P/odbrp-parallel.cpp` on the HPC (source uploaded by sftp, CRLF stripped with `sed -i 's/\r$//'`, then `make` gives `a.out`; the
  build used for batch 2 is the one without the +3600 s slack, D28). The submit scripts need `SOLVER=<folder with a.out>`; ask the user which folder (the scaling test used it).
- File transfer: `transfer.gbar.dtu.dk` accepts **sftp only** (interactive, password); for `scp`/`ssh`/`rsync` use `login2.hpc.dtu.dk`.
- Memory on LSF is **per slot**; LSF "Max Memory" is unreliable for short jobs (use GNU `/usr/bin/time -v`, "Maximum resident set size"). Admins kill jobs that over-reserve.

## Folder layout under /work3/micqu/isa

`configs/batch_NN/` (generator configs), `design/batch_NN.csv` and `design/batch_NN_fleets.json`, `Capital Region of Denmark/` (network, `csv_format/<instance>_1.csv` solver instances,
`json_format/`, `travel_time_matrix/station_travel_time_matrix.{csv,npy}`, `csv/...zone_weights.csv`), `logs/`, `results/` (features, perf matrices, instance space, maps),
`solver/batch_NN/` (`jobs.csv`, `submit.sh`, `run_rows.sh`, `results/`, `logs/`), `gtfs/` (PT features: scripts, `2026-11-10/*.parquet`, `features/`, `logs/`).

## Steps and measured resources (batch 2, 5000 requests per instance)

1. **Network and station matrix** (done once): `hpc/build_network.sh`, `hpc/station_matrix.sh` (97 s). **Zone weights** (done once): `bash .../hpc/zone_weights.sh`
   (1 core, 1 GB, 90 min wall, 38 s CPU: waits for Overpass), then `python .../isa_set/zone_weights.py --work /work3/micqu/isa --central-only`.
   Check the demand design before generating: `python .../isa_set/check_weights.py --work /work3/micqu/isa --decay-km 3`.
2. **Draw and generate a batch.** Smoke instances: `make_batch.py --batch 90 --n 1 --set inbound_share=1 --set concentration=1.5`.
   ```bash
   python ~/instance-generator/isa-features-plan-implementation/isa_set/make_batch.py --batch 2 --n 100
   ```
   ```bash
   MEM_GB=4 WALL_H=3 bash ~/instance-generator/isa-features-plan-implementation/isa_set/hpc/generate_batch.sh 2
   ```
   Resources: 100 jobs, 1 core, 4 GB reserved, run time 18-35 min (up to 50 min for 10000 requests), CPU 98-100%, peak memory 2.87 GB; about 40 core-hours.
   Checks: `ls "/work3/micqu/isa/Capital Region of Denmark/csv_format/" | grep -c ISA_b02` (100); `grep -L "Successfully completed" /work3/micqu/isa/logs/batch_02/gen_*.out` (none);
   `grep -c "failed after" /work3/micqu/isa/logs/batch_02/gen_*.out | grep -v ":0$"` (none); incomplete requests: `hpc/regenerate_incomplete.sh`.
   Summary of all instances: `python .../isa_set/plot_requests.py --batch 2 --summary` (maps: `plot_requests.py ISA_b02_i001`, PNG in `results/maps/`).
3. **Features** (1 core, 2 GB reserved, about 100 s per instance, peak 1.4-1.6 GB, 98-99% CPU; 10 jobs of 10 instances, each writes its own part file):
   ```bash
   for s in $(seq 1 10 91); do bsub -J isa_feat_$s -q hpc -n 1 -R "rusage[mem=2GB]" -M 2GB -W 1:00 -o /work3/micqu/isa/logs/features_batch_02_$s.out bash ~/instance-generator/isa-features-plan-implementation/isa_set/hpc/features_batch.sh 2 --first $s --last $((s+9)) --out /work3/micqu/isa/results/features_batch_02_part$(printf %02d $(( (s-1)/10+1 ))).csv; done
   ```
   ```bash
   python ~/instance-generator/isa-features-plan-implementation/isa_set/merge_features.py --batch 2
   ```
4. **Solver rows** (SA seeds 1-3 and GR per fleet: 1200 rows; one instance = 12 rows = 1 job; 1 core, 2 GB, run time per row 35-870 s, about 50 min per job, peak 1.1-1.6 GB, about 67 core-hours):
   ```bash
   python ~/instance-generator/isa-features-plan-implementation/isa_set/make_solver_jobs.py --batch 2
   ```
   ```bash
   SOLVER=<folder with a.out> PER_JOB=12 MEM_GB=2 WALL_H=3 bash /work3/micqu/isa/solver/batch_02/submit.sh
   ```
   SA variants (rows appended after the standard 1200; cap in moves per call): `make_solver_jobs.py --batch N --sa-variants 1000,2000,10000 --variants-max-fleet-percent 25`, then
   `SOLVER=... PER_JOB=27 WALL_H=3 bash .../submit.sh 1201-1227` (see `solver_submit.sh` header). Timing probe: `... --batch 91 --sa-seeds 1 --sa-variants 1000 --variants-max-fleet-percent 25`, `PER_JOB=1 MEM_GB=2 WALL_H=6 bash .../batch_91/submit.sh 7-8`.
   Checks: results count (`ls .../results | wc -l`), **empty files** (`find .../results -size 0 | wc -l`, the solver can report success with an empty file), failed runs (`cat .../logs/rows_*.out | grep -c "exit [1-9]"`),
   peak memory (`grep -h "Maximum resident" .../logs/*.err | awk '{print $NF}' | sort -n | tail -n 1`), CPU and run time of the jobs (sum the "CPU time" and "Run time" lines of `logs/rows_*.out`).
   Limit: the solver caps passengers at 6000 (10000-request instances give empty results).
5. **Collect and instance space** (venv active):
   ```bash
   python ~/instance-generator/isa-features-plan-implementation/isa_set/collect_batch.py --batch 2
   ```
   (`--partial` when SA variants exist only for some keys.) It writes `results/perf_matrix_{mean,best}_batch_NN.csv` (total), `perf_matrix_{served,quality}_{mean,best}_...`, `solver_runs_batch_NN.csv`.
   ```bash
   for m in mean served_mean quality_mean; do echo "== $m"; ISA_ALGORITHMS=SA,GR python ~/instance-generator/src/isa_pipeline.py --features /work3/micqu/isa/results/features_batch_02.csv --perf /work3/micqu/isa/results/perf_matrix_${m}_batch_02.csv --out /work3/micqu/isa/results/instance_space_${m}_batch_02.csv --footprint-summary /work3/micqu/isa/results/footprint_summary_${m}_batch_02.csv --figures /work3/micqu/isa/results/figures_${m}_batch_02 --method pilot --footprint-threshold 0.95 | grep -E "variance|good "; done
   ```
   Third-party share by fleet band:
   ```bash
   python - <<'EOF'
   import json, pandas as pd
   r = pd.read_csv('/work3/micqu/isa/results/solver_runs_batch_02.csv')
   fl = json.load(open('/work3/micqu/isa/design/batch_02_fleets.json'))
   r['pct'] = r['key'].map(lambda k: fl[k]['fleet_percent_of_requests'])
   r['band'] = pd.cut(r['pct'], [10, 15, 20, 25, 30, 40], include_lowest=True)
   r['tp'] = r['third_party'] / (r['served'] + r['third_party']) * 100
   print(r.groupby(['band', 'algo'], observed=True)['tp'].agg(['mean', 'count']).round(1).unstack().to_string())
   EOF
   ```
6. **PT (GTFS) features** (scripts are uncommitted scratch files in the repo root; upload with `scp` to `login2.hpc.dtu.dk`, not GitHub): files go to `/work3/micqu/isa/gtfs/` (scripts) and `/work3/micqu/isa/gtfs/2026-11-10/` (`stop_events`, `stop_supply`, `route_patterns`, `route_summary` parquet).
   10 jobs of 10 instances, 1 core, 1 GB, 2 h (estimate 40-55 min each, unmeasured on real instances):
   ```bash
   for j in $(seq 0 9); do f=$((10*j+1)); l=$((10*j+10)); bsub -J ptf_$j -q hpc -n 1 -R "rusage[mem=1GB]" -M 1GB -W 2:00 -o /work3/micqu/isa/gtfs/logs/ptf_$j.out "module load python3/3.11.13; source $HOME/venvs/reqreate/bin/activate; cd /work3/micqu/isa/gtfs; python scratch_gtfs_features.py --batch 2 --first $f --last $l --gtfs-dir /work3/micqu/isa/gtfs/2026-11-10 --out features/pt_features_b02_$(printf %03d $f)_$(printf %03d $l).csv"; done
   ```
   ```bash
   cd /work3/micqu/isa/gtfs && python scratch_gtfs_features.py --merge "features/pt_features_b02_*.csv" --out features/pt_features_b02.csv
   ```
7. **Bring files to the PC** (only through `login2`/sftp): `scp "micqu@login2.hpc.dtu.dk:/work3/micqu/isa/results/maps/*.png" isa-features-plan-implementation\maps\`.

## Habits that saved time

- Reminders: ask for the expected wall time and let the assistant schedule a reminder with the exact check commands (they run only while the session is open).
- Before a long job: run 3 instances or one job first and read CPU time, run time and max memory; then size the full set (cores busy, modest memory headroom).
- `grep -L "Successfully completed" logs/*.out` finds failed jobs; the LSF summary at the end of a `.out` log has the resource lines; the error is above it (`head`, not `tail`).
- A venv that is not activated on the login node gives `No module named 'pandas'`.
