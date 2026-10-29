# 05. HPC set-up, resources and how to reproduce

## Environment (DTU HPC)

| Item | Value |
|---|---|
| Scheduler | LSF (`bsub`), queue `hpc`, login node `hpclogin1`, file transfer `transfer.gbar.dtu.dk` (sftp only) |
| Work directory | `/work3/<user>/isa` |
| Python | module `python3/3.11.13`, virtualenv `~/venvs/reqreate` (osmnx 2.1.1) |
| Code | `~/instance-generator` (git clone, branch `isa-features`) |
| Solver | built from `ODBRP_P` (SA, GR; Gurobi module `gurobi/11.0.2` for the exact model) |

## Resource rule

Single-threaded phases get one core. LSF memory is **per slot**: total = memory x cores. Size from
the `Max Memory` line of earlier logs, with modest headroom. After each run, report CPU time,
run time and max memory.

| Phase | Cores | Memory | Measured |
|---|---|---|---|
| Network build (once) | 1 | 6 GB | 4.47 GB max (73%), 28% CPU (waits on Overpass), 3.4 h |
| Generation profile, 6 requests | 1 | 8 GB | 1.6 GB (19%), 77% CPU, 250 s |
| Generation of one instance, 218 requests with station matrix | 1 | 8 GB | **OPEN**, see 06 (7.1 GB and 19443 s CPU at 5.3 h, not finished) |
| SA / GR solver job | 1 | 16 GB reserved in the pilot | about 6 GB used. **OPEN**: resize |
| Exact model v2 | threads | not yet sized | |

## Commands (HPC, bash)

Build the network once:

    bash ~/instance-generator/isa-features-plan-implementation/isa_set/hpc/build_network.sh

Draw a batch of 100 instances, then generate (one single-core job per config):

    python ~/instance-generator/isa-features-plan-implementation/isa_set/make_batch.py --batch 1 --n 100
    bash ~/instance-generator/isa-features-plan-implementation/isa_set/hpc/generate_batch.sh 1

`MEM_GB` and `WALL_H` override the 8 GB / 6 h defaults. Logs: `/work3/<user>/isa/logs/batch_XX/gen_<instance>.out`.

Profile a small generation run:

    bsub -J isa_prof -q hpc -n 1 -R "rusage[mem=8GB]" -M 8GB -W 1:00 -o /work3/$USER/isa/logs/profile.out bash ~/instance-generator/isa-features-plan-implementation/isa_set/hpc/profile_one.sh <config.json> 6

## Pipeline for the full set

1. Build the network (done).
2. Write the station travel-time matrix once (**OPEN**, see 06).
3. Draw a batch, generate instances, collect the feature rows.
4. Run SA and GR on each instance and fleet (3 fleets per instance); collect `perf_matrix.csv`.
5. Project into the instance space (`src/isa_pipeline.py`); measure growth of the space.
6. Stop when the share of new instances far from existing ones falls below about 5%, or at the cap.

Step 5 must use the shared station matrix when computing the time-window features: with no
matrix in the instance, `tw_tightness` falls back to each request's direct travel time, which
is a different definition. **OPEN**: the feature step that attaches the shared matrix is not written.
