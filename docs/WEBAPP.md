# REQreate Web Interface

For installation, startup instructions, output locations, and expected generation
times, see the [web-interface quick start](index.md#quick-start).

## Interface features

- 🎨 **Visual Interface**: User-friendly forms instead of editing JSON files
- 📍 **Easy Configuration**: All parameters in one place with helpful descriptions
- 🔄 **Real-time Progress**: See generation progress as it happens
- 📋 **Job Queue**: Batch network downloads and instance generations, one at a time
- 📊 **Instance Browser**: View and explore existing instances
- 📚 **Built-in Documentation**: Help and guides right in the app

## Usage

### Create New Instance

1. Open **Networks** and download a network by place name or by providing a point and radius. Optionally include public transport fixed lines.
2. Open **Create New Instance** and select a network that has already been built.
3. Set vehicle speed and other instance parameters, then click **Generate Instance**.
4. Open **Queue** to monitor progress; generated files are saved automatically.
5. Download completed results from **View Existing Instances**.

Network downloads and instance generation are separate operations. The instance form only lists networks already present in the working directory.
Networks can also be built from a terminal with `reqreate network "Maastricht, Netherlands"`.

### Networks

- Browse built networks with their last-modified date and folder size.
- Download a network by place name or around a longitude/latitude point and radius.
- Bus stop locations are included with the built network.
- Clicking **Download network** adds a job; follow its progress on **Queue**.

### Queue

**Download network** and **Generate Instance** confirm “Added to the queue
(position N)”. Jobs run automatically, oldest first, with only one running at a
time across browser sessions. Each runs the network or generate CLI in a
separate Python process in the app's working directory; a failed job does not
stop later jobs.

The **Queue** page refreshes every three seconds and shows each job's type,
network/config, status, creation time, duration, and exit code. The running job's
stdout and stderr appear in a live log tail; finished jobs have expandable logs.

- **Cancel** removes a queued job or stops a running job and its child processes
  (including on Windows). Partial output files are left in place.
- **Resume** starts pending work after an app restart. Previously running jobs
  are marked **interrupted**, not retried; queued jobs stay paused until Resume,
  including new submissions made while paused.
- **Clear finished** removes done, failed, cancelled, and interrupted job
  records, logs, and config snapshots. It does not delete generated outputs.

Jobs persist as JSON files in `queue/` under the app's working directory, with
per-job logs in `queue/logs/` and independent generation config snapshots in
`queue/configs/`. Statuses are **queued**, **running**, **done**, **failed**,
**cancelled**, and **interrupted**. Restart the app in the same directory to
recover the queue. One app process owns the queue; do not run multiple app
processes against the same working directory.

### View Existing Instances

- Browse all generated instances
- Preview CSV files
- See file statistics
- Download data

## Tips

- Start with default parameters for your first test
- Reduce number of requests (10-50) for faster generation
- Use "rank_model" distribution for more realistic patterns
- Check logs if generation fails
