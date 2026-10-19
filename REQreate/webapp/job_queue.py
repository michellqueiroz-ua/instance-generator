"""Persistent, single-worker job queue, independent of Streamlit."""

import copy
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil


FINISHED = {"done", "failed", "cancelled", "interrupted"}


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    def __init__(self, working_directory):
        self.working_directory = Path(working_directory).resolve()
        self.directory = self.working_directory / "queue"
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "logs").mkdir(exist_ok=True)
        self._condition = threading.Condition()
        self._jobs = {}
        self._process = None
        self._closed = False
        for path in self.directory.glob("*.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            self._jobs[job["id"]] = job
            if job["status"] == "running":
                if job.get("pid") is not None:
                    self._stop_tree(job["pid"], job["process_created"])
                job.update(status="interrupted", finished=_now())
                self._save(job)
        self._paused = any(job["status"] == "queued" for job in self._jobs.values())
        self._worker = threading.Thread(target=self._run, name="reqreate-queue", daemon=True)
        self._worker.start()

    def _save(self, job):
        path = self.directory / f'{job["id"]}.json'
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(path)

    @property
    def paused(self):
        with self._condition:
            return self._paused

    def jobs(self):
        with self._condition:
            return copy.deepcopy(sorted(self._jobs.values(), key=lambda job: job["created"]))

    def _enqueue(self, job_id, job_type, parameters):
        with self._condition:
            if self._closed:
                raise RuntimeError("The job queue is closed.")
            job = {
                "id": job_id,
                "type": job_type,
                "parameters": parameters,
                "status": "queued",
                "created": _now(),
                "started": None,
                "finished": None,
                "log_path": str(self.directory / "logs" / f"{job_id}.log"),
                "exit_code": None,
                "pid": None,
                "process_created": None,
            }
            self._save(job)
            self._jobs[job_id] = job
            position = sum(item["status"] in {"queued", "running"} for item in self._jobs.values())
            self._condition.notify_all()
            return copy.deepcopy(job), position

    def submit_network(self, place, point=None, dist=None, fixed_lines=None):
        return self._enqueue(uuid.uuid4().hex, "network", {
            "place": place, "point": point, "dist": dist, "fixed_lines": fixed_lines,
        })

    def submit_generate(self, config_path, configuration=None):
        config_path = Path(config_path).resolve()
        config = configuration if configuration is not None else json.loads(
            config_path.read_text(encoding="utf-8")
        )
        job_id = uuid.uuid4().hex
        # Keep the basename (used in generated output names), but isolate each submission.
        snapshot = self.directory / "configs" / job_id / config_path.name
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return self._enqueue(job_id, "generate", {
            "config": str(snapshot), "network": config.get("network", ""),
        })

    def _command(self, job):
        command = [sys.executable, "-u", "-m", "REQreate.cli", job["type"]]
        parameters = job["parameters"]
        if job["type"] == "generate":
            return command + [parameters["config"]]
        command.append(parameters["place"])
        if parameters["point"] is not None:
            command.extend(["--point", *(str(value) for value in parameters["point"])])
            command.extend(["--dist", str(parameters["dist"])])
        if parameters["fixed_lines"] is not None:
            command.extend(["--fixed-lines", parameters["fixed_lines"]])
        return command

    def resume(self):
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    @staticmethod
    def _stop_tree(pid, process_created):
        try:
            process = psutil.Process(pid)
            # Persist creation time as well as PID so restart cannot kill a reused PID.
            if process.create_time() != process_created:
                return
            if os.name == "nt":
                # terminate()/kill() alone leave multiprocessing/Ray children alive.
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if result.returncode and process.is_running():
                    raise RuntimeError(f"Could not stop job process tree {pid}.")
            else:
                os.killpg(pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass

    def _stop_process(self):
        if self._process is None or self._process.poll() is not None:
            return
        self._stop_tree(self._process.pid, self._process_created)
        self._process.wait()

    def cancel(self, job_id):
        with self._condition:
            job = self._jobs[job_id]
            if job["status"] not in {"queued", "running"}:
                return
            if job["status"] == "running":
                if self._process.poll() is not None:
                    return
                self._stop_process()
                job["exit_code"] = self._process.returncode
            job.update(status="cancelled", finished=_now())
            self._save(job)
            self._condition.notify_all()

    def clear_finished(self):
        with self._condition:
            for job_id, job in list(self._jobs.items()):
                if job["status"] in FINISHED:
                    (self.directory / f"{job_id}.json").unlink()
                    Path(job["log_path"]).unlink(missing_ok=True)
                    snapshot_directory = self.directory / "configs" / job_id
                    if snapshot_directory.exists():
                        shutil.rmtree(snapshot_directory)
                    del self._jobs[job_id]

    def tail(self, job_id, max_bytes=16384):
        with self._condition:
            path = Path(self._jobs[job_id]["log_path"])
            try:
                with path.open("rb") as log:
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - max_bytes))
                    return log.read().decode("utf-8", errors="replace")
            except FileNotFoundError:
                return ""

    def close(self):
        """Stop the worker without discarding pending work (also used at app exit)."""
        with self._condition:
            self._closed = True
            self._stop_process()
            self._condition.notify_all()
        self._worker.join()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (
                    not self._paused and any(
                        job["status"] == "queued" for job in self._jobs.values()
                    )
                ))
                if self._closed:
                    return
                job = min(
                    (job for job in self._jobs.values() if job["status"] == "queued"),
                    key=lambda job: job["created"],
                )
                job.update(status="running", started=_now())
                self._save(job)
                try:
                    with open(job["log_path"], "ab", buffering=0) as log:
                        options = (
                            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                            if os.name == "nt" else {"start_new_session": True}
                        )
                        self._process = subprocess.Popen(
                            self._command(job), cwd=self.working_directory,
                            stdout=log, stderr=subprocess.STDOUT, **options,
                        )
                        try:
                            self._process_created = psutil.Process(self._process.pid).create_time()
                        except psutil.NoSuchProcess:
                            self._process_created = None
                        job.update(pid=self._process.pid, process_created=self._process_created)
                        self._save(job)
                except Exception as error:
                    with open(job["log_path"], "a", encoding="utf-8") as log:
                        log.write(f"Could not start job: {error}\n")
                    job.update(status="failed", finished=_now())
                    self._save(job)
                    continue
                process = self._process

            exit_code = process.wait()
            with self._condition:
                self._process = None
                if job["status"] == "running":
                    job.update(
                        status="interrupted" if self._closed else (
                            "done" if exit_code == 0 else "failed"
                        ),
                        finished=_now(), exit_code=exit_code,
                    )
                    self._save(job)
