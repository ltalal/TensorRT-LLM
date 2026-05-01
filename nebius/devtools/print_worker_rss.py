#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Print host RSS memory for running TensorRT-LLM MPI worker processes.

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


WORKER_CMDLINE_TOKEN = "mpi4py.futures.server"
COMPILE_WORKER_CMDLINE_TOKEN = "torch/_inductor/compile_worker"


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    rss_kib: int
    rank: str
    local_rank: str
    cmdline: str


def _read_cmdline(proc_dir: Path) -> str:
    return proc_dir.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()


def _read_ppid(proc_dir: Path) -> int:
    for line in proc_dir.joinpath("status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return -1


def _read_rss_kib(proc_dir: Path) -> int:
    for line in proc_dir.joinpath("status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _read_rank_env(proc_dir: Path) -> tuple[str, str]:
    rank = "?"
    local_rank = "?"
    for entry in proc_dir.joinpath("environ").read_bytes().split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        if key in {b"OMPI_COMM_WORLD_RANK", b"PMIX_RANK", b"PMI_RANK"} and rank == "?":
            rank = value.decode(errors="replace")
        if key in {b"OMPI_COMM_WORLD_LOCAL_RANK", b"LOCAL_RANK"} and local_rank == "?":
            local_rank = value.decode(errors="replace")
    return rank, local_rank


def _iter_processes() -> list[ProcessInfo]:
    processes = []
    self_pid = os.getpid()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue

        pid = int(proc_dir.name)
        if pid == self_pid:
            continue

        try:
            cmdline = _read_cmdline(proc_dir)
            ppid = _read_ppid(proc_dir)
            rss_kib = _read_rss_kib(proc_dir)
            rank, local_rank = _read_rank_env(proc_dir)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue

        processes.append(
            ProcessInfo(
                pid=pid,
                ppid=ppid,
                rss_kib=rss_kib,
                rank=rank,
                local_rank=local_rank,
                cmdline=cmdline,
            )
        )
    return processes


def _kib_to_gib(value: int) -> float:
    return value / 1024 / 1024


def _sort_key(process: ProcessInfo) -> tuple[int, int]:
    try:
        rank = int(process.rank)
    except ValueError:
        rank = 1 << 30
    return rank, process.pid


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print RSS memory for running TensorRT-LLM workers.")
    parser.add_argument(
        "--include-children",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include direct child process RSS in per-worker totals, such as Torch Inductor compile workers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    processes = _iter_processes()
    workers = [process for process in processes if WORKER_CMDLINE_TOKEN in process.cmdline]
    child_rss_by_ppid: dict[int, int] = {}

    if args.include_children:
        for process in processes:
            if COMPILE_WORKER_CMDLINE_TOKEN in process.cmdline:
                child_rss_by_ppid[process.ppid] = child_rss_by_ppid.get(process.ppid, 0) + process.rss_kib

    if not workers:
        print("No running TensorRT-LLM MPI worker processes found.")
        return 1

    print(
        f"{'RANK':>6} {'LOCAL':>6} {'PID':>8} {'PPID':>8} "
        f"{'RSS_GiB':>10} {'CHILD_GiB':>10} {'TOTAL_GiB':>10}"
    )
    print("-" * 74)

    for worker in sorted(workers, key=_sort_key):
        child_rss_kib = child_rss_by_ppid.get(worker.pid, 0)
        total_rss_kib = worker.rss_kib + child_rss_kib
        print(
            f"{worker.rank:>6} {worker.local_rank:>6} {worker.pid:8d} {worker.ppid:8d} "
            f"{_kib_to_gib(worker.rss_kib):10.2f} {_kib_to_gib(child_rss_kib):10.2f} "
            f"{_kib_to_gib(total_rss_kib):10.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
