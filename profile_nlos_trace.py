#!/usr/bin/env python3
"""
Parse RTKLIB trace logs and summarize NLOS ONNX profiling lines.

Expected lines (trace level 1):
  NLOS-ONNX prof[periodic]: calls=1000 hits=700 misses=300 runs=300 avg_call_us=...
  NLOS-ONNX prof[shutdown]: calls=... hits=... misses=... runs=... avg_call_us=...
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


RE_PROF = re.compile(
    r"NLOS-ONNX prof\[(?P<tag>[^\]]+)\]:\s*"
    r"calls=(?P<calls>\d+)\s+hits=(?P<hits>\d+)\s+misses=(?P<misses>\d+)\s+runs=(?P<runs>\d+)\s+"
    r"avg_call_us=(?P<avg_call_us>-?\d+(?:\.\d+)?)\s+"
    r"avg_hit_us=(?P<avg_hit_us>-?\d+(?:\.\d+)?)\s+"
    r"avg_miss_us=(?P<avg_miss_us>-?\d+(?:\.\d+)?)\s+"
    r"avg_run_us=(?P<avg_run_us>-?\d+(?:\.\d+)?)\s+"
    r"run_share=(?P<run_share>-?\d+(?:\.\d+)?)"
)
RE_TS_FULL = re.compile(r"(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
RE_TS_HMS = re.compile(r"(?P<ts>\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


@dataclass
class ProfileRec:
    line_no: int
    tag: str
    calls: int
    hits: int
    misses: int
    runs: int
    avg_call_us: float
    avg_hit_us: float
    avg_miss_us: float
    avg_run_us: float
    run_share: float
    time_s: Optional[float]


def parse_line_time_seconds(line: str, t0_holder: Dict[str, Optional[datetime]]) -> Optional[float]:
    m = RE_TS_FULL.search(line)
    if m:
        txt = m.group("ts")
        fmt = "%Y/%m/%d %H:%M:%S.%f" if "." in txt else "%Y/%m/%d %H:%M:%S"
        t = datetime.strptime(txt, fmt)
        if t0_holder["t0"] is None:
            t0_holder["t0"] = t
        return (t - t0_holder["t0"]).total_seconds()
    m = RE_TS_HMS.search(line)
    if m:
        txt = m.group("ts")
        fmt = "%H:%M:%S.%f" if "." in txt else "%H:%M:%S"
        t = datetime.strptime(txt, fmt)
        if t0_holder["t0"] is None:
            t0_holder["t0"] = t
        return (t - t0_holder["t0"]).total_seconds()
    return None


def parse_trace(path: Path) -> List[ProfileRec]:
    records: List[ProfileRec] = []
    t0_holder: Dict[str, Optional[datetime]] = {"t0": None}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            m = RE_PROF.search(line)
            if not m:
                continue
            records.append(
                ProfileRec(
                    line_no=line_no,
                    tag=m.group("tag"),
                    calls=int(m.group("calls")),
                    hits=int(m.group("hits")),
                    misses=int(m.group("misses")),
                    runs=int(m.group("runs")),
                    avg_call_us=float(m.group("avg_call_us")),
                    avg_hit_us=float(m.group("avg_hit_us")),
                    avg_miss_us=float(m.group("avg_miss_us")),
                    avg_run_us=float(m.group("avg_run_us")),
                    run_share=float(m.group("run_share")),
                    time_s=parse_line_time_seconds(line, t0_holder),
                )
            )
    return records


def summarize(records: List[ProfileRec]) -> None:
    if not records:
        print("[ERR] Nessuna riga 'NLOS-ONNX prof[...]' trovata nel trace.")
        return

    periodic = [r for r in records if r.tag == "periodic"]
    shutdown = [r for r in records if r.tag == "shutdown"]
    latest = shutdown[-1] if shutdown else records[-1]

    print(f"[OK] Profiling lines trovate: {len(records)}")
    if periodic:
        print(f"     periodic: {len(periodic)}")
    if shutdown:
        print(f"     shutdown: {len(shutdown)}")

    print("\n=== ULTIMA MISURA (piu' rappresentativa) ===")
    print(f"calls={latest.calls} hits={latest.hits} misses={latest.misses} runs={latest.runs}")
    print(f"avg_call_us={latest.avg_call_us:.2f}")
    print(f"avg_hit_us={latest.avg_hit_us:.2f}")
    print(f"avg_miss_us={latest.avg_miss_us:.2f}")
    print(f"avg_run_us={latest.avg_run_us:.2f}")
    print(f"run_share={latest.run_share:.3f}")
    if latest.calls > 0:
        print(f"hit_ratio={latest.hits/latest.calls:.3f} miss_ratio={latest.misses/latest.calls:.3f}")

    if periodic:
        print("\n=== TREND PERIODIC (media/min/max) ===")
        for name in ("avg_call_us", "avg_hit_us", "avg_miss_us", "avg_run_us", "run_share"):
            vals = [getattr(r, name) for r in periodic]
            print(
                f"{name}: mean={mean(vals):.2f} min={min(vals):.2f} max={max(vals):.2f}"
            )


def write_csv(path: Path, records: List[ProfileRec]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "line_no",
                "tag",
                "time_s",
                "calls",
                "hits",
                "misses",
                "runs",
                "avg_call_us",
                "avg_hit_us",
                "avg_miss_us",
                "avg_run_us",
                "run_share",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.line_no,
                    r.tag,
                    "" if r.time_s is None else f"{r.time_s:.3f}",
                    r.calls,
                    r.hits,
                    r.misses,
                    r.runs,
                    f"{r.avg_call_us:.6f}",
                    f"{r.avg_hit_us:.6f}",
                    f"{r.avg_miss_us:.6f}",
                    f"{r.avg_run_us:.6f}",
                    f"{r.run_share:.6f}",
                ]
            )
    print(f"[OK] CSV scritto: {path}")


def plot_periodic(records: List[ProfileRec], out_png: Path) -> None:
    periodic = [r for r in records if r.tag == "periodic"]
    if not periodic:
        print("[WARN] Nessun record periodic: plot non generato.")
        return

    x = [r.time_s if r.time_s is not None else float(i) for i, r in enumerate(periodic)]
    x_label = "time_s" if any(r.time_s is not None for r in periodic) else "index"
    avg_call = [r.avg_call_us for r in periodic]
    avg_run = [r.avg_run_us for r in periodic]
    hit_ratio = [r.hits / r.calls if r.calls > 0 else 0.0 for r in periodic]
    miss_ratio = [r.misses / r.calls if r.calls > 0 else 0.0 for r in periodic]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(x, avg_call, label="avg_call_us", linewidth=1.6)
    ax1.plot(x, avg_run, label="avg_run_us", linewidth=1.6)
    ax1.set_ylabel("microseconds")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    ax1.set_title("NLOS-ONNX profiling trend")

    ax2.plot(x, hit_ratio, label="hit_ratio", linewidth=1.6)
    ax2.plot(x, miss_ratio, label="miss_ratio", linewidth=1.6)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("ratio")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Plot scritto: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NLOS ONNX profiling lines from RTKLIB trace.")
    parser.add_argument("trace_file", type=Path, help="Path del file trace RTKLIB.")
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV opzionale.")
    parser.add_argument("--plot", type=Path, default=None, help="Output PNG opzionale per trend periodic.")
    args = parser.parse_args()

    if not args.trace_file.exists():
        raise SystemExit(f"[ERR] File non trovato: {args.trace_file}")

    records = parse_trace(args.trace_file)
    summarize(records)
    if args.csv:
        write_csv(args.csv, records)
    if args.plot:
        plot_periodic(records, args.plot)


if __name__ == "__main__":
    main()

