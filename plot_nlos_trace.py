#!/usr/bin/env python3
"""
Parse RTKLIB trace logs and plot NLOS ONNX probabilities.

Supported log lines:
  - NLOS-ONNX in: sat=1 f=1 snr=44.000 rs=7
  - NLOS-ONNX p: sat=1 f=1 type=phase P=0.046596
  - NLOS-ONNX out: sat=1 f=1 P_los=0.953404 P_nlos=0.046596
  - NLOS-ONNX cache: sat=1 f=1 P_nlos=0.046596
  - NLOS-ONNX scale: sat=1 f=1 gain=1.500 scale=1.074
  - NLOS-ONNX gate: sat=1 f=1 P=0.812 thr=0.800
  - NLOS-ONNX epochsat: epoch=123 dir=F time=2026/04/15 12:00:00.000 sat=1 P=0.046596
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt


RE_IN = re.compile(
    r"NLOS-ONNX in:\s*sat=(?P<sat>\d+)\s+f=(?P<freq>\d+)\s+snr=(?P<snr>-?\d+(?:\.\d+)?)\s+rs=(?P<rs>\d+)"
)
RE_P = re.compile(
    r"NLOS-ONNX p:\s*sat=(?P<sat>\d+)\s+f=(?P<freq>\d+)\s+type=(?P<type>\w+)\s+P=(?P<p>-?\d+(?:\.\d+)?)"
)
RE_OUT = re.compile(
    r"NLOS-ONNX out:\s*sat=(?P<sat>\d+)\s+f=(?P<freq>\d+)\s+P_los=(?P<plos>-?\d+(?:\.\d+)?)\s+P_nlos=(?P<pnlos>-?\d+(?:\.\d+)?)"
)
RE_CACHE = re.compile(
    r"NLOS-ONNX cache:\s*sat=(?P<sat>\d+)\s+f=(?P<freq>\d+)\s+P_nlos=(?P<pnlos>-?\d+(?:\.\d+)?)"
)
RE_SCALE = re.compile(
    r"NLOS-ONNX scale:\s*sat=(?P<sat>\d+)\s+f=(?P<freq>\d+)\s+gain=(?P<gain>-?\d+(?:\.\d+)?)\s+scale=(?P<scale>-?\d+(?:\.\d+)?)"
)
RE_GATE = re.compile(
    r"NLOS-ONNX gate:\s*sat=(?P<sat>\d+)\s+f=(?P<freq>\d+)\s+P=(?P<p>-?\d+(?:\.\d+)?)\s+thr=(?P<thr>-?\d+(?:\.\d+)?)"
)
RE_EPOCHSAT = re.compile(
    r"NLOS-ONNX epochsat:\s*epoch=(?P<epoch>\d+)\s+dir=(?P<dir>[FB])\s+time=(?P<timestr>\S+\s+\S+)\s+sat=(?P<sat>\d+)\s+P=(?P<p>-?\d+(?:\.\d+)?)"
)
RE_DDRES = re.compile(r"\bddres\s*:\s*dt=")
RE_TS_FULL = re.compile(
    r"(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)
RE_TS_HMS = re.compile(r"(?P<ts>\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


@dataclass
class Rec:
    line_no: int
    sat: int
    freq: int
    obs_type: str  # phase/code/unknown
    p_nlos: float
    p_los: Optional[float]
    snr: Optional[float]
    receiver_state: Optional[int]
    source: str  # p/out/cache/gate
    scale: Optional[float] = None
    gain: Optional[float] = None
    threshold: Optional[float] = None
    epoch: Optional[int] = None
    time_s: Optional[float] = None
    direction: Optional[str] = None


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


def parse_trace(path: Path) -> List[Rec]:
    latest_in: Dict[Tuple[int, int], Tuple[float, int]] = {}
    records: List[Rec] = []
    current_epoch = -1
    t0_holder: Dict[str, Optional[datetime]] = {"t0": None}

    print(f"[1/4] Parsing trace: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            line_time_s = parse_line_time_seconds(line, t0_holder)

            if i % 200000 == 0:
                print(f"  parsed {i:,} lines, collected {len(records):,} NLOS records...")

            if RE_DDRES.search(line):
                current_epoch += 1

            m_epochsat = RE_EPOCHSAT.search(line)
            if m_epochsat:
                epoch = int(m_epochsat.group("epoch"))
                sat = int(m_epochsat.group("sat"))
                p_nlos = float(m_epochsat.group("p"))
                direction = m_epochsat.group("dir")
                time_txt = m_epochsat.group("timestr")
                line_time_s = parse_line_time_seconds(time_txt, t0_holder)
                records.append(
                    Rec(
                        line_no=i,
                        sat=sat,
                        freq=0,
                        obs_type="epoch",
                        p_nlos=p_nlos,
                        p_los=(1.0 - p_nlos),
                        snr=None,
                        receiver_state=None,
                        source="epochsat",
                        epoch=epoch,
                        time_s=line_time_s,
                        direction=direction,
                    )
                )
                continue

            m_in = RE_IN.search(line)
            if m_in:
                sat = int(m_in.group("sat"))
                freq = int(m_in.group("freq"))
                snr = float(m_in.group("snr"))
                rs = int(m_in.group("rs"))
                latest_in[(sat, freq)] = (snr, rs)
                continue

            m_p = RE_P.search(line)
            if m_p:
                sat = int(m_p.group("sat"))
                freq = int(m_p.group("freq"))
                obs_type = m_p.group("type")
                p_nlos = float(m_p.group("p"))
                snr, rs = latest_in.get((sat, freq), (None, None))
                records.append(
                    Rec(
                        line_no=i,
                        sat=sat,
                        freq=freq,
                        obs_type=obs_type,
                        p_nlos=p_nlos,
                        p_los=(1.0 - p_nlos),
                        snr=snr,
                        receiver_state=rs,
                        source="p",
                        epoch=current_epoch if current_epoch >= 0 else None,
                        time_s=line_time_s,
                    )
                )
                continue

            m_out = RE_OUT.search(line)
            if m_out:
                sat = int(m_out.group("sat"))
                freq = int(m_out.group("freq"))
                p_los = float(m_out.group("plos"))
                p_nlos = float(m_out.group("pnlos"))
                snr, rs = latest_in.get((sat, freq), (None, None))
                records.append(
                    Rec(
                        line_no=i,
                        sat=sat,
                        freq=freq,
                        obs_type="unknown",
                        p_nlos=p_nlos,
                        p_los=p_los,
                        snr=snr,
                        receiver_state=rs,
                        source="out",
                        epoch=current_epoch if current_epoch >= 0 else None,
                        time_s=line_time_s,
                    )
                )
                continue

            m_cache = RE_CACHE.search(line)
            if m_cache:
                sat = int(m_cache.group("sat"))
                freq = int(m_cache.group("freq"))
                p_nlos = float(m_cache.group("pnlos"))
                snr, rs = latest_in.get((sat, freq), (None, None))
                records.append(
                    Rec(
                        line_no=i,
                        sat=sat,
                        freq=freq,
                        obs_type="unknown",
                        p_nlos=p_nlos,
                        p_los=(1.0 - p_nlos),
                        snr=snr,
                        receiver_state=rs,
                        source="cache",
                        epoch=current_epoch if current_epoch >= 0 else None,
                        time_s=line_time_s,
                    )
                )
                continue

            m_gate = RE_GATE.search(line)
            if m_gate:
                sat = int(m_gate.group("sat"))
                freq = int(m_gate.group("freq"))
                p_nlos = float(m_gate.group("p"))
                thr = float(m_gate.group("thr"))
                snr, rs = latest_in.get((sat, freq), (None, None))
                records.append(
                    Rec(
                        line_no=i,
                        sat=sat,
                        freq=freq,
                        obs_type="phase",
                        p_nlos=p_nlos,
                        p_los=(1.0 - p_nlos),
                        snr=snr,
                        receiver_state=rs,
                        source="gate",
                        threshold=thr,
                        epoch=current_epoch if current_epoch >= 0 else None,
                        time_s=line_time_s,
                    )
                )
                continue

            m_scale = RE_SCALE.search(line)
            if m_scale:
                sat = int(m_scale.group("sat"))
                freq = int(m_scale.group("freq"))
                gain = float(m_scale.group("gain"))
                scale = float(m_scale.group("scale"))
                # attach scale/gain to the latest record for same sat/freq if present
                for rec in reversed(records):
                    if rec.sat == sat and rec.freq == freq:
                        rec.gain = gain
                        rec.scale = scale
                        break

    print(f"  done: parsed {i:,} lines, collected {len(records):,} NLOS records.")
    return records


def filter_records(
    records: List[Rec],
    sat: Optional[int],
    freq: Optional[int],
    obs_type: str,
    source: str,
) -> List[Rec]:
    out: List[Rec] = []
    for r in records:
        if sat is not None and r.sat != sat:
            continue
        if freq is not None and r.freq != freq:
            continue
        if obs_type != "all" and r.obs_type != obs_type:
            continue
        if source != "all" and r.source != source:
            continue
        out.append(r)
    return out


def write_csv(records: List[Rec], out_csv: Path) -> None:
    print(f"[2/4] Writing CSV: {out_csv}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "line_no",
                "epoch",
                "time_s",
                "sat",
                "freq",
                "type",
                "source",
                "direction",
                "snr_dbhz",
                "receiver_state",
                "p_nlos",
                "p_los",
                "scale",
                "gain",
                "threshold",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.line_no,
                    r.epoch if r.epoch is not None else "",
                    r.time_s if r.time_s is not None else "",
                    r.sat,
                    r.freq,
                    r.obs_type,
                    r.source,
                    r.direction if r.direction is not None else "",
                    r.snr if r.snr is not None else "",
                    r.receiver_state if r.receiver_state is not None else "",
                    r.p_nlos,
                    r.p_los if r.p_los is not None else "",
                    r.scale if r.scale is not None else "",
                    r.gain if r.gain is not None else "",
                    r.threshold if r.threshold is not None else "",
                ]
            )
    print(f"  CSV rows written: {len(records):,}")


def build_x(records: List[Rec], x_mode: str) -> Tuple[List[float], str]:
    if x_mode == "line":
        return [float(r.line_no) for r in records], "Trace line"
    if x_mode == "epoch":
        xs = [float(r.epoch) if r.epoch is not None else float(k) for k, r in enumerate(records)]
        return xs, "Epoch index"
    if x_mode == "time":
        have_time = any(r.time_s is not None for r in records)
        if have_time:
            xs = [r.time_s if r.time_s is not None else float(k) for k, r in enumerate(records)]
            return xs, "Time (s)"
        xs = [float(r.epoch) if r.epoch is not None else float(k) for k, r in enumerate(records)]
        return xs, "Epoch index"
    return list(range(len(records))), "Sample index"


def plot_records(records: List[Rec], title: str, out_png: Path, x_mode: str) -> None:
    x, x_label = build_x(records, x_mode)
    p = [r.p_nlos for r in records]
    snr = [r.snr for r in records]
    rs = [r.receiver_state for r in records]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(x, p, color="tab:red", linewidth=1.4, label="P(NLOS)")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("P(NLOS)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    x_snr = [i for i, v in enumerate(snr) if v is not None]
    y_snr = [snr[i] for i in x_snr]
    axes[1].plot(x_snr, y_snr, color="tab:blue", linewidth=1.0, label="SNR (dBHz)")
    axes[1].set_ylabel("SNR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    x_rs = [i for i, v in enumerate(rs) if v is not None]
    y_rs = [rs[i] for i in x_rs]
    axes[2].step(x_rs, y_rs, where="mid", color="tab:green", linewidth=1.0, label="Receiver State")
    axes[2].set_ylabel("State")
    axes[2].set_yticks(range(0, 8))
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    scales = [r.scale for r in records]
    gains = [r.gain for r in records]
    x_sc = [i for i, v in enumerate(scales) if v is not None]
    y_sc = [scales[i] for i in x_sc]
    x_gn = [i for i, v in enumerate(gains) if v is not None]
    y_gn = [gains[i] for i in x_gn]
    if x_sc:
        axes[3].plot(x_sc, y_sc, color="tab:purple", linewidth=1.0, label="Variance scale")
    if x_gn:
        axes[3].plot(x_gn, y_gn, color="tab:orange", linewidth=1.0, linestyle="--", label="Gain")
    axes[3].set_ylabel("Scale/Gain")
    axes[3].set_xlabel(x_label)
    axes[3].grid(True, alpha=0.3)
    if x_sc or x_gn:
        axes[3].legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def group_key(rec: Rec, x_mode: str, time_bin_s: float) -> float:
    if x_mode == "line":
        return float(rec.line_no)
    if x_mode == "epoch":
        return float(rec.epoch if rec.epoch is not None else rec.line_no)
    if x_mode == "time":
        if rec.time_s is not None:
            if time_bin_s <= 0:
                return float(rec.time_s)
            return float(round(rec.time_s / time_bin_s) * time_bin_s)
        return float(rec.epoch if rec.epoch is not None else rec.line_no)
    return float(rec.line_no)


def plot_total_los_nlos(
    records: List[Rec],
    out_png: Path,
    x_mode: str,
    nlos_threshold: float,
    time_bin_s: float,
) -> None:
    # key -> sat -> max P_nlos seen for that satellite in that bucket
    sat_buckets: dict[tuple[float, str], dict[int, float]] = defaultdict(dict)
    for r in records:
        k = group_key(r, x_mode=x_mode, time_bin_s=time_bin_s)
        dir_key = r.direction if r.direction is not None else "-"
        prev = sat_buckets[(k, dir_key)].get(r.sat)
        if prev is None or r.p_nlos > prev:
            sat_buckets[(k, dir_key)][r.sat] = r.p_nlos

    xs = sorted(sat_buckets.keys())
    los: list[int] = []
    nlos: list[int] = []
    x_vals = [k[0] for k in xs]
    for k in xs:
        sat_probs = sat_buckets[k].values()
        nlos_count = sum(1 for p in sat_probs if p >= nlos_threshold)
        los_count = sum(1 for p in sat_probs if p < nlos_threshold)
        los.append(los_count)
        nlos.append(nlos_count)
    tot = [los[i] + nlos[i] for i in range(len(xs))]
    nlos_ratio = [nlos[i] / tot[i] if tot[i] > 0 else 0.0 for i in range(len(xs))]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].stackplot(x_vals, los, nlos, labels=["LOS count", "NLOS count"], colors=["tab:green", "tab:red"], alpha=0.7)
    axes[0].plot(x_vals, tot, color="tab:blue", linewidth=1.0, label="Total")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"LOS vs NLOS satellites (threshold={nlos_threshold:.3f})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(x_vals, nlos_ratio, color="tab:red", linewidth=1.2, label="NLOS ratio")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("NLOS ratio")
    axes[1].set_xlabel("Time (s)" if x_mode == "time" else ("Epoch index" if x_mode == "epoch" else "Trace line"))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ratio_only(
    records: List[Rec],
    out_png: Path,
    x_mode: str,
    nlos_threshold: float,
    time_bin_s: float,
) -> None:
    sat_buckets: dict[tuple[float, str], dict[int, float]] = defaultdict(dict)
    for r in records:
        k = group_key(r, x_mode=x_mode, time_bin_s=time_bin_s)
        dir_key = r.direction if r.direction is not None else "-"
        prev = sat_buckets[(k, dir_key)].get(r.sat)
        if prev is None or r.p_nlos > prev:
            sat_buckets[(k, dir_key)][r.sat] = r.p_nlos

    xs = sorted(sat_buckets.keys())
    los_ratio: list[float] = []
    nlos_ratio: list[float] = []
    x_vals = [k[0] for k in xs]
    for k in xs:
        vals = list(sat_buckets[k].values())
        total = len(vals)
        if total == 0:
            los_ratio.append(0.0)
            nlos_ratio.append(0.0)
            continue
        nlos_count = sum(1 for p in vals if p >= nlos_threshold)
        nlos_ratio.append(nlos_count / total)
        los_ratio.append(1.0 - nlos_ratio[-1])

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.stackplot(x_vals, los_ratio, nlos_ratio, labels=["LOS %", "NLOS %"], colors=["tab:green", "tab:red"], alpha=0.75)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Time (s)" if x_mode == "time" else ("Epoch index" if x_mode == "epoch" else "Trace line"))
    ax.set_title(f"LOS/NLOS satellite ratio over time (threshold={nlos_threshold:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract and plot NLOS ONNX probabilities from RTKLIB trace.")
    ap.add_argument("trace_file", type=Path, help="Path to RTKLIB trace file")
    ap.add_argument("--sat", type=int, default=None, help="Filter by satellite number")
    ap.add_argument("--freq", type=int, default=None, help="Filter by frequency index (1-based)")
    ap.add_argument("--type", choices=["all", "phase", "code", "unknown"], default="all", help="Filter observation type")
    ap.add_argument(
        "--source",
        choices=["all", "p", "out", "cache", "gate", "epochsat"],
        default="p",
        help="Which log source to plot (default: p)",
    )
    ap.add_argument("--out", type=Path, default=Path("nlos_trace_plot.png"), help="Output PNG path")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    ap.add_argument(
        "--x",
        choices=["sample", "line", "epoch", "time"],
        default="time",
        help="X-axis mode (default: time; falls back to epoch if no timestamps)",
    )
    ap.add_argument(
        "--nlos-threshold",
        type=float,
        default=0.5,
        help="Threshold to classify a sample as NLOS in total plot (default: 0.5)",
    )
    ap.add_argument(
        "--time-bin",
        type=float,
        default=1.0,
        help="Time bin in seconds for total plot when --x time (default: 1.0)",
    )
    ap.add_argument(
        "--no-total-plot",
        action="store_true",
        help="Disable generation of aggregated LOS/NLOS total plot",
    )
    ap.add_argument(
        "--no-detail-plot",
        action="store_true",
        help="Disable detailed per-record plot and only generate CSV/aggregate plots",
    )
    args = ap.parse_args()

    records = parse_trace(args.trace_file)
    if not records:
        raise SystemExit("No NLOS-ONNX records found in trace.")

    freq = args.freq
    effective_source = args.source
    filtered = filter_records(records, sat=args.sat, freq=freq, obs_type=args.type, source=args.source)
    if not filtered and args.source == "p":
        epochsat_filtered = filter_records(records, sat=args.sat, freq=None, obs_type="epoch", source="epochsat")
        if epochsat_filtered:
            filtered = epochsat_filtered
            effective_source = "epochsat"
            print("[2/4] No detailed 'p' records found; falling back to 'epochsat' summary records.")
    if not filtered:
        raise SystemExit("No records after applying filters.")
    print(f"[2/4] Filtered records: {len(filtered):,}")
    aggregate_records = filtered
    if effective_source in {"all", "p"}:
        epochsat_records = [r for r in records if r.source == "epochsat"]
        if args.sat is not None:
            epochsat_records = [r for r in epochsat_records if r.sat == args.sat]
        if epochsat_records:
            aggregate_records = epochsat_records
            print(f"  using epochsat summary records for aggregate plots: {len(aggregate_records):,}")

    if args.csv:
        write_csv(filtered, args.csv)

    title_parts = [f"source={effective_source}", f"type={args.type}", f"n={len(filtered)}"]
    if args.sat is not None:
        title_parts.append(f"sat={args.sat}")
    if args.freq is not None:
        title_parts.append(f"f={args.freq}")
    title = "NLOS Trace Plot | " + ", ".join(title_parts)

    if not args.no_detail_plot:
        print(f"[3/4] Generating detail plot: {args.out}")
        plot_records(filtered, title=title, out_png=args.out, x_mode=args.x)
    total_out = args.out.with_name(f"{args.out.stem}_total{args.out.suffix}")
    ratio_out = args.out.with_name(f"{args.out.stem}_ratio{args.out.suffix}")
    if not args.no_total_plot:
        thr = max(0.0, min(1.0, args.nlos_threshold))
        print(f"[4/4] Generating aggregate plots with threshold={thr:.3f}, x={args.x}, time_bin={max(0.0, args.time_bin):.3f}")
        plot_total_los_nlos(
            aggregate_records,
            out_png=total_out,
            x_mode=args.x,
            nlos_threshold=thr,
            time_bin_s=max(0.0, args.time_bin),
        )
        plot_ratio_only(
            aggregate_records,
            out_png=ratio_out,
            x_mode=args.x,
            nlos_threshold=thr,
            time_bin_s=max(0.0, args.time_bin),
        )

    print(f"Parsed records: {len(records)}")
    print(f"Filtered records: {len(filtered)}")
    if args.csv:
        print(f"CSV saved: {args.csv}")
    if not args.no_detail_plot:
        print(f"Detail plot saved: {args.out}")
    if not args.no_total_plot:
        print(f"Total LOS/NLOS plot saved: {total_out}")
        print(f"LOS/NLOS ratio plot saved: {ratio_out}")


if __name__ == "__main__":
    main()

