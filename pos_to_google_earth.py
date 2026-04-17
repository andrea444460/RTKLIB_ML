#!/usr/bin/env python3
"""
Convertitore avanzato da file POS (RTKLIB) a GPX/KMZ.
Supporta l'esportazione di tracce colorate in base alla qualità del fix (Q)
oppure al numero di satelliti NLOS rilevati da un CSV esterno.

Esempi di utilizzo:
  python pos_to_gpx.py traccia1.pos --format kmz
  python pos_to_gpx.py traccia1.pos traccia2.pos --format both
  python pos_to_gpx.py traccia1.pos --alt-mode clampToGround
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrackPoint:
    lat: float
    lon: float
    ele: float
    time_utc: Optional[str]
    q: Optional[int] = None  # Qualità del fix (es. 1=fix, 2=float, 4=dgps, 5=single)
    ns: Optional[int] = None # Numero di satelliti
    nlos_count: Optional[int] = None
    csv_total_sat: Optional[int] = None
@dataclass
class NlosCsvInfo:
    counts_by_time: dict[float, tuple[int, int]]
    min_time_s: Optional[float]
    max_time_s: Optional[float]




GPST_UTC_LEAP_DATES = [
    dt.datetime(1981, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1982, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1983, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1985, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1988, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1990, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1991, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1992, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1993, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1994, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1996, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1997, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(1999, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(2006, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(2009, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(2012, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(2015, 7, 1, tzinfo=dt.timezone.utc),
    dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc),
]


def parse_iso_utc(text: Optional[str]) -> Optional[dt.datetime]:
    """Parsa una stringa ISO8601 UTC in datetime timezone-aware."""
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def get_gpst_minus_utc_seconds(utc_dt: dt.datetime) -> int:
    leaps = 0
    for leap_dt in GPST_UTC_LEAP_DATES:
        if utc_dt >= leap_dt:
            leaps += 1
        else:
            break
    return leaps


def detect_pos_time_system(pos_path: Path) -> str:
    """Legge l'header del POS e rileva il sistema temporale riportato."""
    try:
        with pos_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("%"):
                    break
                upper = stripped.upper()
                if " GPST" in upper:
                    return "GPST"
                if " UTC" in upper:
                    return "UTC"
    except OSError:
        pass
    return "UTC"


def parse_time_tokens(date_token: str, time_token: str, time_system: str) -> Optional[str]:
    """Converte i token di data/ora nel formato ISO8601 UTC."""
    date_token = date_token.strip()
    time_token = time_token.strip()
    if not date_token or not time_token:
        return None

    normalized_time = (
        time_token.replace("UTC", "").replace("GPST", "").replace("Z", "").strip()
    )

    candidates = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S")
    for fmt in candidates:
        try:
            parsed = dt.datetime.strptime(f"{date_token} {normalized_time}", fmt)
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
            if time_system.upper() == "GPST":
                parsed = parsed - dt.timedelta(seconds=get_gpst_minus_utc_seconds(parsed))
            return parsed.isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    return None


def parse_pos_line(line: str, time_system: str) -> Optional[TrackPoint]:
    """Analizza una singola riga del file POS."""
    stripped = line.strip()
    if not stripped or stripped.startswith("%") or stripped.startswith("#"):
        return None

    parts = stripped.split()
    if len(parts) < 5:
        return None

    try:
        lat = float(parts[2])
        lon = float(parts[3])
        ele = float(parts[4])
        
        # Estrazione Qualità (Q) e Numero Satelliti (ns) se presenti
        q = int(parts[5]) if len(parts) > 5 else None
        ns = int(parts[6]) if len(parts) > 6 else None
    except ValueError:
        return None

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    time_utc = parse_time_tokens(parts[0], parts[1], time_system=time_system)
    return TrackPoint(lat=lat, lon=lon, ele=ele, time_utc=time_utc, q=q, ns=ns)


def process_pos_file(pos_path: Path) -> list[TrackPoint]:
    """Legge un file POS e restituisce una lista di punti validi."""
    points: list[TrackPoint] = []
    time_system = detect_pos_time_system(pos_path)
    try:
        with pos_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = parse_pos_line(line, time_system=time_system)
                if p is not None:
                    points.append(p)
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere il file {pos_path.name}: {e}")
    print(f"[TIME] {pos_path.name}: time system rilevato = {time_system}")
    return points


def compute_threshold_bins(values: list[int], max_bins: int) -> list[int]:
    """Calcola soglie discrete dalla distribuzione dei conteggi NLOS."""
    uniq = sorted(set(values))
    if not uniq:
        return []
    if len(uniq) <= max_bins:
        return uniq

    bins = []
    for i in range(max_bins):
        idx = round(i * (len(uniq) - 1) / max(1, max_bins - 1))
        bins.append(uniq[idx])
    deduped: list[int] = []
    for v in bins:
        if not deduped or deduped[-1] != v:
            deduped.append(v)
    return deduped


def count_to_palette_index(count: Optional[int], thresholds: list[int]) -> int:
    if count is None or not thresholds:
        return 0
    idx = 0
    for i, thr in enumerate(thresholds):
        if count >= thr:
            idx = i
        else:
            break
    return idx


def parse_kml_color_list(text: str) -> list[str]:
    """Parsa una lista di colori KML AABBGGRR separati da virgola."""
    colors: list[str] = []
    for raw in text.split(","):
        color = raw.strip().lower()
        if len(color) == 8:
            colors.append(color)
    return colors


def parse_track_modes(text: str, n_tracks: int, default_mode: str) -> list[str]:
    """
    Parsea la lista di modalità per traccia.
    Esempio: "solid,nlos_count" -> traccia 1 solid, traccia 2 nlos_count.
    Se la lista è più corta del numero tracce, replica l'ultima modalità valida.
    """
    allowed = {"fix", "nlos_count", "solid"}
    if n_tracks <= 0:
        return []
    if not text.strip():
        return [default_mode] * n_tracks

    raw_modes = [m.strip().lower() for m in text.split(",") if m.strip()]
    modes: list[str] = []
    for m in raw_modes:
        if m in allowed:
            modes.append(m)

    if not modes:
        return [default_mode] * n_tracks
    if len(modes) < n_tracks:
        modes.extend([modes[-1]] * (n_tracks - len(modes)))
    return modes[:n_tracks]


def load_nlos_counts_csv(csv_path: Path, nlos_threshold: float, time_bin_s: float) -> NlosCsvInfo:
    """
    Legge il CSV generato da plot_nlos_trace.py e restituisce:
      tempo_relativo_bin -> (numero satelliti NLOS, numero satelliti totali)
    """
    rows: list[dict[str, str]] = []
    first_time_s: Optional[float] = None
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if first_time_s is None:
                time_s_txt = (row.get("time_s") or "").strip()
                if time_s_txt:
                    try:
                        first_time_s = float(time_s_txt)
                    except ValueError:
                        pass

    total_grouped: dict[float, set[tuple[int, int]]] = {}
    nlos_grouped: dict[float, set[tuple[int, int]]] = {}
    for row in rows:
        source = (row.get("source") or "").strip()
        if source not in {"p", "out", "cache", "gate", "epochsat"}:
            continue

        try:
            sat = int((row.get("sat") or "").strip())
        except ValueError:
            continue
        if source == "epochsat":
            freq = 0
        else:
            try:
                freq = int((row.get("freq") or "").strip())
            except ValueError:
                continue

        t_val: Optional[float] = None
        time_s_txt = (row.get("time_s") or "").strip()
        epoch_txt = (row.get("epoch") or "").strip()
        if time_s_txt:
            try:
                t_raw = float(time_s_txt)
                if first_time_s is not None:
                    t_raw -= first_time_s
                t_val = round(t_raw / time_bin_s) * time_bin_s if time_bin_s > 0 else t_raw
            except ValueError:
                t_val = None
        elif epoch_txt:
            try:
                t_val = float(int(epoch_txt))
            except ValueError:
                t_val = None
        if t_val is None:
            continue
        sat_key = (sat, freq)
        total_grouped.setdefault(t_val, set()).add(sat_key)

        try:
            p_nlos = float((row.get("p_nlos") or "").strip())
        except ValueError:
            continue
        if p_nlos >= nlos_threshold:
            nlos_grouped.setdefault(t_val, set()).add(sat_key)

    counts_by_time = {
        k: (len(nlos_grouped.get(k, set())), len(total_sats))
        for k, total_sats in total_grouped.items()
    }
    keys = sorted(counts_by_time.keys())
    return NlosCsvInfo(
        counts_by_time=counts_by_time,
        min_time_s=keys[0] if keys else None,
        max_time_s=keys[-1] if keys else None,
    )


def attach_nlos_counts(points: list[TrackPoint], nlos_counts: dict[float, tuple[int, int]], time_bin_s: float) -> tuple[int, int, Optional[float], Optional[float]]:
    """Associa a ciascun TrackPoint il conteggio NLOS più vicino nel tempo relativo."""
    if not points or not nlos_counts:
        return 0, len(points), None, None

    parsed_times = [parse_iso_utc(p.time_utc) for p in points]
    valid_times = [t for t in parsed_times if t is not None]
    if not valid_times:
        return 0, len(points), None, None

    t0 = valid_times[0]
    matched = 0
    rel_times: list[float] = []
    for p, t in zip(points, parsed_times):
        if t is None:
            continue
        rel_s = (t - t0).total_seconds()
        rel_times.append(rel_s)
        key = round(rel_s / time_bin_s) * time_bin_s if time_bin_s > 0 else rel_s
        counts = nlos_counts.get(key)
        if counts is None:
            continue
        p.nlos_count, p.csv_total_sat = counts
        matched += 1
    pos_min_time = min(rel_times) if rel_times else None
    pos_max_time = max(rel_times) if rel_times else None
    return matched, len(points), pos_min_time, pos_max_time


def build_gpx_tree(tracks_data: list[tuple[str, list[TrackPoint]]]) -> ET.ElementTree:
    """Costruisce la struttura XML del file GPX base."""
    ET.register_namespace("", "http://www.topografix.com/GPX/1/1")
    gpx = ET.Element("gpx", attrib={"version": "1.1", "creator": "pos_to_gpx_converter"})

    for track_name, points in tracks_data:
        trk = ET.SubElement(gpx, "trk")
        ET.SubElement(trk, "name").text = track_name
        trkseg = ET.SubElement(trk, "trkseg")
        for p in points:
            trkpt = ET.SubElement(trkseg, "trkpt", attrib={"lat": f"{p.lat:.8f}", "lon": f"{p.lon:.8f}"})
            ET.SubElement(trkpt, "ele").text = f"{p.ele:.4f}"
            if p.time_utc:
                ET.SubElement(trkpt, "time").text = p.time_utc
            if p.ns is not None:
                ET.SubElement(trkpt, "sat").text = str(p.ns)
            if p.q is not None:
                q_label = {1: "FIX", 2: "FLOAT", 3: "SBAS", 4: "DGPS", 5: "SINGLE", 6: "PPP"}.get(p.q, str(p.q))
                ET.SubElement(trkpt, "desc").text = f"Soluzione: {q_label} (Q={p.q}) | Satelliti: {p.ns}"

    if hasattr(ET, "indent"):
        ET.indent(gpx, space="  ", level=0)
    return ET.ElementTree(gpx)


def build_kml_tree(
    tracks_data: list[tuple[str, list[TrackPoint]]],
    alt_mode: str = "clampToGround",
    color_mode: str = "fix",
    nlos_bins: int = 6,
    solid_colors: Optional[list[str]] = None,
    track_modes: Optional[list[str]] = None,
) -> ET.ElementTree:
    """Costruisce la struttura KML disegnando segmenti colorati per fix o conteggio NLOS."""
    ET.register_namespace("", "http://www.opengis.net/kml/2.2")
    kml = ET.Element("kml", attrib={"xmlns": "http://www.opengis.net/kml/2.2"})
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = "Tracce POS RTKLIB"

    # Definizione stili KML (AABBGGRR).
    fix_styles = {
        1: ("fix_style", "ff00ff00", "FIX (Q=1)"),
        2: ("float_style", "ff00ffff", "FLOAT (Q=2)"),
        4: ("dgps_style", "ff00aaff", "DGPS (Q=4)"),
        5: ("single_style", "ff0000ff", "SINGLE (Q=5)"),
        99: ("unknown_style", "ffaaaaaa", "Altro"),
    }
    nlos_palette = [
        "ff00ff00",  # verde
        "ff40d9ff",  # giallo chiaro
        "ff00aaff",  # arancione
        "ff0088ff",  # arancione scuro
        "ff0044ff",  # rosso
        "ff8800cc",  # viola/rosso
        "ff666666",  # fallback
    ]
    solid_palette = solid_colors or [
        "ffffaa00",  # azzurro/ciano
        "ff0000ff",  # rosso
        "ff00ff00",  # verde
        "ff00ffff",  # giallo
        "ffff00ff",  # magenta
        "ffff8800",  # blu
    ]

    all_nlos_counts = [
        p.nlos_count for _, pts in tracks_data for p in pts if p.nlos_count is not None
    ]
    nlos_thresholds = compute_threshold_bins(all_nlos_counts, max(2, nlos_bins))
    effective_track_modes = track_modes or [color_mode] * len(tracks_data)
    need_nlos_mode = any(m == "nlos_count" for m in effective_track_modes)
    need_solid_mode = any(m == "solid" for m in effective_track_modes)
    need_fix_mode = any(m == "fix" for m in effective_track_modes)

    styles: dict[object, tuple[str, str, str]] = {}
    if need_nlos_mode:
        if not nlos_thresholds:
            styles["none"] = ("nlos_none_style", "ffaaaaaa", "NLOS count unavailable")
        else:
            for i, thr in enumerate(nlos_thresholds):
                style_id = f"nlos_count_{i}"
                color = nlos_palette[min(i, len(nlos_palette) - 1)]
                if i == len(nlos_thresholds) - 1:
                    name = f"NLOS >= {thr}"
                else:
                    nxt = nlos_thresholds[i + 1]
                    name = f"NLOS {thr}-{max(thr, nxt - 1)}"
                styles[i] = (style_id, color, name)
    if need_solid_mode:
        for i, (track_name, _) in enumerate(tracks_data):
            if effective_track_modes[i] != "solid":
                continue
            style_id = f"solid_track_{i + 1}"
            color = solid_palette[i % len(solid_palette)]
            styles[f"track_{i + 1}"] = (style_id, color, f"Track {track_name}")
    if need_fix_mode:
        for k, v in fix_styles.items():
            styles[k] = v

    for _, (style_id, color, name) in styles.items():
        style = ET.SubElement(document, "Style", attrib={"id": style_id})
        line_style = ET.SubElement(style, "LineStyle")
        ET.SubElement(line_style, "color").text = color
        ET.SubElement(line_style, "width").text = "5"

    def get_style_id(p: TrackPoint, track_idx: int, mode: str) -> str:
        if mode == "nlos_count":
            if not nlos_thresholds or p.nlos_count is None:
                return "nlos_none_style"
            idx = count_to_palette_index(p.nlos_count, nlos_thresholds)
            return f"nlos_count_{idx}"
        if mode == "solid":
            return f"solid_track_{track_idx + 1}"
        if p.q == 1:
            return "fix_style"
        if p.q == 2:
            return "float_style"
        if p.q == 4:
            return "dgps_style"
        if p.q == 5:
            return "single_style"
        return "unknown_style"

    def segment_value(p: TrackPoint, track_idx: int, mode: str) -> object:
        if mode == "nlos_count":
            if p.nlos_count is None or not nlos_thresholds:
                return "none"
            return count_to_palette_index(p.nlos_count, nlos_thresholds)
        if mode == "solid":
            return f"track_{track_idx}"
        return p.q

    for track_idx, (track_name, points) in enumerate(tracks_data):
        track_mode = effective_track_modes[track_idx] if track_idx < len(effective_track_modes) else color_mode
        folder = ET.SubElement(document, "Folder")
        ET.SubElement(folder, "name").text = track_name

        if not points:
            continue

        current_value = segment_value(points[0], track_idx, track_mode)
        segment_points = [points[0]]
        segment_count = 1

        def add_segment(pts: list[TrackPoint], seg_value: object, seg_num: int):
            if len(pts) < 2:
                return

            pm = ET.SubElement(folder, "Placemark")
            if track_mode == "nlos_count":
                sample_count = pts[-1].nlos_count
                label = f"NLOS count {sample_count}" if sample_count is not None else "NLOS unavailable"
                total_sat = pts[-1].csv_total_sat
                los_count: Optional[int] = None
                nlos_pct: Optional[float] = None
                los_pct: Optional[float] = None
                if sample_count is not None and total_sat is not None and total_sat > 0:
                    los_count = max(0, total_sat - sample_count)
                    nlos_pct = 100.0 * sample_count / total_sat
                    los_pct = 100.0 * los_count / total_sat
                ET.SubElement(pm, "name").text = f"Segmento {seg_num} ({label})"
                desc_parts = [f"NLOS detected: {sample_count}", f"Punti nel segmento: {len(pts)}"]
                if total_sat is not None:
                    desc_parts.append(f"Satelliti totali CSV: {total_sat}")
                if los_count is not None and nlos_pct is not None and los_pct is not None:
                    desc_parts.append(f"LOS detected: {los_count}")
                    desc_parts.append(f"NLOS: {nlos_pct:.1f}%")
                    desc_parts.append(f"LOS: {los_pct:.1f}%")
                desc = "<br/>".join(desc_parts)
            else:
                if track_mode == "solid":
                    desc = f"Traccia: {track_name}<br/>Punti nel segmento: {len(pts)}"
                    ET.SubElement(pm, "name").text = f"Segmento {seg_num} ({track_name})"
                else:
                    q_val = pts[-1].q
                    q_label = {1: "FIX", 2: "FLOAT", 3: "SBAS", 4: "DGPS", 5: "SINGLE", 6: "PPP"}.get(q_val, "Sconosciuto") if q_val else "N/A"
                    desc = f"Qualità: {q_label} (Q={q_val})<br/>Punti nel segmento: {len(pts)}"
                    ET.SubElement(pm, "name").text = f"Segmento {seg_num} ({q_label})"
            ET.SubElement(pm, "styleUrl").text = f"#{get_style_id(pts[-1], track_idx, track_mode)}"
            ET.SubElement(pm, "description").text = desc

            linestring = ET.SubElement(pm, "LineString")
            ET.SubElement(linestring, "tessellate").text = "1"
            ET.SubElement(linestring, "altitudeMode").text = alt_mode

            coords = ET.SubElement(linestring, "coordinates")
            coords.text = " ".join(f"{p.lon:.8f},{p.lat:.8f},{p.ele:.4f}" for p in pts)

        for i in range(1, len(points)):
            p = points[i]
            if segment_value(p, track_idx, track_mode) == current_value:
                segment_points.append(p)
            else:
                add_segment(segment_points, current_value, segment_count)
                segment_count += 1
                current_value = segment_value(p, track_idx, track_mode)
                segment_points = [points[i-1], p]

        add_segment(segment_points, current_value, segment_count)

    if hasattr(ET, "indent"):
        ET.indent(kml, space="  ", level=0)
    return ET.ElementTree(kml)


def save_kmz(kml_tree: ET.ElementTree, output_path: Path):
    """Salva la struttura KML comprimendola in formato KMZ."""
    kml_bytes = ET.tostring(kml_tree.getroot(), encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_bytes)


def open_file(filepath: Path):
    """Apre un file con l'applicazione predefinita del sistema operativo."""
    try:
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.call(["open", filepath])
        else:
            subprocess.call(["xdg-open", filepath])
        print(f"[OK] Apertura di {filepath.name} avviata.")
    except Exception as e:
        print(f"[ATTENZIONE] Impossibile aprire automaticamente il file: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Converti POS in GPX/KMZ colorati per fix o conteggio NLOS.")
    parser.add_argument("input_pos", type=Path, nargs="+", help="File .pos")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Percorso base output")
    parser.add_argument(
        "--format",
        choices=["gpx", "kmz", "both"],
        default="kmz",
        help="Scegli il formato (default: kmz).",
    )
    # NUOVO ARGOMENTO PER GESTIRE L'ALTITUDINE
    parser.add_argument(
        "--alt-mode",
        choices=["clampToGround", "absolute", "relativeToGround"],
        default="clampToGround",
        help="Modalità altitudine per il KML. 'clampToGround' (default) incolla la traccia alla mappa.",
    )
    parser.add_argument(
        "--color-by",
        choices=["fix", "nlos_count", "solid"],
        default="fix",
        help="Colora la traccia per qualità fix o per conteggio NLOS (default: fix).",
    )
    parser.add_argument(
        "--track-colors",
        type=str,
        default="ffffaa00,ff0000ff",
        help="Colori fissi KML AABBGGRR separati da virgola, usati con --color-by solid.",
    )
    parser.add_argument(
        "--track-modes",
        type=str,
        default="",
        help="Modalità per singola traccia (fix|nlos_count|solid) separate da virgola, es: solid,nlos_count.",
    )
    parser.add_argument(
        "--nlos-csv",
        type=Path,
        default=None,
        help="CSV prodotto da plot_nlos_trace.py, usato con --color-by nlos_count.",
    )
    parser.add_argument(
        "--nlos-threshold",
        type=float,
        default=0.5,
        help="Soglia P_nlos per classificare un satellite come NLOS nel CSV (default: 0.5).",
    )
    parser.add_argument(
        "--nlos-time-bin",
        type=float,
        default=1.0,
        help="Bin temporale in secondi per agganciare CSV NLOS e POS (default: 1.0).",
    )
    parser.add_argument(
        "--nlos-bins",
        type=int,
        default=6,
        help="Numero massimo di classi colore discrete per il conteggio NLOS (default: 6).",
    )
    args = parser.parse_args()

    input_paths: list[Path] = args.input_pos
    tracks_data: list[tuple[str, list[TrackPoint]]] = []
    total_points = 0

    print("--- Inizio analisi file POS ---")

    for path in input_paths:
        if not path.is_file():
            continue
        points = process_pos_file(path)
        if points:
            tracks_data.append((path.stem, points))
            total_points += len(points)
            print(f"[LETTO] {path.name}: trovati {len(points)} punti validi.")
    track_modes = parse_track_modes(args.track_modes, len(tracks_data), args.color_by)
    if track_modes:
        print(f"[MODE] Modalita per traccia: {', '.join(track_modes)}")

    if "nlos_count" in track_modes:
        if args.nlos_csv is None:
            print("\n[ERRORE] Con modalità nlos_count devi specificare --nlos-csv.")
            return
        if not args.nlos_csv.is_file():
            print(f"\n[ERRORE] CSV NLOS non trovato: {args.nlos_csv}")
            return
        nlos_info = load_nlos_counts_csv(
            args.nlos_csv,
            nlos_threshold=max(0.0, min(1.0, args.nlos_threshold)),
            time_bin_s=max(0.0, args.nlos_time_bin),
        )
        nlos_counts = nlos_info.counts_by_time
        if not nlos_counts:
            print(f"\n[ERRORE] Nessun conteggio NLOS valido trovato nel CSV: {args.nlos_csv}")
            return
        total_matched = 0
        total_track_points = 0
        pos_span_min: Optional[float] = None
        pos_span_max: Optional[float] = None
        for _, points in tracks_data:
            matched, total_points_track, track_min, track_max = attach_nlos_counts(
                points, nlos_counts, time_bin_s=max(0.0, args.nlos_time_bin)
            )
            total_matched += matched
            total_track_points += total_points_track
            if track_min is not None:
                pos_span_min = track_min if pos_span_min is None else min(pos_span_min, track_min)
            if track_max is not None:
                pos_span_max = track_max if pos_span_max is None else max(pos_span_max, track_max)
        print(f"[NLOS] Conteggi caricati da CSV: {len(nlos_counts)} istanti aggregati.")
        if nlos_info.min_time_s is not None and nlos_info.max_time_s is not None:
            print(
                f"[NLOS] Copertura CSV: {nlos_info.min_time_s:.1f}s -> {nlos_info.max_time_s:.1f}s "
                f"(span {nlos_info.max_time_s - nlos_info.min_time_s:.1f}s)"
            )
        if pos_span_min is not None and pos_span_max is not None:
            print(
                f"[NLOS] Copertura POS: {pos_span_min:.1f}s -> {pos_span_max:.1f}s "
                f"(span {pos_span_max - pos_span_min:.1f}s)"
            )
        if total_track_points > 0:
            match_pct = 100.0 * total_matched / total_track_points
            print(
                f"[NLOS] Match temporali POS<-CSV: {total_matched}/{total_track_points} punti "
                f"({match_pct:.1f}%)"
            )
            if match_pct < 95.0:
                print(
                    "[NLOS][WARN] Il CSV copre solo una parte del percorso o non e' ben allineato nel tempo; "
                    "i tratti senza match verranno colorati in grigio."
                )

    if not tracks_data:
        print("\n[ERRORE] Nessun dato estratto.")
        return

    if args.output:
        output_base = args.output.with_suffix("")
    else:
        output_base = input_paths[0].with_suffix("") if len(input_paths) == 1 else input_paths[0].parent / "tracce_combinate"

    output_base.parent.mkdir(parents=True, exist_ok=True)
    print("\n--- Salvataggio in corso ---")

    if args.format in ["kmz", "both"]:
        kmz_path = output_base.with_suffix(".kmz")
        kml_tree = build_kml_tree(
            tracks_data,
            alt_mode=args.alt_mode,
            color_mode=args.color_by,
            nlos_bins=max(2, args.nlos_bins),
            solid_colors=parse_kml_color_list(args.track_colors),
            track_modes=track_modes,
        )
        save_kmz(kml_tree, kmz_path)
        print(f"[OK] Salvato KMZ: {kmz_path.absolute()}")
        open_file(kmz_path)

    if args.format in ["gpx", "both"]:
        gpx_path = output_base.with_suffix(".gpx")
        gpx_tree = build_gpx_tree(tracks_data)
        gpx_tree.write(gpx_path, encoding="utf-8", xml_declaration=True)
        print(f"[OK] Salvato GPX: {gpx_path.absolute()}")


if __name__ == "__main__":
    main()