#!/usr/bin/env python3
# Usage: python scripts/gradio_app.py (for Colab helper modules).
# Ownership: Copyright (c) 2026 Adrian Meyer
# License: AGPL-3.0 only (code); model weights and dependencies may be under AGPL-3.0.
# v0.2.0 (2026-10-20): detection merging now follows the base pipeline
# (waldstrom/collembot scripts/run_inference.py): IoU-based graph-cut fusion. AI-assisted (Claude).
# v0.1.1 (2026-10-20): removed Single-image tab and ruler auto-detection; added full-resolution
# annotated ZIP export; size classes only with a px/mm scale. Changes AI-assisted (Claude); see CHANGELOG.md.
"""Gradio-ready inference helper for Google Colab.

This module mirrors the essential inference utilities from ``scripts/run_inference.py``
while keeping the surface area small enough for quick interactive use. The
primary entry point is :func:`launch_app`, which spins up a Gradio interface
that lets users upload a slide image, run tiled inference (with both original
and shifted grids), perform graph-cut fusion, and visualise detections together
with the final collembola count.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import warnings
import base64
import csv
import tempfile
import zipfile
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd

import yaml

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid
from shapely.ops import unary_union
import shapely
if int(shapely.__version__.split(".")[0]) < 2:
    raise ImportError(f"CollemBot needs Shapely >= 2.0 (found {shapely.__version__})")
import torch

torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------
TILE_SIZE = 576
SHIFT = 288
PREDICT_BATCH = 8
DEFAULT_DEVICE = "auto"
BEST_CONF = 0.450
BEST_IOU = 0.101
ALPHA = 45.0
CIRCLE_CENTER_MARGIN = 0.20
CIRCLE_RADIUS_MIN_REL = 0.35
CIRCLE_RADIUS_MAX_REL = 0.52
CIRCLE_RADIUS_SCALE = 1.05
ROI_S_MIN = 40
ROI_V_MIN = 60
ROI_V_MAX = 210
ROI_KERNEL_SIZE = 21
CPU_WORKERS = max(1, os.cpu_count() or 1)


def _load_config_overrides(config_path: Path = Path("configs/inference.yaml")) -> None:
    global TILE_SIZE, SHIFT, PREDICT_BATCH, BEST_CONF, BEST_IOU, ALPHA

    if not config_path.is_file():
        return

    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return

    tile_cfg = cfg.get("tile", {})
    TILE_SIZE = int(tile_cfg.get("size", TILE_SIZE))
    SHIFT = int(tile_cfg.get("shift", SHIFT))

    par_cfg = cfg.get("parallel", {})
    PREDICT_BATCH = int(par_cfg.get("batch_size", PREDICT_BATCH))

    fusion_cfg = cfg.get("fusion", {})
    BEST_CONF = float(fusion_cfg.get("best_conf", BEST_CONF))
    BEST_IOU = float(fusion_cfg.get("best_iou", BEST_IOU))
    ALPHA = float(fusion_cfg.get("alpha", ALPHA))


_load_config_overrides()

# ----------------------------------------------------------------------------
# Circle detection helpers
# ----------------------------------------------------------------------------

def _to_bgr_image(img_source):
    import cv2
    if img_source is None:
        return None
    if isinstance(img_source, (str, os.PathLike, Path)):
        return cv2.imread(str(img_source))
    if isinstance(img_source, Image.Image):
        arr = np.array(img_source.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return None


def find_main_circle(img_source):
    import cv2
    img = _to_bgr_image(img_source)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    max_dim = max(h, w)
    TARGET_MAX_DIM = 1600
    scale = max(1.0, max_dim / float(TARGET_MAX_DIM))
    if scale > 1.0:
        new_w = int(round(w / scale))
        new_h = int(round(h / scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    else:
        scale = 1.0
    mindim = min(h, w)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    mask = (s_ch > ROI_S_MIN) & (v_ch > ROI_V_MIN) & (v_ch < ROI_V_MAX)
    mask_u8 = mask.astype("uint8") * 255
    kernel_size = max(3, int(round(ROI_KERNEL_SIZE / scale)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return None
    cx_mask = float(xs.mean())
    cy_mask = float(ys.mean())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    gray_masked = cv2.bitwise_and(gray_blur, gray_blur, mask=mask_u8)
    min_radius = int(mindim * 0.30)
    max_radius = int(mindim * 0.60)
    circles = cv2.HoughCircles(
        gray_masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h / 2,
        param1=120, param2=20, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return None
    candidates = []
    for x_f, y_f, r_f in circles[0]:
        x = int(round(x_f))
        y = int(round(y_f))
        r = int(round(r_f))
        if not (w * CIRCLE_CENTER_MARGIN <= x <= w * (1.0 - CIRCLE_CENTER_MARGIN)):
            continue
        if not (h * CIRCLE_CENTER_MARGIN <= y <= h * (1.0 - CIRCLE_CENTER_MARGIN)):
            continue
        rel_r = r / float(mindim)
        if not (CIRCLE_RADIUS_MIN_REL <= rel_r <= CIRCLE_RADIUS_MAX_REL):
            continue
        dist_norm = math.hypot(x - cx_mask, y - cy_mask) / float(mindim)
        if dist_norm > 0.25:
            continue
        candidates.append((x, y, r, dist_norm))
    if not candidates:
        return None
    x, y, r, _ = min(candidates, key=lambda c: (c[3], -c[2]))
    r = int(r * CIRCLE_RADIUS_SCALE)
    if scale != 1.0:
        x = int(round(x * scale))
        y = int(round(y * scale))
        r = int(round(r * scale))
    return int(x), int(y), int(r)


def point_inside_circle(px: float, py: float, circle):
    if circle is None:
        return True
    cx, cy, r = circle
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def filter_polygons_by_circle(polys: Sequence[dict], circle):
    if circle is None:
        return polys
    filtered = []
    for poly in polys:
        geom = poly.get("polygon")
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        if point_inside_circle(c.x, c.y, circle):
            filtered.append(poly)
    return filtered


# ----------------------------------------------------------------------------
# Geometry utilities (graph-cut fusion)
# ----------------------------------------------------------------------------

def polygon_iou(a: Polygon, b: Polygon) -> float:
    return _polygon_iou_internal(a, b, assume_valid=False)


def _clean_polygon(poly: Polygon):
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.is_empty:
            return None
    return poly


def _polygon_iou_internal(a: Polygon, b: Polygon, *, assume_valid: bool) -> float:
    if a is None or b is None:
        return 0.0
    if not assume_valid:
        a = _clean_polygon(a)
        b = _clean_polygon(b)
        if a is None or b is None:
            return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    if union <= 0 or math.isnan(union):
        return 0.0
    return inter / union


def concave_hull(coords: Sequence[Tuple[float, float]], alpha: float = ALPHA):
    if not coords:
        return None
    mp = MultiPoint(coords)
    if mp.is_empty:
        return None
    try:
        hull = mp.alpha_shape(alpha)
        return hull if hull.area > 0 else mp.convex_hull
    except Exception:
        return mp.convex_hull


def measure_polygon(geom) -> dict:
    """Return area, length, and width of a polygon in pixels.

    Length  = major axis of the minimum rotated bounding rectangle.
    Width   = minor axis of the minimum rotated bounding rectangle.
    Area    = polygon area in pixels².

    Also returns the two endpoints of the major axis so the caller can draw
    a measurement line on the visualisation.
    """
    if geom is None or geom.is_empty:
        return {"area_px": 0.0, "length_px": 0.0, "width_px": 0.0,
                "axis_start": None, "axis_end": None}
    area_px = float(geom.area)
    try:
        mrr    = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)          # 5 pts (closed ring)
        d1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        d2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        if d1 >= d2:
            length_px  = d1
            width_px   = d2
            axis_start = ((coords[0][0] + coords[3][0]) / 2,
                          (coords[0][1] + coords[3][1]) / 2)
            axis_end   = ((coords[1][0] + coords[2][0]) / 2,
                          (coords[1][1] + coords[2][1]) / 2)
        else:
            length_px  = d2
            width_px   = d1
            axis_start = ((coords[0][0] + coords[1][0]) / 2,
                          (coords[0][1] + coords[1][1]) / 2)
            axis_end   = ((coords[3][0] + coords[2][0]) / 2,
                          (coords[3][1] + coords[2][1]) / 2)
    except Exception:
        length_px  = math.sqrt(area_px)
        width_px   = length_px
        axis_start = axis_end = None
    return {
        "area_px":    round(area_px,   1),
        "length_px":  round(length_px, 1),
        "width_px":   round(width_px,  1),
        "axis_start": axis_start,
        "axis_end":   axis_end,
    }


# ── v0.2.0: functions copied verbatim from the base pipeline ──────────────────
# Source: waldstrom/collembot @ a437598, scripts/run_inference.py
# (renamed polygon_iou -> _base_polygon_iou, safe_polygon -> _base_safe_polygon)
INVALID_INTERSECTIONS = 0


def _base_polygon_iou(a, b):
    global INVALID_INTERSECTIONS
    if a is None or b is None:
        return 0
    if not a.is_valid or a.is_empty:
        a = make_valid(a)
        if a.is_empty:
            return 0
    if not b.is_valid or b.is_empty:
        b = make_valid(b)
        if b.is_empty:
            return 0
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="invalid value encountered in intersection"
            )
            inter = a.intersection(b).area
    except RuntimeWarning:
        INVALID_INTERSECTIONS += 1
        return 0
    union = a.area + b.area - inter
    if union <= 0 or math.isnan(union):
        return 0
    return inter / union


def _base_safe_polygon(coords):
    """
    Create a valid Shapely polygon from coords, ignoring degenerate cases.
    """
    # remove consecutive duplicates
    cleaned = []
    for (x,y) in coords:
        if not cleaned or (x,y) != cleaned[-1]:
            cleaned.append((x,y))
    if len(cleaned)<3:
        return None
    # ensure ring closure
    if cleaned[0]!= cleaned[-1]:
        cleaned.append(cleaned[0])
    # drop polygons with non-finite coords
    for x, y in cleaned:
        if not math.isfinite(x) or not math.isfinite(y):
            return None

    try:
        poly = Polygon(cleaned)
        if poly.is_empty:
            return None
        if not poly.is_valid:
            poly = make_valid(poly)
        if isinstance(poly, GeometryCollection):
            # union all pieces that have area
            poly = unary_union([g for g in poly.geoms if g.area>0])
        if poly.is_empty:
            return None
        return poly
    except:
        return None


def _build_adjacency(polys: Sequence[dict], iou_th: float):
    """Links for graph-cut fusion, equal to the base pipeline.

    Base (run_inference.py, fuse_graphcut): two detections are linked when
    their polygon IoU > ``iou_th``, tested for all pairs. Pairs whose polygons
    do not intersect have IoU 0 and can never be linked, so testing only the
    intersecting pairs (STRtree pre-filter, Shapely >= 2) gives the same links,
    much faster for images with hundreds of animals.

    v0.2.0 replaces the v0.1 version, whose IoU step never ran with Shapely 2
    (it relied on ``STRtree.query_bulk``, which Shapely 2 removed) and whose
    extra centroid-distance step is not part of the base pipeline.
    """
    n = len(polys)
    adj = [[] for _ in range(n)]
    geom_by_idx = {}
    for idx, poly in enumerate(polys):
        geom = poly.get("polygon")
        if geom is not None:
            geom_by_idx[idx] = geom
    indices = list(geom_by_idx.keys())
    if len(indices) < 2:
        return adj, geom_by_idx
    valid = [_clean_polygon(geom_by_idx[i]) for i in indices]
    keep = [k for k, g in enumerate(valid) if g is not None]
    if len(keep) < 2:
        return adj, geom_by_idx
    tree = STRtree([valid[k] for k in keep])
    src_idx, dst_idx = tree.query([valid[k] for k in keep], predicate="intersects")
    for s, d in zip(src_idx.tolist(), dst_idx.tolist()):
        if s >= d:
            continue
        a, b = indices[keep[s]], indices[keep[d]]
        if _base_polygon_iou(geom_by_idx[a], geom_by_idx[b]) > iou_th:
            adj[a].append(b)
            adj[b].append(a)
    return adj, geom_by_idx


def fuse_graphcut(polys: Sequence[dict], iou_th: float = BEST_IOU, alpha: float = ALPHA):
    if not polys:
        return []
    n = len(polys)
    adj, geom_by_idx = _build_adjacency(polys, iou_th)
    visited = set()
    out = []
    for i in range(n):
        if i in visited or geom_by_idx.get(i) is None:
            continue
        stack = [i]
        comp = []
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            stack.extend(adj[v])
        coords = []
        confs = []
        for cidx in comp:
            pg = geom_by_idx.get(cidx)
            confs.append(polys[cidx].get("confidence", 0))
            if isinstance(pg, (MultiPolygon, GeometryCollection)):
                coords.extend([tuple(pt) for subg in pg.geoms if subg.geom_type == "Polygon" for pt in subg.exterior.coords])
            else:
                coords.extend(list(pg.exterior.coords))
        hull = concave_hull(coords, alpha=alpha)
        if hull and not hull.is_empty:
            out.append({"polygon": hull, "confidence": max(confs) if confs else 0})
    return out


# ----------------------------------------------------------------------------
# Tiling and model inference
# ----------------------------------------------------------------------------

def _resolve_device(preferred: Optional[str] = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tile_coordinates(width: int, height: int, offset: Tuple[int, int]):
    off_x, off_y = offset
    x_tiles = (width - off_x) // TILE_SIZE
    y_tiles = (height - off_y) // TILE_SIZE
    for xx in range(x_tiles):
        for yy in range(y_tiles):
            x0 = off_x + xx * TILE_SIZE
            y0 = off_y + yy * TILE_SIZE
            yield (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)


def build_tiles(img: Image.Image, *, max_workers: Optional[int] = None):
    width, height = img.size
    tasks = []
    for label, offset in (("ori", (0, 0)), ("shift", (SHIFT, SHIFT))):
        for bbox in _tile_coordinates(width, height, offset):
            tasks.append((label, bbox))
    workers = max(1, max_workers or CPU_WORKERS)

    def _crop(task):
        label, bbox = task
        x0, y0, x1, y1 = bbox
        crop = img.crop((x0, y0, x1, y1)).convert("RGB")
        return {"tile": crop, "offset": (x0, y0), "tag": label}

    if len(tasks) <= 1 or workers == 1:
        return [_crop(t) for t in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_crop, tasks))


def _polygon_from_result(res, idx: int, offset: Tuple[int, int], grid: str):
    ox, oy = offset
    polygons = []
    if res.masks and res.masks.xy is not None:
        coords2d = res.masks.xy[idx]
        pts = [(float(x) + ox, float(y) + oy) for (x, y) in coords2d]
        geom = _base_safe_polygon(pts)   # v0.2.0: same polygon cleaning as the base pipeline
        if geom is not None:
            polygons.append({"polygon": geom, "confidence": float(res.boxes.conf[idx]), "grid": grid})
    else:
        box = res.boxes.xyxy[idx].tolist()
        x0, y0, x1, y1 = box
        pts = [(x0 + ox, y0 + oy), (x1 + ox, y0 + oy), (x1 + ox, y1 + oy), (x0 + ox, y1 + oy)]
        geom = Polygon(pts)
        polygons.append({"polygon": geom, "confidence": float(res.boxes.conf[idx]), "grid": grid})
    return polygons


def run_model_on_tiles(
    model,
    device: str,
    tiles: Sequence[dict],
    *,
    batch_size: int = PREDICT_BATCH,
    use_half: Optional[bool] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    progress_range: Tuple[float, float] = (0.0, 1.0),
):
    detections: List[dict] = []
    half = False
    _clear_cuda_cache()
    total_batches = max(1, math.ceil(len(tiles) / float(batch_size)))
    start_frac, end_frac = progress_range
    for batch_idx, batch_start in enumerate(range(0, len(tiles), batch_size)):
        batch = tiles[batch_start: batch_start + batch_size]
        imgs = [t["tile"] for t in batch]
        offsets = [t["offset"] for t in batch]
        grids = [t.get("tag", "ori") for t in batch]
        if progress:
            frac = start_frac + (batch_idx / total_batches) * (end_frac - start_frac)
            progress(frac, desc=f"Running model batch {batch_idx + 1}/{total_batches}")
        with torch.inference_mode():
            results = model.predict(
                imgs, conf=0.01, iou=0.7, device=device, verbose=False, half=half,
            )
        results = [res.cpu() for res in results]
        for res, offset, grid in zip(results, offsets, grids):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for idx in range(len(res.boxes)):
                detections.extend(_polygon_from_result(res, idx, offset, grid))
    if progress:
        progress(end_frac, desc="Inference complete")
    return detections


def load_model(
    weights_path: str,
    fuse: bool = False,
    *,
    preferred_device: Optional[str] = None,
    use_half: Optional[bool] = None,
):
    import os
    from ultralytics import YOLO
    device = _resolve_device(preferred_device)
    half = False
    _clear_cuda_cache()
    # Tell PyTorch/Ultralytics to trust our own weights without pickle warning.
    # Safe because these are our published model weights (Zenodo 10.5281/zenodo.17987886).
    os.environ.setdefault("TRUST_REMOTE_CODE", "1")
    try:
        import torch
        # For PyTorch ≥ 2.4 which requires weights_only safety checks
        torch.serialization.add_safe_globals([])
    except Exception:
        pass
    model = YOLO(weights_path)
    model.to(device).eval()
    if fuse:
        try:
            model.fuse()
        except Exception:
            pass
    return model, device, half


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------

def iter_subpolygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for sg in geom.geoms:
            if sg.geom_type == "Polygon" and not sg.is_empty:
                out.append(sg)
        return out
    return []


def _build_tile_metadata(tiles: Sequence[dict]):
    meta = {"ori": [], "shift": []}
    for t in tiles:
        tag = t.get("tag", "ori")
        if tag not in meta:
            meta[tag] = []
        ox, oy = t.get("offset", (0, 0))
        meta[tag].append((ox, oy, ox + TILE_SIZE, oy + TILE_SIZE))
    return meta


def _draw_tile_boundaries(draw, tile_meta, grids=None):
    if not tile_meta:
        return
    colors = {"ori": (0, 255, 0, 200), "shift": (0, 128, 255, 200)}
    targets = grids if grids is not None else tile_meta.keys()
    for grid in targets:
        boxes = tile_meta.get(grid, [])
        color = colors.get(grid)
        if not color:
            continue
        for bbox in boxes:
            draw.rectangle(bbox, outline=color, width=2)


def _annotate_polygons(draw, polys, *, font, label_ids: bool, draw_axes: bool = False):
    for idx, poly in enumerate(polys):
        geom = poly.get("polygon")
        conf = poly.get("confidence", 0.0)
        if geom is None or geom.is_empty:
            continue
        for subp in iter_subpolygons(geom):
            coords = list(subp.exterior.coords)
            draw.polygon(coords, outline=(255, 215, 0, 255), fill=(255, 215, 0, 60))
            if label_ids or draw_axes:
                m = poly.get("measurements") or measure_polygon(subp)
                if draw_axes and m["axis_start"] and m["axis_end"]:
                    draw.line([m["axis_start"], m["axis_end"]],
                              fill=(0, 200, 255, 230), width=2)
                if label_ids:
                    cent = subp.centroid
                    length_str = f" L:{m['length_px']:.0f}" if m["length_px"] else ""
                    lbl = f"{idx}:{conf:.2f}{length_str}"
                    draw.text((cent.x, cent.y), lbl, fill=(255, 255, 255, 255), font=font)


def _draw_circle(draw, circle, *, width=3):
    if not circle:
        return
    cx, cy, r = circle
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(255, 255, 0, 255), width=width)


def compose_visualisations(
    base: Image.Image,
    tile_meta: dict,
    raw_polys: Sequence[dict],
    fused_polys: Sequence[dict],
    circle,
):
    font = ImageFont.load_default()
    base_rgb = base.convert("RGB")

    def _panel(polys, *, grids=None, label_ids=False, draw_axes=False):
        canvas = base_rgb.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")
        _draw_circle(draw, circle, width=4 if label_ids else 3)
        _draw_tile_boundaries(draw, tile_meta, grids=grids)
        _annotate_polygons(draw, polys, font=font, label_ids=label_ids, draw_axes=draw_axes)
        return canvas

    raw_ori   = _panel([p for p in raw_polys if p.get("grid") == "ori"],   grids=["ori"],   label_ids=False)
    raw_shift = _panel([p for p in raw_polys if p.get("grid") == "shift"], grids=["shift"], label_ids=False)
    fused     = _panel(fused_polys, grids=None, label_ids=True, draw_axes=True)

    w, h = base_rgb.size
    combined = Image.new("RGB", (w * 2, h * 2), color=(0, 0, 0))
    combined.paste(fused,     (0, 0))
    combined.paste(raw_ori,   (w, 0))
    combined.paste(raw_shift, (0, h))

    # ── info panel with size stats ────────────────────────────────────────────
    lengths = [p["measurements"]["length_px"] for p in fused_polys
               if p.get("measurements") and p["measurements"]["length_px"] > 0]
    areas   = [p["measurements"]["area_px"]   for p in fused_polys
               if p.get("measurements") and p["measurements"]["area_px"]   > 0]

    info = Image.new("RGB", (w, h), color=(20, 20, 20))
    draw_info = ImageDraw.Draw(info)
    summary_lines = [
        "Fused detections",
        f"  count : {len(fused_polys)}",
        "",
        "Grid detections",
        f"  ori   : {len([p for p in raw_polys if p.get('grid') == 'ori'])}",
        f"  shift : {len([p for p in raw_polys if p.get('grid') == 'shift'])}",
        "",
        f"IoU fusion threshold: {BEST_IOU}",
    ]
    if lengths:
        summary_lines += [
            "",
            "── Length (px) ──",
            f"  mean  : {sum(lengths)/len(lengths):.1f}",
            f"  min   : {min(lengths):.1f}",
            f"  max   : {max(lengths):.1f}",
        ]
    if areas:
        summary_lines += [
            "",
            "── Area (px²) ──",
            f"  mean  : {sum(areas)/len(areas):.1f}",
            f"  min   : {min(areas):.1f}",
            f"  max   : {max(areas):.1f}",
        ]
    if circle:
        cx, cy, r = circle
        summary_lines += ["", f"circle: ({cx}, {cy}, r={r})"]
    y = 10
    for line in summary_lines:
        draw_info.text((10, y), line, fill=(255, 255, 255), font=font)
        y += 18
    combined.paste(info, (w, h))
    return combined


def draw_polygons(base: Image.Image, polygons: Sequence[dict], circle, *, label_ids: bool = False):
    font = ImageFont.load_default()
    canvas = base.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    _draw_circle(draw, circle)
    _annotate_polygons(draw, polygons, font=font, label_ids=label_ids)
    return canvas


def _serialise_polygon(geom):
    polys = []
    for subp in iter_subpolygons(geom):
        polys.append([[float(x), float(y)] for (x, y) in subp.exterior.coords])
    return polys


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

# ── ZeroGPU support ───────────────────────────────────────────────────────────
# On HuggingFace ZeroGPU, the @spaces.GPU decorator allocates a shared GPU
# for the duration of the decorated function, then releases it.
# If the spaces package is not available (local run, CPU-only Space), the
# decorator is a no-op so the code works unchanged in both environments.
try:
    import spaces
    _gpu_decorator = spaces.GPU
except Exception:
    def _gpu_decorator(fn):
        return fn


@_gpu_decorator
def infer_image(
    image: Image.Image,
    model,
    device: str,
    *,
    use_half: bool,
    batch_size: int = PREDICT_BATCH,
    conf_threshold: float = BEST_CONF,
    progress: Optional[Callable[[float, str], None]] = None,
):
    if progress:
        progress(0.05, desc="Tiling image")
    tiles = build_tiles(image, max_workers=CPU_WORKERS)
    tile_meta = _build_tile_metadata(tiles)
    raw_polys = run_model_on_tiles(
        model, device, tiles,
        batch_size=batch_size, use_half=use_half,
        progress=progress, progress_range=(0.15, 0.80),
    )
    filtered_raw = [p for p in raw_polys if p.get("confidence", 0.0) >= conf_threshold]
    sub_threshold = [p for p in raw_polys if p.get("confidence", 0.0) < conf_threshold]
    circle_source = getattr(image, "filename", None) or image
    circle = find_main_circle(circle_source)
    filtered_roi = filter_polygons_by_circle(filtered_raw, circle)
    # Also apply circle filter and basic deduplication to candidates
    candidates_roi = filter_polygons_by_circle(sub_threshold, circle)
    if progress:
        progress(0.82, desc="Fusing detections")
    fused = fuse_graphcut(filtered_roi, iou_th=BEST_IOU, alpha=ALPHA)
    # v0.2.0: base pipeline filters by the dish circle before fusion only
    # Attach measurements to every fused detection so visualisation and export
    # can both use them without recomputing.
    for pol in fused:
        pol["measurements"] = measure_polygon(pol.get("polygon"))
    viz = compose_visualisations(image, tile_meta, filtered_roi, fused, circle)
    if progress:
        progress(0.98, desc="Done")
    return fused, viz, circle, filtered_roi, candidates_roi


# ----------------------------------------------------------------------------
# Histogram helpers
# ----------------------------------------------------------------------------

def _lengths_from_fused(fused: Sequence[dict], scale: Optional[float]) -> List[float]:
    """Return per-organism lengths, in mm if scale given, else in pixels."""
    vals = []
    for pol in fused:
        m = pol.get("measurements", {})
        l = m.get("length_px", 0.0)
        if l > 0:
            vals.append(l / scale if scale else l)
    return vals


def _make_single_histogram(lengths: List[float], label: str,
                            unit: str) -> Image.Image:
    """Return a PIL image of a single-sample length histogram with KDE."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    if lengths:
        ax.hist(lengths, bins=30, color="#4CAF50", edgecolor="white",
                alpha=0.75, density=True, label="histogram")
        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(lengths, bw_method="scott")
            xs  = np.linspace(min(lengths), max(lengths), 300)
            ax.plot(xs, kde(xs), color="#1B5E20", linewidth=2, label="KDE")
        except Exception:
            pass
        ax.axvline(sum(lengths) / len(lengths), color="red",
                   linestyle="--", linewidth=1.5,
                   label=f"mean = {sum(lengths)/len(lengths):.1f} {unit}")
    ax.set_xlabel(f"Length ({unit})", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Body length distribution — {label}\n"
                 f"n = {len(lengths)}", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(buf.name)
    plt.close(fig)
    return buf.name


def _make_comparison_plot(groups: List[Tuple[str, List[float]]],
                           unit: str) -> str:
    """Overlaid histogram + KDE for multiple groups. Returns path to PNG."""
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
              "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    # ── left: overlaid KDE ────────────────────────────────────────────────────
    ax_kde = axes[0]
    for i, (name, lengths) in enumerate(groups):
        if not lengths:
            continue
        col = colors[i % len(colors)]
        ax_kde.hist(lengths, bins=25, color=col, alpha=0.25, density=True)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(lengths, bw_method="scott")
            xs  = np.linspace(min(lengths), max(lengths), 300)
            ax_kde.plot(xs, kde(xs), color=col, linewidth=2,
                        label=f"{name}  (n={len(lengths)}, "
                              f"mean={sum(lengths)/len(lengths):.1f})")
        except Exception:
            ax_kde.hist(lengths, bins=25, color=col, alpha=0.5,
                        density=True, label=f"{name} (n={len(lengths)})")
    ax_kde.set_xlabel(f"Length ({unit})", fontsize=11)
    ax_kde.set_ylabel("Density", fontsize=11)
    ax_kde.set_title("Length distributions (overlaid)", fontsize=12)
    ax_kde.legend(fontsize=8)

    # ── right: box plot ───────────────────────────────────────────────────────
    ax_box = axes[1]
    data   = [lengths for _, lengths in groups if lengths]
    labels = [name    for name, lengths in groups if lengths]
    bp = ax_box.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax_box.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax_box.set_ylabel(f"Length ({unit})", fontsize=11)
    ax_box.set_title("Box plot comparison", fontsize=12)

    fig.suptitle("Collembola body-length comparison", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(buf.name)
    plt.close(fig)
    return buf.name


def _make_combined_csv(groups: List[Tuple[str, Sequence[dict]]],
                        scale: Optional[float]) -> str:
    """Write one CSV with all detections, group and image columns included."""
    path = Path(tempfile.mkdtemp()) / "collembot_comparison.csv"
    fieldnames = ["group", "image", "id", "confidence",
                  "area_px2", "length_px", "width_px"]
    if scale:
        fieldnames += ["area_mm2", "length_mm", "width_mm"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for group_name, img_name, fused in groups:
            for idx, pol in enumerate(fused):
                m = pol.get("measurements", {})
                row = {
                    "group":      group_name,
                    "image":      img_name,
                    "id":         idx,
                    "confidence": round(float(pol.get("confidence", 0.0)), 4),
                    "area_px2":   m.get("area_px",   0.0),
                    "length_px":  m.get("length_px", 0.0),
                    "width_px":   m.get("width_px",  0.0),
                }
                if scale:
                    row["area_mm2"]  = round(m.get("area_px",   0.0) / scale**2, 4)
                    row["length_mm"] = round(m.get("length_px", 0.0) / scale,    4)
                    row["width_mm"]  = round(m.get("width_px",  0.0) / scale,    4)
                writer.writerow(row)
    return str(path)


def _make_combined_csv_flagged(
    all_results: List[Tuple[str, str, Sequence[dict], dict]],
    scale: Optional[float],
) -> str:
    """Write the batch comparison CSV with per-image quality flag columns.

    Column layout
    -------------
    Metadata : group, image, needs_review, flag_grid_divergence,
               flag_low_confidence, flag_fusion_ratio, flag_low_count,
               flag_no_circle, flag_grubbs_outlier,
               grid_divergence_pct, mean_confidence, fusion_ratio,
               ori_count, shift_count
    Per-organism : id, confidence, area_px2, length_px, width_px
                   [+ area_mm2, length_mm, width_mm if scale given]

    The flag columns are repeated on every row for the same image so that
    the CSV can be filtered in R or Excel without a join.
    """
    flag_fields = [
        "needs_review",
        "flag_grid_divergence", "flag_low_confidence",
        "flag_fusion_ratio",    "flag_low_count",
        "flag_no_circle",       "flag_grubbs_outlier",
        "grid_divergence_pct",  "mean_confidence",
        "fusion_ratio",         "ori_count", "shift_count",
    ]
    det_fields = ["id", "confidence", "area_px2", "length_px", "width_px"]
    if scale:
        det_fields += ["area_mm2", "length_mm", "width_mm"]

    fieldnames = ["group", "image"] + flag_fields + det_fields

    path = Path(tempfile.mkdtemp()) / "collembot_batch_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for group_name, img_name, fused, flags, _gmm in all_results:
            flag_row = {k: flags.get(k) for k in flag_fields}
            for idx, pol in enumerate(fused):
                m = pol.get("measurements", {})
                row = {
                    "group":      group_name,
                    "image":      img_name,
                    **flag_row,
                    "id":         idx,
                    "confidence": round(float(pol.get("confidence", 0.0)), 4),
                    "area_px2":   m.get("area_px",   0.0),
                    "length_px":  m.get("length_px", 0.0),
                    "width_px":   m.get("width_px",  0.0),
                }
                if scale:
                    row["area_mm2"]  = round(m.get("area_px",   0.0) / scale**2, 4)
                    row["length_mm"] = round(m.get("length_px", 0.0) / scale,    4)
                    row["width_mm"]  = round(m.get("width_px",  0.0) / scale,    4)
                writer.writerow(row)
    return str(path)


# ============================================================================
# Hand correction helpers (Tab 3) — native Gradio, zero JavaScript
# ============================================================================
# Architecture: gr.Image .select() fires a Python callback with pixel (x,y)
# on every click. Python finds the nearest detection, toggles it, redraws
# the PIL image, and updates the DataFrame. No JS, no iframes, no shadow DOM.

CORR_DISPLAY_MAX  = 900   # max px on longest side for correction canvas
CORR_CLICK_RADIUS = 40    # px — max distance from centroid to register a click
MODEL_VERSION_STR = "collembot_yolo11xseg_v_0_1_0"


def _build_correction_state(
    pil_img: Image.Image,
    fused: Sequence[dict],
    candidates: Sequence[dict] = (),
) -> dict:
    """Build correction state from inference output.

    The original model detections (``dets``) are never mutated.
    All corrections live in ``removed`` (set of IDs) and ``added`` (list of points).
    ``candidates`` are sub-threshold YOLO detections — shown as faint outlines
    so the user can promote them by clicking rather than placing blind circles.
    Every action is appended to ``log`` for the audit trail.
    """
    orig_w, orig_h = pil_img.size
    sc = min(1.0, CORR_DISPLAY_MAX / max(orig_w, orig_h))
    dw = max(1, int(orig_w * sc))
    dh = max(1, int(orig_h * sc))
    disp = pil_img.resize((dw, dh), Image.LANCZOS) if sc < 1.0 else pil_img.copy()

    dets = []
    for idx, pol in enumerate(fused):
        geom = pol.get("polygon")
        m    = pol.get("measurements", {})
        cx, cy, poly_pts = 0.0, 0.0, []
        if geom is not None:
            c = geom.centroid
            cx, cy = c.x * sc, c.y * sc
            for subp in iter_subpolygons(geom):
                poly_pts = [[float(x) * sc, float(y) * sc]
                            for x, y in subp.exterior.coords]
                break
        dets.append({
            "id":         idx,
            "confidence": round(float(pol.get("confidence", 0.0)), 4),
            "cx":         round(cx, 1),
            "cy":         round(cy, 1),
            "poly":       poly_pts,
            "length_px":  m.get("length_px", 0.0),
            "area_px":    m.get("area_px",   0.0),
        })

    # Deduplicate candidates by centroid proximity (50px) and build display list
    cand_display = []
    seen_cx: list = []
    for pol in candidates:
        geom = pol.get("polygon")
        if geom is None or geom.is_empty:
            continue
        c  = geom.centroid
        cx = round(c.x * sc, 1)
        cy = round(c.y * sc, 1)
        # Skip if too close to an existing fused detection
        too_close = any(math.hypot(cx - d["cx"], cy - d["cy"]) < 20 for d in dets)
        if too_close:
            continue
        # Skip duplicates within candidates themselves
        dup = any(math.hypot(cx - sx, cy - sy) < 15 for sx, sy in seen_cx)
        if dup:
            continue
        seen_cx.append((cx, cy))
        poly_pts = []
        for subp in iter_subpolygons(geom):
            poly_pts = [[float(x) * sc, float(y) * sc]
                        for x, y in subp.exterior.coords]
            break
        # Compute measurements in original pixel space
        m = measure_polygon(geom)
        cand_display.append({
            "confidence": round(float(pol.get("confidence", 0.0)), 4),
            "cx": cx, "cy": cy, "poly": poly_pts,
            "length_px": m["length_px"],
            "area_px":   m["area_px"],
            "width_px":  m["width_px"],
        })

    return {
        "base_img":      disp,
        "scale":         sc,
        "orig_w":        orig_w,
        "orig_h":        orig_h,
        "dets":          dets,
        "candidates":    cand_display,   # sub-threshold, shown as faint outlines
        "removed":       set(),
        "added":         [],
        "log":           [],
        "model_version": MODEL_VERSION_STR,
        "ts":            datetime.now(timezone.utc).isoformat(),
    }


def _draw_correction_image(state: dict) -> Image.Image:
    """Redraw the correction canvas from current state. Fast PIL-only operation."""
    if not state:
        return None
    canvas = state["base_img"].copy().convert("RGB")
    draw   = ImageDraw.Draw(canvas, "RGBA")
    font   = ImageFont.load_default()
    rem    = state["removed"]

    # ── Sub-threshold candidates — faint grey dashed outlines ─────────────────
    # Shown so the user can see where the model had low-confidence detections.
    # Clicking near one in Add mode promotes it with its actual polygon.
    for cand in state.get("candidates", []):
        pts = cand["poly"]
        cx, cy = cand["cx"], cand["cy"]
        if len(pts) >= 3:
            flat = [tuple(p) for p in pts]
            draw.polygon(flat, fill=(180, 180, 180, 35), outline=(180, 180, 180, 100))
        else:
            r = 8
            draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)],
                         fill=(180, 180, 180, 35), outline=(180, 180, 180, 100))

    # ── Confirmed / removed fused detections ──────────────────────────────────
    for d in state["dets"]:
        did    = d["id"]
        cx, cy = d["cx"], d["cy"]
        pts    = d["poly"]
        is_rem = did in rem

        fill   = (220, 50,  50,  130) if is_rem else (255, 215, 0, 65)
        stroke = (255, 60,  60,  255) if is_rem else (255, 215, 0, 255)

        if len(pts) >= 3:
            flat = [tuple(p) for p in pts]
            draw.polygon(flat, fill=fill, outline=stroke)
        else:
            r = 12
            draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)], fill=fill, outline=stroke)

        if is_rem:
            r = 9
            draw.line([(cx-r, cy-r), (cx+r, cy+r)], fill=(255, 60, 60, 255), width=3)
            draw.line([(cx+r, cy-r), (cx-r, cy+r)], fill=(255, 60, 60, 255), width=3)

        draw.text((cx, cy), str(did), fill=(255, 255, 255, 220), font=font, anchor="mm")

    # ── Manually added / promoted organisms — small green polygons or dots ─────
    for pt in state["added"]:
        poly_pts = pt.get("poly", [])
        px, py   = pt["dx"], pt["dy"]
        if len(poly_pts) >= 3:
            flat = [tuple(p) for p in poly_pts]
            draw.polygon(flat, fill=(0, 210, 80, 100), outline=(0, 230, 80, 255))
            draw.text((px, py), "✓", fill=(160, 255, 160, 230), font=font, anchor="mm")
        else:
            # Fallback: small circle for freehand points
            r = 7
            draw.ellipse([(px-r, py-r), (px+r, py+r)],
                         fill=(0, 210, 80, 140), outline=(0, 220, 80, 255))
            draw.text((px, py), "+", fill=(160, 255, 160, 230), font=font, anchor="mm")

    return canvas


def _build_correction_df(state: dict) -> pd.DataFrame:
    """Return detection list as a pandas DataFrame for gr.DataFrame."""
    if not state:
        return pd.DataFrame(columns=["ID", "Status", "Size class", "Confidence", "Length (px)"])
    rem     = state["removed"]
    lengths = [d["length_px"] for d in state["dets"] if d["length_px"] > 0]
    # Run GMM on current active detections for live classification
    active_lengths = [d["length_px"] for d in state["dets"]
                      if d["id"] not in rem and d["length_px"] > 0]
    gmm = classify_adults_juveniles(active_lengths) if len(active_lengths) >= 6 else {}
    gmm_labels = gmm.get("labels", [])

    rows = []
    label_idx = 0
    for d in state["dets"]:
        lp = d["length_px"]
        if lp > 0 and label_idx < len(gmm_labels) and d["id"] not in rem:
            size_class = gmm_labels[label_idx]
            label_idx += 1
        else:
            size_class = "—"
        rows.append({
            "ID":           d["id"],
            "Status":       "removed" if d["id"] in rem else "active",
            "Size class":   size_class,
            "Confidence":   round(d["confidence"], 3),
            "Length (px)":  round(d["length_px"], 1),
        })
    for i, pt in enumerate(state["added"]):
        source = "promoted" if pt.get("poly") else "manual"
        length = round(pt.get("length_px", 0.0), 1) if pt.get("length_px") else "—"
        rows.append({
            "ID":          f"M{i}",
            "Status":      f"added ({source})",
            "Size class":  "—",
            "Confidence":  pt.get("confidence") or "—",
            "Length (px)": length,
        })
    return pd.DataFrame(rows)


def _correction_tally(state: dict) -> str:
    if not state:
        return ""
    n_model = len(state["dets"])
    n_rem   = len(state["removed"])
    n_add   = len(state["added"])
    final   = n_model - n_rem + n_add
    return (f"**Model: {n_model}** &nbsp;|&nbsp; "
            f"🗑 Removed: {n_rem} &nbsp;|&nbsp; "
            f"➕ Added: {n_add} &nbsp;|&nbsp; "
            f"✅ **Final: {final}**")


def _find_nearest_detection(x: float, y: float, dets: list,
                             removed: set, radius: float = CORR_CLICK_RADIUS) -> Optional[int]:
    """Return the ID of the detection whose centroid is closest to (x,y),
    within ``radius`` pixels. Returns None if nothing is close enough."""
    best_id, best_d = None, radius
    for d in dets:
        dist = math.hypot(d["cx"] - x, d["cy"] - y)
        if dist < best_d:
            best_d, best_id = dist, d["id"]
    return best_id


def _toggle_detection(det_id: int, state: dict) -> dict:
    """Toggle a detection between active and removed, recording to audit log."""
    removed = set(state["removed"])
    log     = list(state["log"])
    ts      = datetime.now(timezone.utc).isoformat()
    if det_id in removed:
        removed.remove(det_id)
        log.append({"ts": ts, "action": "restored",  "det_id": det_id,
                    "orig_x": None, "orig_y": None})
    else:
        removed.add(det_id)
        log.append({"ts": ts, "action": "removed",   "det_id": det_id,
                    "orig_x": None, "orig_y": None})
    return {**state, "removed": removed, "log": log}


def _add_manual_point(dx: float, dy: float, state: dict,
                       candidate: Optional[dict] = None,
                       use_auto_polygon: bool = True) -> dict:
    """Add an organism to the correction state.

    Priority order for polygon source:
    1. Promoted candidate — sub-threshold YOLO polygon (most accurate)
    2. Auto-polygon from local image thresholding — only if use_auto_polygon=True
    3. Fallback: plain circle marker
    """
    sc    = state.get("scale", 1.0)
    log   = list(state["log"])
    ts    = datetime.now(timezone.utc).isoformat()
    added = list(state["added"])
    entry = {
        "dx": dx, "dy": dy,
        "orig_x": round(dx / sc, 1),
        "orig_y": round(dy / sc, 1),
        "ts":     ts,
        "poly":   [],
        "confidence": None,
    }

    if candidate is not None:
        # Use promoted sub-threshold YOLO detection
        entry["poly"]       = candidate.get("poly", [])
        entry["confidence"] = candidate.get("confidence")
        entry["length_px"]  = candidate.get("length_px")
        entry["area_px"]    = candidate.get("area_px")
        entry["width_px"]   = candidate.get("width_px")
    elif use_auto_polygon:
        # Try to auto-fit a polygon from the local image patch
        auto_pts = _auto_polygon_from_click(state["base_img"], dx, dy, sc)
        if auto_pts:
            entry["poly"] = auto_pts
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                geom = ShapelyPolygon(auto_pts)
                if geom.is_valid and not geom.is_empty:
                    m = measure_polygon(geom)
                    entry["length_px"] = round(m["length_px"] / sc, 1)
                    entry["area_px"]   = round(m["area_px"]   / sc**2, 1)
                    entry["width_px"]  = round(m["width_px"]  / sc, 1)
            except Exception:
                pass
    # else: use_auto_polygon=False → entry keeps empty poly → draws as circle

    added.append(entry)
    log.append({"ts": ts, "action": "added", "det_id": None,
                "orig_x": entry["orig_x"], "orig_y": entry["orig_y"]})
    return {**state, "added": added, "log": log}


def _auto_polygon_from_click(
    base_img: Image.Image,
    dx: float, dy: float,
    sc: float,
    patch_radius: int = 35,
) -> list:
    """Try to extract a polygon for a manually clicked point using local thresholding.

    Extracts a patch around the click, runs Otsu thresholding, finds the
    connected component closest to the click centre, and returns its contour
    as a polygon in display-scale coordinates.

    Size constraints prevent the function from returning oversized polygons
    caused by background reflections or glass edges being misidentified as
    the target organism.

    Returns an empty list if no suitable contour is found (caller falls back
    to a plain circle).
    """
    try:
        import cv2
        arr = np.array(base_img.convert("RGB"))
        h, w = arr.shape[:2]
        cx, cy = int(round(dx)), int(round(dy))

        # Crop patch with padding
        x1 = max(0, cx - patch_radius)
        y1 = max(0, cy - patch_radius)
        x2 = min(w, cx + patch_radius)
        y2 = min(h, cy + patch_radius)
        patch = arr[y1:y2, x1:x2]
        if patch.size == 0:
            return []

        patch_area = (x2 - x1) * (y2 - y1)

        # Convert to grayscale and threshold
        gray  = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        # Invert if background is dark (typical for collembot images with ink)
        if gray.mean() < 128:
            gray = 255 - gray
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Pick the contour whose centroid is closest to the click centre
        # AND whose size is plausible for a Collembola
        pcx, pcy = cx - x1, cy - y1   # click relative to patch
        best_cnt, best_d = None, patch_radius
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # ── size constraints ──────────────────────────────────────────────
            # Min area: 20px² — ignore specks of noise
            if area < 20:
                continue
            # Max area: 50% of the patch — if the contour fills more than half
            # the patch it's almost certainly background, not an organism
            if area > 0.50 * patch_area:
                continue
            # Max bounding-box dimension: 80% of patch width/height
            _, _, bw, bh = cv2.boundingRect(cnt)
            if bw > 0.80 * (x2 - x1) or bh > 0.80 * (y2 - y1):
                continue
            # ─────────────────────────────────────────────────────────────────
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            ccx = M["m10"] / M["m00"]
            ccy = M["m01"] / M["m00"]
            d   = math.hypot(ccx - pcx, ccy - pcy)
            if d < best_d:
                best_d, best_cnt = d, cnt

        if best_cnt is None:
            return []

        # Convert contour coords back to display-image space
        pts = []
        for pt in best_cnt.reshape(-1, 2):
            pts.append([float(pt[0] + x1), float(pt[1] + y1)])

        # Simplify polygon (reduce vertex count while preserving shape)
        if len(pts) > 20:
            epsilon = 0.02 * cv2.arcLength(best_cnt, True)
            approx  = cv2.approxPolyDP(best_cnt, epsilon, True)
            pts = [[float(p[0][0] + x1), float(p[0][1] + y1)] for p in approx]

        return pts if len(pts) >= 3 else []

    except Exception:
        return []


# ============================================================================
# Adult / juvenile classification — biology-aware when scale is available
# ============================================================================
#
# Strategy
# --------
# When px/mm scale IS known:
#   Use DEB-grounded size thresholds for F. candida at 20°C (OECD 232):
#     - Adults (original 10 animals, 37–40 days old at test end): > ADULT_SIZE_MM mm
#     - Juveniles: < ADULT_SIZE_MM mm
#   Additionally run a multi-component GMM on just the juvenile fraction to
#   identify egg clutches (different hatching cohorts produce distinct size peaks).
#
# When px/mm scale is NOT known:
#   Fall back to a 2-component GMM on pixel lengths.
#   This is less reliable — the GMM may split clutches rather than life stages.
#
# Biological basis (DEB entry, Hamda 2014):
#   Lb = 0.307 mm (birth)    Li = 2.38 mm (ultimate)
#   At OECD test end (~37–40 days), original animals are ~1.2–1.8 mm.
#   A threshold of 1.0 mm cleanly separates them from offspring (<0.9 mm).
#   Tolerance raised to 1.2 mm to account for fast-growing late-stage juveniles.

ADULT_SIZE_MM          = 1.0   # mm — below this = juvenile
ADULT_SIZE_MM_UNCERTAIN = 1.2  # mm — between 1.0 and 1.2 = uncertain (could be
                                #       very large juvenile or small adult)
GMM_MIN_SEPARATION_SD  = 1.0   # fallback GMM: min separation in SD units
MAX_CLUTCH_COMPONENTS  = 4     # maximum number of clutch groups to fit


def classify_adults_juveniles(
    lengths: List[float],
    scale: Optional[float] = None,
) -> dict:
    """Classify organisms as adult or juvenile with clutch detection.

    Parameters
    ----------
    lengths : list of float
        Body lengths in pixels.
    scale : float or None
        pixels per mm. If provided, uses biology-based thresholds.

    Returns
    -------
    dict with keys:
        labels          : list[str]  — "adult", "juvenile", or "uncertain"
        threshold       : float      — decision boundary (in mm or px)
        threshold_source: str        — "biology" or "GMM"
        n_adult / n_juvenile / n_uncertain : int
        adult_mean / juvenile_mean  : float
        separation_ok   : bool
        unit            : "px" or "mm"
        warning         : str or None
        clutches        : list[dict] — clutch analysis (only when scale known)
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return {"labels": ["uncertain"] * len(lengths),
                "warning": "scikit-learn not available"}

    unit = "mm" if scale else "px"
    vals = np.array(lengths, dtype=float)

    if len(vals) < 6:
        return {
            "labels":   ["uncertain"] * len(lengths),
            "warning":  f"Too few detections ({len(vals)}) for classification — need ≥ 6.",
            "unit":     unit,
        }

    # ── Biology-based classification when scale is known ─────────────────────
    if scale and scale > 0:
        vals_mm = vals / scale

        threshold_lo = ADULT_SIZE_MM
        threshold_hi = ADULT_SIZE_MM_UNCERTAIN

        labels = []
        for v in vals_mm:
            if v >= threshold_hi:
                labels.append("adult")
            elif v < threshold_lo:
                labels.append("juvenile")
            else:
                labels.append("uncertain")

        n_adult     = labels.count("adult")
        n_juvenile  = labels.count("juvenile")
        n_uncertain = labels.count("uncertain")

        adult_vals = vals_mm[np.array(labels) == "adult"]
        juv_vals   = vals_mm[np.array(labels) == "juvenile"]

        adult_mean   = float(adult_vals.mean())   if len(adult_vals)   > 0 else None
        juv_mean     = float(juv_vals.mean())     if len(juv_vals)     > 0 else None

        warning = None
        if n_adult > FLAG_MAX_ADULTS:
            warning = (
                f"{n_adult} organisms classified as adults — OECD 232 expects "
                f"≤{FLAG_MAX_ADULTS}. Check for large juveniles being "
                f"misclassified or fusion errors creating oversized polygons."
            )

        # ── Clutch detection on juvenile fraction ─────────────────────────────
        clutches = []
        if len(juv_vals) >= 8:
            clutches = _detect_clutches(juv_vals, unit="mm")

        return {
            "labels":           labels,
            "threshold":        round(threshold_lo, 2),
            "threshold_hi":     round(threshold_hi, 2),
            "threshold_source": "biology (DEB Lb=0.31mm, Li=2.38mm)",
            "adult_mean":       round(adult_mean,  3) if adult_mean  else None,
            "juvenile_mean":    round(juv_mean,    3) if juv_mean    else None,
            "separation_ok":    True,
            "separation_sd":    None,
            "n_adult":          n_adult,
            "n_juvenile":       n_juvenile,
            "n_uncertain":      n_uncertain,
            "unit":             "mm",
            "warning":          warning,
            "clutches":         clutches,
        }

    # ── Fallback: 2-component GMM on pixel lengths ────────────────────────────
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          n_init=10, random_state=42)
    gmm.fit(vals.reshape(-1, 1))

    means = gmm.means_.flatten()
    stds  = np.sqrt(gmm.covariances_.flatten())
    order = np.argsort(means)
    means, stds = means[order], stds[order]

    w0, w1    = gmm.weights_[order[0]], gmm.weights_[order[1]]
    threshold = (means[0]/stds[0]**2 + means[1]/stds[1]**2) / \
                (1/stds[0]**2 + 1/stds[1]**2)

    pooled_sd     = math.sqrt((stds[0]**2 + stds[1]**2) / 2)
    separation    = (means[1] - means[0]) / (pooled_sd + 1e-9)
    separation_ok = separation >= GMM_MIN_SEPARATION_SD

    uncertain_band = pooled_sd * 0.5
    labels = []
    for v in vals:
        if abs(v - threshold) < uncertain_band:
            labels.append("uncertain")
        elif v < threshold:
            labels.append("juvenile")
        else:
            labels.append("adult")

    n_adult, n_juvenile, n_uncertain = (labels.count(c) for c in
                                         ("adult", "juvenile", "uncertain"))
    warning = None
    if not separation_ok:
        warning = (
            f"GMM peaks overlap (separation={separation:.1f} SD). "
            f"Without a scale, classification splits length clusters — "
            f"these may be egg clutches, not life stages. "
            f"Enter px/mm for biology-based adult/juvenile classification."
        )

    return {
        "labels":           labels,
        "threshold":        round(float(threshold), 2),
        "threshold_source": "GMM (no scale — enter px/mm for biology-based)",
        "adult_mean":       round(float(means[1]), 2),
        "juvenile_mean":    round(float(means[0]), 2),
        "separation_ok":    bool(separation_ok),
        "separation_sd":    round(float(separation), 2),
        "n_adult":          n_adult,
        "n_juvenile":       n_juvenile,
        "n_uncertain":      n_uncertain,
        "unit":             "px",
        "warning":          warning,
        "clutches":         [],
    }


def _detect_clutches(lengths_mm: np.ndarray, unit: str = "mm") -> List[dict]:
    """Fit a multi-component GMM to juvenile lengths to identify egg clutches.

    In a standard OECD 232 test at 20°C, F. candida lays eggs every ~7–10 days.
    Each clutch produces juveniles of similar size at test end, creating
    distinct peaks in the juvenile length distribution.

    Returns a list of dicts, one per detected clutch, sorted oldest → youngest:
        mean_length, std_length, n, unit, clutch_index (1=oldest/largest)
    """
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.model_selection import cross_val_score

        vals = lengths_mm.reshape(-1, 1)
        n    = len(vals)

        if n < 8:
            return []

        # Select optimal number of components by BIC
        best_bic, best_k = np.inf, 1
        for k in range(1, min(MAX_CLUTCH_COMPONENTS + 1, n // 4 + 1)):
            try:
                gm = GaussianMixture(n_components=k, covariance_type="full",
                                     n_init=5, random_state=42)
                gm.fit(vals)
                bic = gm.bic(vals)
                if bic < best_bic - 5:   # require meaningful improvement
                    best_bic, best_k = bic, k
            except Exception:
                break

        if best_k == 1:
            # Single cluster — one clutch or too few organisms to distinguish
            return [{
                "clutch_index": 1,
                "mean_length":  round(float(lengths_mm.mean()), 3),
                "std_length":   round(float(lengths_mm.std()),  3),
                "n":            n,
                "unit":         unit,
                "label":        "single cohort",
            }]

        gm = GaussianMixture(n_components=best_k, covariance_type="full",
                             n_init=10, random_state=42)
        gm.fit(vals)

        means   = gm.means_.flatten()
        stds    = np.sqrt(gm.covariances_.flatten())
        weights = gm.weights_
        labels  = gm.predict(vals)

        # Sort clusters by mean — largest = oldest clutch (laid first)
        order  = np.argsort(means)[::-1]
        result = []
        for rank, idx in enumerate(order):
            n_in_cluster = int((labels == idx).sum())
            result.append({
                "clutch_index": rank + 1,
                "mean_length":  round(float(means[idx]),   3),
                "std_length":   round(float(stds[idx]),    3),
                "n":            n_in_cluster,
                "unit":         unit,
                "label":        f"Clutch {rank+1} (oldest→youngest)",
            })
        return result

    except Exception:
        return []


def _undo_last_correction(state: dict) -> dict:
    """Reverse the most recent correction."""
    if not state or not state.get("log"):
        return state
    log  = list(state["log"])
    last = log.pop()
    if last["action"] in ("removed", "restored"):
        det_id  = last["det_id"]
        removed = set(state["removed"])
        if last["action"] == "removed"   and det_id in removed: removed.remove(det_id)
        elif last["action"] == "restored" and det_id not in removed: removed.add(det_id)
        return {**state, "removed": removed, "log": log}
    elif last["action"] == "added":
        added = list(state["added"])
        if added: added.pop()
        return {**state, "added": added, "log": log}
    return state


def _export_correction(state: dict, px_per_mm: float):
    """Export corrected detections CSV + audit log CSV + summary JSON."""
    if not state:
        return None, None, "{}"

    dets    = state["dets"]
    removed = state["removed"]
    added   = state["added"]
    scale   = px_per_mm if px_per_mm and px_per_mm > 0 else None

    fields = ["id", "source", "status", "confidence", "length_px", "area_px"]
    if scale:
        fields += ["length_mm", "area_mm2"]

    rows = []
    for d in dets:
        status = "removed" if d["id"] in removed else "confirmed"
        row = {"id": d["id"], "source": "model", "status": status,
               "confidence": d["confidence"],
               "length_px":  d["length_px"],
               "area_px":    d["area_px"]}
        if scale:
            row["length_mm"] = round(d["length_px"] / scale,    4)
            row["area_mm2"]  = round(d["area_px"]   / scale**2, 4)
        rows.append(row)

    for i, pt in enumerate(added):
        source = "promoted" if pt.get("poly") else "manual"
        row = {"id": f"manual_{i}", "source": source, "status": "added",
               "confidence": pt.get("confidence"),
               "length_px":  pt.get("length_px"),
               "area_px":    pt.get("area_px")}
        if scale:
            lp = pt.get("length_px")
            ap = pt.get("area_px")
            row["length_mm"] = round(lp / scale,    4) if lp else None
            row["area_mm2"]  = round(ap / scale**2, 4) if ap else None
        rows.append(row)

    det_csv = Path(tempfile.mkdtemp()) / "collembot_corrected_detections.csv"
    with open(det_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    log_csv = Path(tempfile.mkdtemp()) / "collembot_correction_audit_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","action","det_id","orig_x","orig_y"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(state.get("log", []))

    n_rem, n_add = len(removed), len(added)
    summary = {
        "verification_status": "MANUALLY_VERIFIED" if (n_rem or n_add) else "AUTOMATED",
        "model_version":       state.get("model_version", "unknown"),
        "inference_timestamp": state.get("ts", ""),
        "export_timestamp":    datetime.now(timezone.utc).isoformat(),
        "image_original_size": [state.get("orig_w"), state.get("orig_h")],
        "scale_px_per_mm":     scale,
        "model_count":         len(dets),
        "removed_count":       n_rem,
        "added_count":         n_add,
        "corrected_count":     len(dets) - n_rem + n_add,
        "removed_ids":         sorted(removed),
        "n_corrections":       len(state.get("log", [])),
    }
    return str(det_csv), str(log_csv), json.dumps(summary, indent=2)



# ============================================================================
# Quality-flagging helpers (Tab 2)
# ============================================================================

# Thresholds — documented here so they appear in the methods section
# of any paper that uses these flags.
FLAG_GRID_DIVERGENCE_THRESHOLD = 0.20   # relative |ori-shift|/mean(ori,shift)
FLAG_CONF_THRESHOLD            = 0.55   # minimum acceptable mean confidence
FLAG_FUSION_LOW                = 0.45   # fusion_ratio = fused/(ori+shift)
FLAG_FUSION_HIGH               = 0.82
FLAG_LOW_COUNT                 = 10     # minimum plausible offspring count
FLAG_MAX_ADULTS                = 10     # OECD 232: exactly 10 adults added at start
FLAG_MAX_ADULTS_TOLERANCE      = 12     # >12 is almost certainly an error
GRUBBS_ALPHA                   = 0.05   # Grubbs outlier test significance level


def _compute_image_flags(fused: Sequence[dict],
                          raw_filtered: Sequence[dict],
                          circle,
                          n_adults: Optional[int] = None) -> dict:
    """Compute per-image quality flags.

    Returns a dict of flag booleans and diagnostic values.
    All thresholds are module-level constants so they are reproducible
    and citable in a methods section.

    Flag definitions
    ----------------
    flag_grid_divergence
        |ori − shift| / mean(ori, shift) > 20 %.  The two overlapping
        tile grids should detect similar counts because ~75 % of the
        image is covered by both.  Large divergence suggests tiling
        artefacts, foam, or spatially uneven image quality.

    flag_low_confidence
        Mean detection confidence < 0.55.  Since BEST_CONF = 0.45 is
        the hard minimum, a mean just above it indicates the model was
        broadly uncertain — typically from low contrast, foam, or
        unusual substrate colour.

    flag_fusion_ratio
        fused / (ori + shift) outside [0.45, 0.82].  With 50 % tile
        overlap the expected ratio is ~0.57; values outside the range
        indicate systematic under- or over-deduplication.

    flag_low_count
        Fused count < 10.  May reflect genuine high toxicity or a
        processing failure; both require manual confirmation.

    flag_no_circle
        Dish boundary detection failed.  Detections are not spatially
        filtered, so background false positives may be included.
    """
    ori_n   = sum(1 for p in raw_filtered if p.get("grid") == "ori")
    shift_n = sum(1 for p in raw_filtered if p.get("grid") == "shift")
    total_raw = ori_n + shift_n
    fused_n   = len(fused)

    # 1. Grid divergence
    if total_raw > 0:
        mean_raw   = total_raw / 2.0
        divergence = abs(ori_n - shift_n) / mean_raw
    else:
        divergence = 0.0
    flag_div = divergence > FLAG_GRID_DIVERGENCE_THRESHOLD

    # 2. Mean confidence
    confs     = [p.get("confidence", 0.0) for p in fused]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    flag_conf = (mean_conf < FLAG_CONF_THRESHOLD) and fused_n > 0

    # 3. Fusion efficiency
    if total_raw > 0:
        fusion_ratio = fused_n / total_raw
        flag_fusion  = not (FLAG_FUSION_LOW <= fusion_ratio <= FLAG_FUSION_HIGH)
    else:
        fusion_ratio = None
        flag_fusion  = False

    # 4. Low count
    flag_low = fused_n < FLAG_LOW_COUNT

    # 5. No circle
    flag_circ = circle is None

    # 6. Excess adults (OECD 232 biological constraint)
    # >10 adults classified by GMM is suspicious; >12 is almost certainly wrong.
    flag_excess_adults = False
    if n_adults is not None and n_adults > FLAG_MAX_ADULTS:
        flag_excess_adults = True

    needs_review = any([flag_div, flag_conf, flag_fusion, flag_low,
                        flag_circ, flag_excess_adults])

    return {
        "needs_review":          needs_review,
        "flag_grid_divergence":  flag_div,
        "flag_low_confidence":   flag_conf,
        "flag_fusion_ratio":     flag_fusion,
        "flag_low_count":        flag_low,
        "flag_no_circle":        flag_circ,
        "flag_excess_adults":    flag_excess_adults,
        "n_adults_classified":   n_adults,
        # Diagnostic values (reportable)
        "ori_count":             ori_n,
        "shift_count":           shift_n,
        "grid_divergence_pct":   round(divergence * 100, 1),
        "mean_confidence":       round(mean_conf, 3),
        "fusion_ratio":          round(fusion_ratio, 3) if fusion_ratio is not None else None,
    }


def _grubbs_outliers(values: List[float], alpha: float = GRUBBS_ALPHA) -> List[bool]:
    """Grubbs test for a single outlier (two-sided) in a small sample.

    Returns a boolean list of the same length; True = outlier.
    Only applied for n ≥ 3.  For n < 3 all values return False.

    Reference: Grubbs, F.E. (1950). Sample criteria for testing outlying
    observations. Ann. Math. Stat. 21(1):27-58.
    """
    from scipy import stats as sp_stats
    n = len(values)
    if n < 3:
        return [False] * n
    arr  = np.array(values, dtype=float)
    mean = arr.mean()
    std  = arr.std(ddof=1)
    if std == 0:
        return [False] * n
    G = np.abs(arr - mean) / std
    # Critical value via t-distribution
    t_crit = sp_stats.t.ppf(1.0 - alpha / (2.0 * n), df=n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    return [bool(g > G_crit) for g in G]


def _make_batch_thumbnail(
    base: Image.Image,
    fused: Sequence[dict],
    flags: dict,
    img_name: str,
    group: str,
    max_dim: int = 800,
) -> Image.Image:
    """Compact annotated thumbnail for the batch gallery.

    Shows fused detections on the original image with a count badge
    and a coloured flag strip at the top — green (ok) or red (needs review).
    Kept small so the gallery loads quickly.
    """
    # Downsample for display
    w, h   = base.size
    scale  = min(1.0, max_dim / max(w, h))
    dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
    thumb  = base.resize((dw, dh), Image.LANCZOS).convert("RGB")

    draw = ImageDraw.Draw(thumb, "RGBA")
    font = ImageFont.load_default()

    # Draw detections
    for pol in fused:
        geom = pol.get("polygon")
        if geom is None or geom.is_empty:
            continue
        for subp in iter_subpolygons(geom):
            pts = [(x * scale, y * scale) for x, y in subp.exterior.coords]
            draw.polygon(pts, outline=(255, 215, 0, 230),
                         fill=(255, 215, 0, 45))

    # Flag strip at top
    needs_review = flags.get("needs_review", False)
    strip_colour = (200, 40, 40, 220) if needs_review else (30, 160, 60, 220)
    draw.rectangle([(0, 0), (dw, 22)], fill=strip_colour)

    active = []
    if flags.get("flag_grid_divergence"):  active.append("grid")
    if flags.get("flag_low_confidence"):   active.append("conf")
    if flags.get("flag_fusion_ratio"):     active.append("fusion")
    if flags.get("flag_low_count"):        active.append("count")
    if flags.get("flag_no_circle"):        active.append("circle")
    if flags.get("flag_grubbs_outlier"):   active.append("Grubbs")

    status_txt = ("⚠ " + ", ".join(active)) if active else "✓ ok"
    header = f"{img_name}  |  {group}  |  n={len(fused)}  |  {status_txt}"
    draw.text((4, 4), header, fill=(255, 255, 255, 255), font=font)

    return thumb


# ============================================================================
# v0.1.1 additions — full-resolution annotated export (Compare groups tab)
# ============================================================================
# AI-assisted (Claude Opus 5.5, 2026-10-20); reviewed by: <pending>.
# Nothing in this block changes detection, fusion or counting.

APP_VERSION = "0.2.2"

# v0.2.1: title page shown before the app (interface text only).
USAGE_NOTICE_MD = """
## Before you start — please read

**CollemBot is free to use, but not free to run.**
This online version runs on a cloud GPU. We, the CollemBot developers, pay for this GPU
ourselves, for every hour it runs, and offer the service free of charge.
Usage has recently increased sharply.
To save costs, the app goes to sleep after 15 minutes without use. The next visitor wakes it
up, which may take a few minutes, and results that were not downloaded may be lost.

**Please help us keep CollemBot online for everyone:**

- **Run only what you need.** Upload only the images you want to analyse and avoid
  re-running the same images.
- **Test first.** Check threshold and px/mm scale on 2–3 images before you run a whole batch.
- **Download your results straight away** (ZIP or CSV), so you do not have to run them again
  and they are safe when the app goes to sleep.
- **Large or routine analyses:** please use the free **desktop version** on your own computer.
  It is this same app; your images stay on your computer.
  [Download and instructions](https://github.com/MichaWehrli/collembot_huggingface/tree/main/desktop).

**Questions or problems?** Contact [micha.wehrli@eawag.ch](mailto:micha.wehrli@eawag.ch).

If CollemBot is useful for your work, please cite
[doi:10.1093/etojnl/vgag068](https://doi.org/10.1093/etojnl/vgag068). Thank you!
"""

USAGE_REMINDER_MD = (
    "💡 *Free to use, but we pay for every GPU hour: please run only what you need and "
    "download results straight away (the app sleeps after 15 min without use). "
    "Large or routine analyses: use the desktop version "
    "([download](https://github.com/MichaWehrli/collembot_huggingface/tree/main/desktop)). "
    "Help: [micha.wehrli@eawag.ch](mailto:micha.wehrli@eawag.ch)*"
)

# Title-page graphic (file next to this script on the Space: /app/collembot_title.jpg).
# If the file is missing, the title page is shown without it.
TITLE_IMAGE = Path(__file__).resolve().parent / "collembot_title.jpg"


def _title_image_kwargs():
    """Hide download/share buttons on the graphic (Gradio 6: `buttons`; older: flags)."""
    import inspect
    params = inspect.signature(gr.Image.__init__).parameters
    if "buttons" in params:
        return {"buttons": ["fullscreen"]}
    return {k: False for k in ("show_download_button", "show_share_button") if k in params}


# v0.2.2: where the app runs (recorded in run_settings.json). Hugging Face sets SPACE_ID
# inside a Space; the desktop launcher sets COLLEMBOT_RUNTIME=desktop.
RUNTIME_INFO = {
    "runtime": ("huggingface-space" if os.environ.get("SPACE_ID")
                else os.environ.get("COLLEMBOT_RUNTIME", "local")),
    "device": None,
}


def _title_page_default() -> bool:
    """Title page is shown unless COLLEMBOT_TITLE_PAGE is 0/false/no/off."""
    value = os.environ.get("COLLEMBOT_TITLE_PAGE", "1").strip().lower()
    return value not in ("0", "false", "no", "off")

SCALE_HELP = (
    "Measure once per camera setup in ImageJ/Fiji: draw a straight line between "
    "two ruler marks a known distance apart, Analyze > Measure, then "
    "pixels / mm (e.g. 480 px over 10 mm = 48 px/mm). "
    "Measure over 10 mm or more for precision."
)

_FLAG_SHORT_NAMES = [
    ("flag_grid_divergence", "grid"),
    ("flag_low_confidence",  "conf"),
    ("flag_fusion_ratio",    "fusion"),
    ("flag_low_count",       "count"),
    ("flag_no_circle",       "circle"),
    ("flag_grubbs_outlier",  "Grubbs"),
    ("flag_excess_adults",   "adults>10"),
]


def _active_flag_names(flags: dict) -> List[str]:
    """Short names of the quality flags that are set (same names as the table)."""
    return [short for key, short in _FLAG_SHORT_NAMES if flags.get(key)]


def _annotation_font(size_px: int):
    """TrueType font at ``size_px`` (DejaVu Sans ships with matplotlib)."""
    try:
        from matplotlib import font_manager
        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
        return ImageFont.truetype(path, size_px)
    except Exception:
        try:
            return ImageFont.load_default(size=size_px)   # Pillow >= 10.1
        except Exception:
            return ImageFont.load_default()


def _make_fullres_annotation(
    base: Image.Image,
    fused: Sequence[dict],
    circle,
    flags: dict,
    img_name: str,
    group: str,
    *,
    conf_threshold: float,
    scale: Optional[float],
) -> Image.Image:
    """Annotated copy of the original image at full resolution, for checking.

    Draws on the original pixels (nothing is resized):
      - each fused detection outline (yellow) with its ID; the ID equals the
        ``id`` column of the batch CSV for this image,
      - the length axis used for ``length_px`` (cyan),
      - the dish circle used to filter detections (yellow ring), if found.
    A footer strip is added below the image (original pixels stay visible) with
    file, group, count, review flags, threshold, scale, model/app version, date.
    Line width and font size scale with the image size.
    """
    img = base.convert("RGB").copy()
    w, h = img.size
    s = max(w, h) / 2000.0                       # 1.0 for a 2000 px long side
    lw = max(1, int(round(1.5 * s)))     # outline width
    fsize = max(14, int(round(20 * s)))
    font = _annotation_font(fsize)
    stroke = max(1, int(round(2 * s)))
    draw = ImageDraw.Draw(img, "RGBA")

    if circle:
        cx, cy, r = circle
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                     outline=(255, 255, 0, 200), width=lw)

    for idx, pol in enumerate(fused):
        geom = pol.get("polygon")
        if geom is None or geom.is_empty:
            continue
        for subp in iter_subpolygons(geom):
            pts = [(float(x), float(y)) for x, y in subp.exterior.coords]
            draw.line(pts + [pts[0]], fill=(255, 215, 0, 255), width=lw, joint="curve")
        m = pol.get("measurements") or measure_polygon(geom)
        if m.get("axis_start") and m.get("axis_end"):
            draw.line([m["axis_start"], m["axis_end"]],
                      fill=(0, 200, 255, 230), width=max(1, lw // 2))
        minx, miny, maxx, maxy = geom.bounds
        tx = min(max(0.0, maxx + 2 * lw), w - 4 * fsize)
        ty = min(max(0.0, miny - fsize * 0.2), h - fsize)
        draw.text((tx, ty), str(idx), font=font, fill=(255, 255, 255, 255),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 255))

    # ── footer strip ──────────────────────────────────────────────────────────
    active = _active_flag_names(flags)
    review = f"YES ({', '.join(active)})" if flags.get("needs_review") else "no"
    scale_txt = f"{scale:.2f} px/mm (manual)" if scale else "not given (no size classes)"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"{img_name}   |   group: {group}   |   count: {len(fused)}   |   needs review: {review}",
        f"confidence threshold: {conf_threshold:.2f}   |   scale: {scale_txt}   |   "
        f"model: {MODEL_VERSION_STR}   |   CollemBot v{APP_VERSION}   |   {stamp}",
    ]
    pad = int(fsize * 0.6)
    band_h = pad * 2 + int(fsize * 1.35) * len(lines)
    out = Image.new("RGB", (w, h + band_h), (20, 20, 20))
    out.paste(img, (0, 0))
    d2 = ImageDraw.Draw(out)
    y = h + pad
    for line in lines:
        d2.text((pad, y), line, font=font, fill=(255, 255, 255))
        y += int(fsize * 1.35)
    return out


def _run_settings(conf_threshold: float, scale: Optional[float],
                  all_results: Sequence[tuple]) -> dict:
    """Settings and software versions of one Compare-groups run (for the ZIP)."""
    import platform
    versions = {"python": platform.python_version()}
    for mod in ("torch", "ultralytics", "shapely", "numpy", "gradio", "PIL", "cv2"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    images = []
    for name, img_name, fused, flags, gmm_pre in all_results:
        entry = {
            "image": img_name,
            "group": name,
            "count": len(fused),
            "needs_review": bool(flags.get("needs_review")),
            "flags": _active_flag_names(flags),
        }
        if scale:
            entry.update({k: gmm_pre.get(k) for k in ("n_adult", "n_juvenile", "n_uncertain")})
        images.append(entry)
    return {
        "collembot_app_version": APP_VERSION,
        "model_version": MODEL_VERSION_STR,
        "runtime": RUNTIME_INFO.get("runtime"),
        "device": RUNTIME_INFO.get("device"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confidence_threshold": round(float(conf_threshold), 3),
        "fusion_iou_threshold": BEST_IOU,
        "fusion_alpha": ALPHA,
        "tile_size_px": TILE_SIZE,
        "tile_shift_px": SHIFT,
        "scale_px_per_mm": scale,
        "scale_source": "manual entry" if scale else "not provided",
        "size_classification": ("biology (DEB thresholds "
                                f"{ADULT_SIZE_MM}-{ADULT_SIZE_MM_UNCERTAIN} mm)")
                               if scale else "not computed (no scale)",
        "software_versions": versions,
        "images": images,
    }


def _build_results_zip(image_paths: Sequence[Path], csv_path: Optional[str],
                       settings: dict) -> str:
    """ZIP with full-resolution annotated JPEGs, the batch CSV and run settings."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    zip_path = Path(tempfile.mkdtemp()) / f"collembot_results_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in image_paths:
            # JPEGs are already compressed: store them as they are
            zf.write(p, arcname=f"annotated_images/{Path(p).name}",
                     compress_type=zipfile.ZIP_STORED)
        if csv_path:
            zf.write(csv_path, arcname=Path(csv_path).name,
                     compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("run_settings.json", json.dumps(settings, indent=2, default=str),
                    compress_type=zipfile.ZIP_DEFLATED)
    return str(zip_path)


def launch_app(
    model_path: str,
    share: bool = False,
    fuse: bool = False,
    example_image: Optional[str] = None,
    preferred_device: str = DEFAULT_DEVICE,
    predict_batch_size: int = PREDICT_BATCH,
    use_half: Optional[bool] = None,
    title_page: Optional[bool] = None,
    inbrowser: bool = False,
):
    if share:
        share = False
    model, device, half = load_model(
        model_path, fuse=fuse,
        preferred_device=preferred_device, use_half=use_half,
    )
    RUNTIME_INFO["device"] = str(device)
    if title_page is None:
        title_page = _title_page_default()

    default_img = None
    examples: List[List[str]] = []
    if example_image:
        try:
            img_path = Path(example_image)
            if img_path.exists():
                default_img = Image.open(img_path)
                default_img.filename = str(img_path)
                examples = [[str(img_path)]]
        except Exception:
            default_img = None
            examples = []

    # v0.1.1: UI helpers moved here from the removed Single-image tab
    def _update_scale_warn(v):
        return gr.update(visible=(not v or float(v) == 0))

    def _update_conf_warn(v):
        if v < 0.25:
            return (f"⚠️ **Very low threshold ({v:.2f})** — expect many false positives. "
                    f"Use the Hand correction tab to clean up results. "
                    f"Document this value in your methods.")
        elif v < 0.35:
            return (f"⚠️ **Low threshold ({v:.2f})** — more false positives likely. "
                    f"Validate with hand correction. Document in your methods.")
        elif v > 0.65:
            return (f"⚠️ **High threshold ({v:.2f})** — may miss real organisms, "
                    f"especially in challenging images. Use only if calibration confirmed this.")
        return f"✓ Standard threshold ({v:.2f})"

    # ── Tab 2: multi-image comparison ─────────────────────────────────────────
    def _compare(files, labels_str: str, px_per_mm_c: float, conf_threshold_c: float,
                 progress=gr.Progress(track_tqdm=True)):
        if not files:
            return None, None, "Upload at least one image.", [], None

        scale = px_per_mm_c if px_per_mm_c and px_per_mm_c > 0 else None
        unit  = "mm" if scale else "px"

        raw_labels = [l.strip() for l in labels_str.split(",") if l.strip()]
        labelled   = []
        for i, f in enumerate(files):
            name = raw_labels[i] if i < len(raw_labels) else Path(f.name).stem
            labelled.append((name, f.name))

        # ── Phase 1: run inference + per-image flags ──────────────────────────
        # all_results: list of (group, img_name, fused, flags)
        all_results = []
        kde_groups  = []
        circles     = []   # v0.1.1: dish circle per image, for the annotated export

        for i, (name, path) in enumerate(labelled):
            progress(i / len(labelled),
                     desc=f"Processing {Path(path).name} ({i+1}/{len(labelled)})")
            pil   = Image.open(path).convert("RGB")
            fused, _, circle, raw_filtered, _ = infer_image(
                pil, model, device,
                use_half=half, batch_size=predict_batch_size,
                conf_threshold=conf_threshold_c,
            )
            # Run GMM here so we can pass n_adults to the flag checker
            lengths_px = [pol.get("measurements", {}).get("length_px", 0.0)
                          for pol in fused
                          if pol.get("measurements", {}).get("length_px", 0.0) > 0]
            gmm_pre    = classify_adults_juveniles(lengths_px, scale=scale)
            # v0.1.1: no adult count / adults>10 flag without a px/mm scale;
            # the GMM fallback is not a reliable life-stage split.
            n_adults   = gmm_pre.get("n_adult") if scale else None
            flags = _compute_image_flags(fused, raw_filtered, circle, n_adults=n_adults)
            # Store GMM result alongside so Phase 3 can reuse it
            all_results.append((name, Path(path).name, fused, flags, gmm_pre))
            circles.append(circle)
            kde_groups.append((name, _lengths_from_fused(fused, scale)))

        # ── Phase 2: Grubbs outlier test within each group ────────────────────
        # Group indices by label
        from collections import defaultdict
        group_indices: dict = defaultdict(list)
        for idx, (name, _, _, _, _) in enumerate(all_results):
            group_indices[name].append(idx)

        grubbs_flags = [False] * len(all_results)
        for name, idxs in group_indices.items():
            counts  = [len(all_results[i][2]) for i in idxs]
            outliers = _grubbs_outliers(counts, alpha=GRUBBS_ALPHA)
            for i, is_out in zip(idxs, outliers):
                grubbs_flags[i] = is_out
                if is_out:
                    all_results[i][3]["flag_grubbs_outlier"] = True
                    all_results[i][3]["needs_review"]        = True

        # Ensure flag_grubbs_outlier exists for non-outliers too
        for i, (_, _, _, flags, _) in enumerate(all_results):
            flags.setdefault("flag_grubbs_outlier", False)

        # ── Phase 3: GMM classification + summary table ───────────────────────
        header  = ("| Image | Group | Count | Adults | Juveniles | Mean length "
                   "| Grid div% | Conf | Fusion | ⚠ Review |")
        divider = "|-------|-------|-------|--------|-----------|-------------|"
        divider += "-----------|------|--------|----------|"
        rows_md = [header, divider]

        # Store per-image GMM results for the comparison plot
        adult_groups    = []  # (label, [adult lengths])
        juvenile_groups = []  # (label, [juvenile lengths])

        # Determine classification method label upfront
        threshold_source = all_results[0][4].get("threshold_source", "") if all_results else ""
        src_label = "biology (DEB)" if "biology" in threshold_source else "GMM (enter px/mm for biology-based)"

        for (name, img_name, fused, flags, gmm_pre) in all_results:
            lengths  = _lengths_from_fused(fused, scale)
            mean_l   = f"{sum(lengths)/len(lengths):.1f} {unit}" if lengths else "—"
            review   = "**⚠ YES**" if flags["needs_review"] else "✓"

            # Reuse GMM computed during inference loop
            n_adult    = gmm_pre.get("n_adult",    "—") if scale else "N/A*"
            n_juvenile = gmm_pre.get("n_juvenile", "—") if scale else "N/A*"
            gmm_warn   = " ⚠" if (scale and gmm_pre.get("warning")) else ""
            adult_flag = " 🔴" if flags.get("flag_excess_adults") else ""

            # Collect lengths by class for the overlaid plot
            labels  = gmm_pre.get("labels", [])
            lbl_idx = 0
            img_adult_l, img_juv_l = [], []
            for pol in fused:
                lp = pol.get("measurements", {}).get("length_px", 0.0)
                if lp > 0 and lbl_idx < len(labels):
                    v = lp / scale if scale else lp
                    if labels[lbl_idx] == "adult":
                        img_adult_l.append(v)
                    elif labels[lbl_idx] == "juvenile":
                        img_juv_l.append(v)
                    lbl_idx += 1
            adult_groups.append((img_name, img_adult_l))
            juvenile_groups.append((img_name, img_juv_l))

            active_flags = []
            if flags["flag_grid_divergence"]:        active_flags.append("grid")
            if flags["flag_low_confidence"]:         active_flags.append("conf")
            if flags["flag_fusion_ratio"]:           active_flags.append("fusion")
            if flags["flag_low_count"]:              active_flags.append("count")
            if flags["flag_no_circle"]:              active_flags.append("circle")
            if flags.get("flag_grubbs_outlier"):     active_flags.append("Grubbs")
            if flags.get("flag_excess_adults"):      active_flags.append("adults>10")
            flag_str = ", ".join(active_flags) if active_flags else "—"

            rows_md.append(
                f"| {img_name} | {name} | {len(fused)} "
                f"| {n_adult}{gmm_warn}{adult_flag} | {n_juvenile}{gmm_warn} "
                f"| {mean_l} "
                f"| {flags['grid_divergence_pct']}% "
                f"| {flags['mean_confidence']} "
                f"| {flags.get('fusion_ratio', '—')} "
                f"| {review} ({flag_str}) |"
            )

        n_flagged = sum(1 for _, _, _, f, _ in all_results if f["needs_review"])
        rows_md += [
            "",
            f"**{n_flagged}/{len(all_results)} images flagged for manual review.**",
            f"**Classification:** {src_label} &nbsp;|&nbsp; "
            f"**Confidence threshold:** {conf_threshold_c:.2f}",
            "" if scale else
            "**⚠️ NO SCALE PROVIDED — adult and juvenile counts show N/A. "
            "Enter px/mm and re-run for biology-based adult/juvenile classification.**",
            "",
            "**Flag key:** `grid` = grid divergence >20% &nbsp;|&nbsp; "
            "`conf` = mean confidence <0.55 &nbsp;|&nbsp; "
            "`fusion` = fusion ratio outside [0.45, 0.82] &nbsp;|&nbsp; "
            "`count` = count <10 &nbsp;|&nbsp; "
            "`circle` = dish boundary not detected &nbsp;|&nbsp; "
            f"`Grubbs` = statistical outlier within group (α={GRUBBS_ALPHA}) &nbsp;|&nbsp; "
            f"`adults>10` 🔴 = >{FLAG_MAX_ADULTS} adults (OECD 232: exactly 10 added) &nbsp;|&nbsp; "
            "⚠ on Adult/Juvenile = classification warning (shown only with a px/mm scale)",
        ]

        progress(0.95, desc="Building plots …")

        # v0.1.1: without a px/mm scale the bars show the total count only
        # (no GMM-based adult/juvenile split).
        if not scale:
            adult_groups    = [(n, []) for n, _ in adult_groups]
            juvenile_groups = [(n, []) for n, _ in juvenile_groups]
        unc_label = "Uncertain" if scale else "Total (enter px/mm for size classes)"

        # ── Decide view mode ──────────────────────────────────────────────────
        # ≤8 images → per-image detail; >8 → per-group aggregate (mean ± SD)
        PER_IMAGE_THRESHOLD = 8
        n_images  = len(all_results)
        per_image = n_images <= PER_IMAGE_THRESHOLD

        from collections import defaultdict as _dd
        group_totals    = _dd(list)
        group_adults    = _dd(list)
        group_juveniles = _dd(list)
        group_lengths   = _dd(list)

        for (name, img_name, fused, flags, gmm_pre), ag, jg in zip(
                all_results, adult_groups, juvenile_groups):
            group_totals[name].append(len(fused))
            group_adults[name].append(len(ag[1]))
            group_juveniles[name].append(len(jg[1]))

        for gname, lengths in kde_groups:
            group_lengths[gname].extend(lengths)

        unique_groups = list(dict.fromkeys(r[0] for r in all_results))

        all_clutches = [r[4].get("clutches", []) for r in all_results]
        has_clutches = any(len(c) > 1 for c in all_clutches)
        n_panels     = 3 if has_clutches else 2

        fig, axes = plt.subplots(1, n_panels,
                                  figsize=(7 * n_panels, 5), dpi=120)
        ax_kde = axes[0]
        ax_bar = axes[1]
        ax_cl  = axes[2] if n_panels == 3 else None

        colors_group = ["#d32f2f", "#1565c0", "#2e7d32", "#f57c00",
                        "#6a1b9a", "#00838f", "#4e342e", "#37474f"]

        # ── Panel 1: KDE pooled by group ───────────────────────────────────────
        all_flat = [l for ls in group_lengths.values() for l in ls]
        xs_global = np.linspace(min(all_flat), max(all_flat), 300) if all_flat else None
        for gi, gname in enumerate(unique_groups):
            lengths = group_lengths[gname]
            if not lengths or xs_global is None:
                continue
            col = colors_group[gi % len(colors_group)]
            ax_kde.hist(lengths, bins=30, color=col, alpha=0.15, density=True)
            try:
                from scipy.stats import gaussian_kde
                ax_kde.plot(xs_global, gaussian_kde(lengths, bw_method="scott")(xs_global),
                            color=col, linewidth=2,
                            label=f"{gname} (n={len(lengths)}, mean={sum(lengths)/len(lengths):.2f})")
            except Exception:
                pass
        if scale:
            ax_kde.axvline(ADULT_SIZE_MM, color="black", linestyle="--",
                           linewidth=1.5, label=f"Adult threshold ({ADULT_SIZE_MM}mm)")
        ax_kde.set_xlabel(f"Length ({unit})", fontsize=11)
        ax_kde.set_ylabel("Density", fontsize=11)
        ax_kde.set_title(f"Length distributions by group\n({src_label})", fontsize=11)
        ax_kde.legend(fontsize=8)

        # ── Panel 2: bar chart ─────────────────────────────────────────────────
        if per_image:
            img_names    = [r[1] for r in all_results]
            x            = np.arange(n_images)
            adult_ns     = [len(ag[1]) for ag in adult_groups]
            juv_ns       = [len(jg[1]) for jg in juvenile_groups]
            total_ns     = [len(fused) for _, _, fused, _, _ in all_results]
            uncertain_ns = [t - a - j for t, a, j in zip(total_ns, adult_ns, juv_ns)]
            width = min(0.6, 4.0 / max(n_images, 1))
            ax_bar.bar(x, juv_ns,       width, label="Juvenile",  color="#2196F3", alpha=0.8)
            ax_bar.bar(x, adult_ns,     width, bottom=juv_ns, label="Adult", color="#FF5722", alpha=0.8)
            ax_bar.bar(x, uncertain_ns, width, bottom=[a+j for a,j in zip(adult_ns, juv_ns)],
                       label=unc_label, color="#9E9E9E", alpha=0.6)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(img_names, rotation=30, ha="right", fontsize=8)
            ax_bar.set_title("Adult / Juvenile per image", fontsize=11)
        else:
            x = np.arange(len(unique_groups))
            width = 0.5
            mean_juv = [np.mean(group_juveniles[g]) for g in unique_groups]
            mean_adu = [np.mean(group_adults[g])    for g in unique_groups]
            mean_tot = [np.mean(group_totals[g])    for g in unique_groups]
            std_tot  = [np.std(group_totals[g], ddof=min(1, len(group_totals[g])-1))
                        for g in unique_groups]
            mean_unc = [max(0, mt-ma-mj) for mt,ma,mj in zip(mean_tot, mean_adu, mean_juv)]
            ax_bar.bar(x, mean_juv, width, label="Juvenile (mean)", color="#2196F3", alpha=0.8)
            ax_bar.bar(x, mean_adu, width, bottom=mean_juv, label="Adult (mean)", color="#FF5722", alpha=0.8)
            ax_bar.bar(x, mean_unc, width, bottom=[a+j for a,j in zip(mean_adu, mean_juv)],
                       label=f"{unc_label} (mean)", color="#9E9E9E", alpha=0.6)
            ax_bar.errorbar(x, mean_tot, yerr=std_tot, fmt="none",
                            color="black", capsize=5, linewidth=1.5, label="±1 SD (total)")
            rng = np.random.default_rng(42)
            for gi, gname in enumerate(unique_groups):
                reps = group_totals[gname]
                jitter = rng.uniform(-0.12, 0.12, len(reps))
                ax_bar.scatter(gi + jitter, reps, color="black", s=20, alpha=0.6, zorder=5)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(unique_groups, rotation=20, ha="right", fontsize=9)
            ax_bar.set_title(
                f"Adult / Juvenile by group (mean ± SD)\nDots = individual replicates (n={n_images})",
                fontsize=10)
        ax_bar.set_ylabel("Count", fontsize=11)
        ax_bar.legend(fontsize=9)

        # ── Panel 3: clutch analysis ───────────────────────────────────────────
        if ax_cl is not None:
            clutch_colors = ["#1a6b3c", "#2e9e5a", "#5bc98a", "#a0e4be"]
            if per_image:
                for img_idx, (r, clist) in enumerate(zip(all_results, all_clutches)):
                    for cl in clist:
                        col = clutch_colors[(cl["clutch_index"]-1) % len(clutch_colors)]
                        ax_cl.bar(img_idx + (cl["clutch_index"]-1)*0.2 - 0.2,
                                  cl["n"], 0.18, color=col, alpha=0.8,
                                  label=f"Clutch {cl['clutch_index']}" if img_idx == 0 else "")
                        ax_cl.text(img_idx + (cl["clutch_index"]-1)*0.2 - 0.2,
                                   cl["n"] + 1, f"{cl['mean_length']:.2f}",
                                   ha="center", fontsize=7, color=col)
                ax_cl.set_xticks(np.arange(n_images))
                ax_cl.set_xticklabels([r[1] for r in all_results], rotation=30, ha="right", fontsize=8)
                ax_cl.set_title("Egg clutch analysis\n(mean length in mm)", fontsize=11)
            else:
                group_clutch = _dd(lambda: _dd(list))
                for (name, _, _, _, gmm_pre), clist in zip(all_results, all_clutches):
                    for cl in clist:
                        group_clutch[name][cl["clutch_index"]].append(cl["n"])
                max_c = max((max(d.keys()) for d in group_clutch.values() if d), default=0)
                bar_w = 0.6 / max(max_c, 1)
                for gi, gname in enumerate(unique_groups):
                    for ci in range(1, max_c + 1):
                        vals = group_clutch[gname].get(ci, [])
                        if not vals:
                            continue
                        col = clutch_colors[(ci-1) % len(clutch_colors)]
                        offset = (ci - (max_c + 1)/2) * bar_w
                        ax_cl.bar(gi + offset, np.mean(vals), bar_w * 0.85, color=col, alpha=0.8,
                                  label=f"Clutch {ci}" if gi == 0 else "")
                        if len(vals) > 1:
                            ax_cl.errorbar(gi + offset, np.mean(vals),
                                           yerr=np.std(vals, ddof=1), fmt="none",
                                           color="black", capsize=3)
                ax_cl.set_xticks(np.arange(len(unique_groups)))
                ax_cl.set_xticklabels(unique_groups, rotation=20, ha="right", fontsize=9)
                ax_cl.set_title("Egg clutch analysis\n(mean ± SD per group)", fontsize=11)
            handles, labels_leg = ax_cl.get_legend_handles_labels()
            seen = {}
            for hh, ll in zip(handles, labels_leg):
                if ll not in seen: seen[ll] = hh
            ax_cl.legend(seen.values(), seen.keys(), fontsize=9)
            ax_cl.set_ylabel("Count", fontsize=11)

        view_note = "per image" if per_image else f"aggregated by group (n={n_images} images)"
        fig.suptitle(f"Collembola body-length & size-class comparison  ·  {view_note}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(buf.name); plt.close(fig)
        plot_path = buf.name

        csv_path  = _make_combined_csv_flagged(all_results, scale)
        table_md  = "\n".join(rows_md)

        # ── Gallery ───────────────────────────────────────────────────────────
        gallery_items = []
        export_dir    = Path(tempfile.mkdtemp())   # v0.1.1: full-resolution export
        export_files  = []
        for i, ((name, img_name, fused, flags, _gp), (_, path)) in enumerate(
                zip(all_results, labelled)):
            pil   = Image.open(path).convert("RGB")
            thumb = _make_batch_thumbnail(pil, fused, flags, img_name, name)
            tmp   = Path(tempfile.mkdtemp()) / f"{img_name}_annotated.jpg"
            thumb.save(str(tmp), quality=88)
            caption = (f"{img_name} | {name} | n={len(fused)}"
                       + (" | ⚠ review" if flags["needs_review"] else " | ✓"))
            gallery_items.append((str(tmp), caption))
            progress(0.96 + 0.03 * i / max(1, len(all_results)),
                     desc=f"Writing full-resolution image {i + 1}/{len(all_results)}")
            full = _make_fullres_annotation(
                pil, fused, circles[i], flags, img_name, name,
                conf_threshold=conf_threshold_c, scale=scale)
            out_path = export_dir / f"{i + 1:03d}_{Path(img_name).stem}_annotated.jpg"
            full.save(str(out_path), quality=95, subsampling=0)
            export_files.append(out_path)
            del full

        zip_path = _build_results_zip(
            export_files, str(csv_path),
            _run_settings(conf_threshold_c, scale, all_results))

        return plot_path, str(csv_path), table_md, gallery_items, zip_path

    # ── Gradio layout ──────────────────────────────────────────────────────────
    with gr.Blocks(title="CollemBot") as demo:
        gr.Markdown(f"# 🪲 CollemBot v{APP_VERSION} — Collembola counting & size analysis")

        # v0.2.1: title page; the tabs are shown only after "continue"
        with gr.Column(visible=title_page) as notice_col:
            with gr.Row(equal_height=False):
                if TITLE_IMAGE.exists():
                    with gr.Column(scale=2, min_width=280):
                        gr.Image(str(TITLE_IMAGE), show_label=False, container=False,
                                 interactive=False, **_title_image_kwargs())
                with gr.Column(scale=3, min_width=320):
                    gr.Markdown(USAGE_NOTICE_MD)
            btn_continue = gr.Button(
                "I understand — continue to CollemBot", variant="primary")
        usage_reminder = gr.Markdown(USAGE_REMINDER_MD, visible=False)

        with gr.Column(visible=not title_page) as app_col, gr.Tabs():

            # ── Tab 2: multi-image comparison ─────────────────────────────────
            with gr.Tab("Compare groups"):
                gr.Markdown(
                    "Upload multiple images and assign group labels "
                    "(e.g. `control, control, treatment, treatment`). "
                    "Images are processed in order and their length distributions "
                    "are overlaid for visual comparison.  \n"
                    "**Note:** use the same zoom/scale across all images for "
                    "pixel lengths to be comparable. Enter px/mm if known."
                )
                with gr.Row():
                    multi_files = gr.File(
                        label="Slide images",
                        file_count="multiple",
                        file_types=["image"],
                    )
                with gr.Row():
                    labels_inp  = gr.Textbox(
                        label="Group labels (comma-separated, one per image)",
                        placeholder="control, control, treatment_1, treatment_1",
                    )
                    px_per_mm_c = gr.Number(
                        label="Scale — pixels per mm (0 = unknown)",
                        value=0, precision=2,
                        info=SCALE_HELP,
                    )

                scale_warning2 = gr.Markdown(
                    "⚠️ **No scale entered** — counts and lengths (px) are reported, "
                    "but adult/juvenile classification is switched off. "
                    "Enter px/mm (see the ImageJ tip under the scale field) to enable it.",
                    visible=True
                )

                px_per_mm_c.change(fn=_update_scale_warn,
                                    inputs=[px_per_mm_c],
                                    outputs=[scale_warning2])
                with gr.Accordion("⚙️ Advanced settings", open=False):
                    conf_slider2 = gr.Slider(
                        minimum=0.10, maximum=0.80, value=BEST_CONF, step=0.05,
                        label=f"Confidence threshold (default: {BEST_CONF})",
                        info="Lower = finds more organisms but more false positives. "
                             "Applied identically to all images in the batch. "
                             "Value is saved in the export CSV.",
                    )
                    conf_warn2 = gr.Markdown("")
                    conf_slider2.change(fn=_update_conf_warn,
                                        inputs=[conf_slider2],
                                        outputs=[conf_warn2])
                btn2 = gr.Button("Run comparison", variant="primary")
                out_plot  = gr.Image(label="Length distributions", type="filepath")
                out_table = gr.Markdown(label="Summary table")
                out_csv2  = gr.File(label="Download combined CSV")
                out_zip   = gr.File(label="Download all results (ZIP): full-resolution "
                                          "annotated images + CSV + run settings")
                out_gallery = gr.Gallery(
                    label="Annotated images (green strip = ok, red strip = needs review)",
                    columns=2,
                    height=500,
                    object_fit="contain",
                )

                btn2.click(
                    fn=_compare,
                    inputs=[multi_files, labels_inp, px_per_mm_c, conf_slider2],
                    outputs=[out_plot, out_csv2, out_table, out_gallery, out_zip],
                )

            # ── Tab 3: hand correction — native Gradio, zero JavaScript ──────
            with gr.Tab("Hand correction"):
                gr.Markdown(
                    "**How to use:**\n"
                    "1. Upload a slide image and click **Run inference**\n"
                    "2. Select **Remove mode** or **Add mode**\n"
                    "3. **Click directly on the image** to remove false positives "
                    "or mark missed organisms — the image updates immediately\n"
                    "4. In **Add mode**, faint grey outlines show sub-threshold detections "
                    "the model found but wasn't confident about — clicking near one "
                    "promotes it using its actual polygon shape\n"
                    "5. Use the **detection list** on the right to remove by ID "
                    "or bulk-remove by typing IDs\n"
                    "6. Click **Export** when done\n\n"
                    "⚠️ Original model output is never modified — "
                    "all changes are recorded in the audit log."
                )

                corr_state = gr.State(None)

                with gr.Row():
                    corr_inp      = gr.Image(type="pil", label="Slide image")
                    px_per_mm_cor = gr.Number(
                        label="Scale — pixels per mm (0 = unknown)",
                        value=0, precision=2,
                    )
                btn_corr_run = gr.Button("▶ Run inference", variant="primary")

                with gr.Accordion("⚙️ Advanced settings", open=False):
                    conf_slider3 = gr.Slider(
                        minimum=0.10, maximum=0.80, value=BEST_CONF, step=0.05,
                        label=f"Confidence threshold (default: {BEST_CONF})",
                        info="Lower threshold shows more sub-threshold candidates "
                             "(faint grey outlines) for manual promotion in Add mode. "
                             "Value is saved in the audit log.",
                    )
                    conf_warn3 = gr.Markdown("")
                    conf_slider3.change(fn=_update_conf_warn,
                                        inputs=[conf_slider3],
                                        outputs=[conf_warn3])

                corr_tally = gr.Markdown("Upload an image and run inference to begin.")

                with gr.Row():
                    corr_mode = gr.Radio(
                        ["Remove mode", "Add mode"],
                        value="Remove mode",
                        label="Click mode",
                        info="Remove: click a detection to mark/unmark as false positive  |  Add: click blank space to mark a missed organism",
                    )
                    corr_add_style = gr.Radio(
                        ["Auto-polygon", "Circle marker"],
                        value="Auto-polygon",
                        label="Add style",
                        info="Auto-polygon: tries to fit a contour from the image  |  Circle marker: always places a plain dot",
                        visible=False,
                    )

                # Show add-style selector only in Add mode
                corr_mode.change(
                    fn=lambda m: gr.update(visible=(m == "Add mode")),
                    inputs=[corr_mode],
                    outputs=[corr_add_style],
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        corr_canvas = gr.Image(
                            label="Click on a detection to correct it  ·  zoom/pan with scroll and drag",
                            type="pil",
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("**Detection list**")
                        corr_bulk = gr.Textbox(
                            label="Bulk remove by ID (e.g. 3,7,12 → Enter)",
                            placeholder="3,7,12",
                        )
                        corr_df = gr.DataFrame(
                            headers=["ID", "Status", "Confidence", "Length (px)"],
                            label="Detections",
                            wrap=True,
                        )

                with gr.Row():
                    btn_corr_undo  = gr.Button("↩ Undo",               variant="secondary")
                    btn_corr_reset = gr.Button("↺ Reset to model output", variant="stop")

                gr.Markdown("---")
                with gr.Row():
                    px_per_mm_exp = gr.Number(
                        label="Scale for export — pixels per mm (0 = skip mm columns)",
                        value=0, precision=2,
                    )
                btn_export    = gr.Button("💾 Export corrected results", variant="primary")
                out_det_csv   = gr.File(label="Corrected detections CSV")
                out_audit_csv = gr.File(label="Correction audit log CSV")
                out_corr_json = gr.Code(label="Correction summary", language="json")

                # ── callbacks — all pure Python, no JS ────────────────────────

                def _corr_run(pil_img, conf_threshold_corr,
                              progress=gr.Progress(track_tqdm=True)):
                    if pil_img is None:
                        return None, None, None, "Upload an image first."
                    fused, _, _, _, candidates = infer_image(
                        pil_img, model, device,
                        use_half=half, batch_size=predict_batch_size,
                        conf_threshold=conf_threshold_corr,
                        progress=progress,
                    )
                    state  = _build_correction_state(pil_img, fused, candidates)
                    canvas = _draw_correction_image(state)
                    df     = _build_correction_df(state)
                    tally  = _correction_tally(state)
                    return state, canvas, df, tally

                btn_corr_run.click(
                    fn=_corr_run,
                    inputs=[corr_inp, conf_slider3],
                    outputs=[corr_state, corr_canvas, corr_df, corr_tally],
                )

                def _corr_click(state, mode, add_style, evt: gr.SelectData):
                    """Called when user clicks on the correction canvas."""
                    if state is None:
                        return state, None, None, "Run inference first."
                    x, y = float(evt.index[0]), float(evt.index[1])
                    if mode == "Remove mode":
                        det_id = _find_nearest_detection(x, y, state["dets"],
                                                          state["removed"])
                        if det_id is not None:
                            state = _toggle_detection(det_id, state)
                    else:  # Add mode — try to promote nearest sub-threshold candidate
                        best_cand, best_dist = None, CORR_CLICK_RADIUS
                        for cand in state.get("candidates", []):
                            d = math.hypot(cand["cx"] - x, cand["cy"] - y)
                            if d < best_dist:
                                best_dist, best_cand = d, cand
                        px = best_cand["cx"] if best_cand else x
                        py = best_cand["cy"] if best_cand else y
                        # Pass add_style so _add_manual_point knows whether to
                        # attempt auto-polygon or always use a plain circle
                        state = _add_manual_point(
                            px, py, state,
                            candidate=best_cand,
                            use_auto_polygon=(add_style == "Auto-polygon"),
                        )
                    canvas = _draw_correction_image(state)
                    df     = _build_correction_df(state)
                    tally  = _correction_tally(state)
                    return state, canvas, df, tally

                corr_canvas.select(
                    fn=_corr_click,
                    inputs=[corr_state, corr_mode, corr_add_style],
                    outputs=[corr_state, corr_canvas, corr_df, corr_tally],
                )

                def _corr_bulk(bulk_str, state):
                    if state is None or not bulk_str:
                        return state, None, None, ""
                    for s in bulk_str.split(","):
                        try:
                            det_id = int(s.strip())
                            if 0 <= det_id < len(state["dets"]):
                                if det_id not in state["removed"]:
                                    state = _toggle_detection(det_id, state)
                        except ValueError:
                            pass
                    canvas = _draw_correction_image(state)
                    df     = _build_correction_df(state)
                    tally  = _correction_tally(state)
                    return state, canvas, df, tally

                corr_bulk.submit(
                    fn=_corr_bulk,
                    inputs=[corr_bulk, corr_state],
                    outputs=[corr_state, corr_canvas, corr_df, corr_tally],
                )

                def _corr_undo(state):
                    if state is None:
                        return state, None, None, ""
                    state  = _undo_last_correction(state)
                    canvas = _draw_correction_image(state)
                    df     = _build_correction_df(state)
                    tally  = _correction_tally(state)
                    return state, canvas, df, tally

                btn_corr_undo.click(
                    fn=_corr_undo,
                    inputs=[corr_state],
                    outputs=[corr_state, corr_canvas, corr_df, corr_tally],
                )

                def _corr_reset(state):
                    if state is None:
                        return state, None, None, ""
                    state  = {**state, "removed": set(), "added": [], "log": []}
                    canvas = _draw_correction_image(state)
                    df     = _build_correction_df(state)
                    tally  = _correction_tally(state)
                    return state, canvas, df, tally

                btn_corr_reset.click(
                    fn=_corr_reset,
                    inputs=[corr_state],
                    outputs=[corr_state, corr_canvas, corr_df, corr_tally],
                )

                btn_export.click(
                    fn=_export_correction,
                    inputs=[corr_state, px_per_mm_exp],
                    outputs=[out_det_csv, out_audit_csv, out_corr_json],
                )

            # ── Tab 4: Calibrate confidence threshold ─────────────────────────
            with gr.Tab("🎯 Calibrate threshold"):
                gr.Markdown(
                    "## Calibrate the confidence threshold for your imaging setup\n\n"
                    "Upload up to three images spanning your full dose-response range:\n"
                    "- **High count** — control vessel (~800 juveniles)\n"
                    "- **Mid count** — intermediate dose (~200–300 juveniles, optional)\n"
                    "- **Zero count** — highest dose (0 juveniles — complete reproduction suppression)\n\n"
                    "The sweep finds the single threshold that minimises combined error across all "
                    "provided images. The zero-count image is especially important — it calibrates "
                    "the false positive rate directly.\n\n"
                    "**Workflow:**\n"
                    "1. Upload 1–3 calibration images with known manual counts\n"
                    "2. Click **🎯 Run calibration sweep** (runs inference at 15 thresholds per image)\n"
                    "3. Review the per-image count curves and combined error plot\n"
                    "4. Click **✅ Apply to all tabs**\n\n"
                    "💡 *One calibration per imaging setup is sufficient. "
                    "Report the threshold and calibration procedure in your methods.*"
                )

                # ── three image rows ───────────────────────────────────────────
                img_labels = [
                    ("Image 1 — High count (control, ~800 juveniles)", "🟢"),
                    ("Image 2 — Mid count (~200–300 juveniles, optional)", "🟡"),
                    ("Image 3 — Zero count (highest dose, complete suppression, optional)", "🔴"),
                ]
                cal_inps, cal_counts, cal_scales = [], [], []

                for label, icon in img_labels:
                    with gr.Row():
                        img = gr.Image(type="pil", label=f"{icon} {label}")
                        with gr.Column(scale=1):
                            cnt = gr.Number(
                                label="Known count",
                                value=-1, precision=0,
                                info="Your manual count. Enter 0 for complete suppression. Leave -1 to skip."
                            )
                            scl = gr.Number(
                                label="px/mm (0 = unknown)",
                                value=0, precision=2,
                            )
                        cal_inps.append(img)
                        cal_counts.append(cnt)
                        cal_scales.append(scl)

                btn_calibrate = gr.Button(
                    "🎯 Run calibration sweep", variant="primary"
                )

                with gr.Row():
                    cal_plot = gr.Image(label="Calibration curves",
                                        type="filepath")
                    with gr.Column():
                        cal_result_md = gr.Markdown("Run calibration to see results.")
                        cal_threshold_out = gr.Number(
                            label="Recommended threshold",
                            value=BEST_CONF, precision=3, interactive=False,
                        )
                        btn_apply_threshold = gr.Button(
                            "✅ Apply to all tabs", variant="primary",
                            interactive=False,
                        )

                cal_apply_status = gr.Markdown("")

                # ── calibration sweep ──────────────────────────────────────────
                def _run_calibration(
                    img1, cnt1, scl1,
                    img2, cnt2, scl2,
                    img3, cnt3, scl3,
                    progress=gr.Progress(track_tqdm=True),
                ):
                    # Build list of active calibration points
                    cal_images, cal_known, cal_scl = [], [], []
                    for img, cnt, scl, label in zip(
                        [img1, img2, img3],
                        [cnt1, cnt2, cnt3],
                        [scl1, scl2, scl3],
                        ["Image 1 (high)", "Image 2 (mid)", "Image 3 (zero)"],
                    ):
                        if img is not None and cnt >= 0:
                            # cnt=0 is valid (complete suppression), cnt=-1 means skip
                            cal_images.append((img, int(cnt), scl or 0, label))

                    if not cal_images:
                        return None, "Upload at least one image.", BEST_CONF, \
                               gr.update(interactive=False)

                    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                                  0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                  0.70, 0.75, 0.80]

                    # counts_per_image[i] = list of counts at each threshold
                    counts_per_image = [[] for _ in cal_images]

                    total_steps = len(thresholds) * len(cal_images)
                    step = 0
                    for ti, th in enumerate(thresholds):
                        for ci, (pil, known, scl, label) in enumerate(cal_images):
                            progress(step / total_steps,
                                     desc=f"th={th:.2f}, {label} "
                                          f"({step+1}/{total_steps}) …")
                            fused, _, _, _, _ = infer_image(
                                pil, model, device,
                                use_half=half, batch_size=predict_batch_size,
                                conf_threshold=th,
                            )
                            counts_per_image[ci].append(len(fused))
                            step += 1

                    # ── find best threshold ────────────────────────────────────
                    # Mean absolute percentage error across all images
                    # For zero-count images use absolute error (can't divide by 0)
                    def combined_error(ti):
                        errs = []
                        for ci, (_, known, _, _) in enumerate(cal_images):
                            detected = counts_per_image[ci][ti]
                            if known > 0:
                                errs.append(abs(detected - known) / known)
                            else:
                                # For zero-count: penalise false positives linearly
                                errs.append(detected / 10.0)
                        return float(np.mean(errs))

                    combined_errs = [combined_error(ti) for ti in range(len(thresholds))]
                    best_idx = int(np.argmin(combined_errs))
                    best_th  = thresholds[best_idx]

                    # Tiebreak: prefer closest to default
                    min_err = combined_errs[best_idx]
                    for i, e in enumerate(combined_errs):
                        if abs(e - min_err) < 1e-6 and \
                           abs(thresholds[i] - BEST_CONF) < abs(best_th - BEST_CONF):
                            best_th, best_idx = thresholds[i], i

                    # ── build plot ─────────────────────────────────────────────
                    img_colors = ["#27ae60", "#f39c12", "#e74c3c",
                                  "#8e44ad", "#2980b9"]
                    n_panels = 2
                    fig, axes = plt.subplots(1, n_panels,
                                             figsize=(14, 5), dpi=120)
                    ax1, ax2 = axes

                    # Left: count-vs-threshold per image
                    for ci, (_, known, _, label) in enumerate(cal_images):
                        col = img_colors[ci % len(img_colors)]
                        cnts = counts_per_image[ci]
                        ax1.plot(thresholds, cnts, "o-", color=col,
                                 linewidth=2, label=f"{label} (target={known})")
                        ax1.axhline(known, color=col, linestyle="--",
                                    linewidth=1, alpha=0.5)
                        # Mark best threshold on each curve
                        ax1.scatter([best_th], [cnts[best_idx]],
                                    color=col, s=100, zorder=5,
                                    edgecolors="black", linewidth=1.5)

                    ax1.axvline(best_th, color="black", linestyle="-",
                                linewidth=2, alpha=0.7,
                                label=f"Best threshold: {best_th:.2f}")
                    ax1.axvline(BEST_CONF, color="#95a5a6", linestyle=":",
                                linewidth=1.5, label=f"Default: {BEST_CONF}")
                    ax1.set_xlabel("Confidence threshold", fontsize=11)
                    ax1.set_ylabel("Detected count", fontsize=11)
                    ax1.set_title("Detected count vs threshold (per image)", fontsize=12)
                    ax1.legend(fontsize=8)
                    ax1.set_xticks(thresholds)
                    ax1.tick_params(axis="x", rotation=45)
                    ax1.grid(True, alpha=0.3)

                    # Right: combined MAPE bar chart
                    bar_cols = ["#e74c3c" if i == best_idx else "#bdc3c7"
                                for i in range(len(thresholds))]
                    ax2.bar(thresholds, combined_errs, width=0.04,
                            color=bar_cols, alpha=0.85)
                    ax2.axvline(best_th, color="black", linestyle="-",
                                linewidth=2, alpha=0.7,
                                label=f"Best: {best_th:.2f}")
                    ax2.set_xlabel("Confidence threshold", fontsize=11)
                    ax2.set_ylabel("Combined mean error", fontsize=11)
                    ax2.set_title(
                        f"Combined error across {len(cal_images)} image(s)",
                        fontsize=12
                    )
                    ax2.set_xticks(thresholds)
                    ax2.tick_params(axis="x", rotation=45)
                    ax2.grid(True, alpha=0.3, axis="y")
                    ax2.legend(fontsize=9)

                    # Sensitivity = range of combined error
                    err_range = max(combined_errs) - min(combined_errs)
                    sensitivity = ("low (robust)" if err_range < 0.10
                                   else "medium" if err_range < 0.30
                                   else "high (validate carefully)")

                    fig.suptitle(
                        f"Multi-image calibration  ·  {len(cal_images)} image(s)  "
                        f"·  best threshold: {best_th:.2f}  "
                        f"·  sensitivity: {sensitivity}",
                        fontsize=12, fontweight="bold",
                    )
                    fig.tight_layout()
                    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    fig.savefig(buf.name); plt.close(fig)

                    # ── result text ────────────────────────────────────────────
                    rows = ""
                    for ci, (_, known, _, label) in enumerate(cal_images):
                        detected = counts_per_image[ci][best_idx]
                        err = abs(detected - known)
                        pct = err / max(known, 1) * 100
                        rows += (f"| {label} | {known} | {detected} "
                                 f"| {err} ({pct:.1f}%) |\n")

                    sensitivity_note = {
                        "low (robust)":
                            "✅ **Low sensitivity** — the model generalises well "
                            "to your imaging setup across the full count range.",
                        "medium":
                            "🟡 **Medium sensitivity** — results vary somewhat with "
                            "threshold. Use the calibrated value consistently.",
                        "high (validate carefully)":
                            "⚠️ **High sensitivity** — the combined error changes "
                            "strongly with threshold. Validate on additional images "
                            "before using for publication.",
                    }[sensitivity]

                    direction = ""
                    if best_th != BEST_CONF:
                        d = "lower" if best_th < BEST_CONF else "higher"
                        direction = (
                            f"\n\nThe optimal threshold ({best_th:.2f}) is **{d}** "
                            f"than the default ({BEST_CONF}). " + (
                            "Your images may need a lower threshold to catch all organisms."
                            if best_th < BEST_CONF else
                            "Your imaging setup produces high-confidence detections — "
                            "a higher threshold reduces false positives.")
                        )

                    result_md = (
                        f"### Calibration result — {len(cal_images)} image(s)\n\n"
                        f"**Recommended threshold: {best_th:.2f}**\n\n"
                        f"| Image | Known | Detected | Error |\n"
                        f"|-------|-------|----------|-------|\n"
                        f"{rows}\n"
                        f"{sensitivity_note}{direction}\n\n"
                        f"*Click **Apply to all tabs** to use this threshold.*"
                    )

                    return buf.name, result_md, best_th, gr.update(interactive=True)

                btn_calibrate.click(
                    fn=_run_calibration,
                    inputs=[
                        cal_inps[0], cal_counts[0], cal_scales[0],
                        cal_inps[1], cal_counts[1], cal_scales[1],
                        cal_inps[2], cal_counts[2], cal_scales[2],
                    ],
                    outputs=[cal_plot, cal_result_md, cal_threshold_out,
                             btn_apply_threshold],
                )

                # ── apply threshold to all tabs ────────────────────────────────
                def _apply_threshold(best_th):
                    msg = (
                        f"✅ Threshold **{best_th:.2f}** applied to all tabs. "
                        f"Run Compare groups or "
                        f"Hand correction — it will use this value."
                    )
                    return (gr.update(value=best_th),
                            gr.update(value=best_th),
                            msg)

                btn_apply_threshold.click(
                    fn=_apply_threshold,
                    inputs=[cal_threshold_out],
                    outputs=[conf_slider2, conf_slider3,
                             cal_apply_status],
                )

        # v0.2.1: leave the title page (no GPU work)
        btn_continue.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=True)),
            inputs=None,
            outputs=[notice_col, usage_reminder, app_col],
            queue=False,
        )

    demo.launch(share=share, debug=True, allowed_paths=["/tmp"], inbrowser=inbrowser)
    return demo


__all__ = [
    "launch_app", "infer_image", "find_main_circle", "fuse_graphcut",
    "filter_polygons_by_circle", "compose_visualisations", "draw_polygons",
]