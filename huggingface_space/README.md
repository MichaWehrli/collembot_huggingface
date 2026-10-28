---
title: COLLEMBOT
emoji: 🪲
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🪲 COLLEMBOT — AI-based Collembola Counting for OECD 232

**⚠️ Beta — validate counts before use in publications.**  
Version 0.2.2 (2026-10-28): desktop version (same app, runs on your own computer); counts unchanged from 0.2.0.  
Version 0.2.1 (2026-10-28): title page with a usage notice.  
Version 0.2.0 (2026-10-20): overlapping detections are now merged exactly as in the published pipeline (base repository `scripts/run_inference.py`). Counts differ from 0.1.x — see CHANGELOG.md.

> **💸 Please use this Space considerately.** It runs on a paid cloud GPU that the developers fund themselves and offer free of charge; every hour it runs costs us money, and usage has recently increased sharply. Run only what you need, test settings on 2–3 images before a full batch, and download your results straight away. The Space goes to sleep after 15 minutes without use; the next visitor wakes it up (this may take a few minutes), and results that were not downloaded may be lost. For large or routine analyses, please use the free **desktop version** (the same app on your own computer; images stay local): [download and instructions](https://github.com/MichaWehrli/collembot_huggingface/tree/main/desktop). Questions or problems: [micha.wehrli@eawag.ch](mailto:micha.wehrli@eawag.ch).

COLLEMBOT is a free, open-source tool for automated counting and body-size measurement of *Folsomia candida* in OECD 232 soil reproduction toxicity tests. It uses a YOLOv11x-seg instance segmentation model trained on images from six European laboratories.

Published in *Environmental Toxicology and Chemistry* (2026) · DOI: [10.1093/etojnl/vgag068](https://doi.org/10.1093/etojnl/vgag068)

---

## Tabs

| Tab | What it does |
|-----|-------------|
| **Compare groups** | Batch analysis with group labels → KDE distributions, stacked adult/juvenile bar chart, egg clutch panel, quality-flagged summary table, gallery. Automatically switches between per-image and per-group (mean ± SD) view depending on dataset size. **Download all results (ZIP):** every image annotated at full resolution (outline + ID matching the CSV, length axis, dish circle, footer with settings), the CSV and a `run_settings.json` |
| **Hand correction** | Click to remove false positives or add missed organisms. Sub-threshold candidates shown as grey outlines for promotion. Full timestamped audit log exported |
| **🎯 Calibrate threshold** | Upload 1–3 images with known counts (high/mid/zero) → sweep 15 thresholds → find optimal confidence threshold for your imaging setup → apply to all tabs in one click |

---

## Size classification

When px/mm scale is provided, biology-based DEB thresholds are used (*F. candida* at 20°C, Hamda 2014):

| Class | Criterion |
|-------|-----------|
| Adult | ≥ 1.2 mm |
| Uncertain | 1.0–1.2 mm |
| Juvenile | < 1.0 mm |

Without a scale, **Compare groups reports no adult/juvenile classes** (counts and pixel lengths only), because the statistical fallback (2-component GMM) tends to split egg clutches rather than life stages. The Hand correction tab still shows a GMM-based size column; treat it as indicative only.

**Egg clutch detection** — when scale is provided and ≥8 juveniles detected, a BIC-selected multi-component GMM identifies distinct hatching cohorts in the juvenile length distribution.

---

## Quality flags

| Flag | Trigger |
|------|---------|
| `grid` | Grid divergence >20% between ori/shift tile grids |
| `conf` | Mean detection confidence <0.55 |
| `fusion` | Fusion ratio outside [0.45, 0.82] |
| `count` | Count <10 |
| `circle` | Dish boundary detection failed |
| `Grubbs` | Statistical outlier within replicate group (α=0.05) |
| `adults>10` 🔴 | >10 adults classified (OECD 232: exactly 10 added); only evaluated when a px/mm scale is entered |

---

## Confidence threshold

Default: **0.732**. Each tab has an ⚙️ Advanced settings accordion with a slider (0.10–0.80). The value is saved in every exported CSV and JSON. Use the **🎯 Calibrate threshold** tab to find the optimal value for your imaging setup — particularly important for labs not represented in the training data.

Validation against manual counts (OECD 232 Test, n=46 replicates): th=0.732 overcounts by +23%, th=0.80 reduces this to −3% with MAE dropping from 79.5 to 31.7. EC50 agreement is <1% difference between methods at both thresholds.

---

## Image scale (px/mm)

Enter the scale manually (automatic ruler detection was removed in v0.1.1 because it was unreliable). Measure once per camera setup in ImageJ/Fiji: draw a straight line between two ruler marks a known distance apart (10 mm or more for precision), Analyze → Measure, then divide the length in pixels by the distance in mm (e.g. 480 px over 10 mm = 48 px/mm). Keep camera height and zoom fixed so the scale stays valid for all images.

---

## How to use

**Quick start:** Compare groups tab → upload one or more images → enter px/mm → run comparison → download CSV or the full ZIP

**Full experiment:** Compare groups tab → upload all images → enter group labels (`control, control, treatment_10, treatment_10`) → enter px/mm → run comparison → review flags → download the ZIP and check the full-resolution images

**New imaging setup:** Calibrate threshold tab first → upload 3 images (control ~800, mid-dose ~300, high-dose ~0) → run sweep → apply → then run full experiment

---

## Model

- Architecture: YOLOv11x-seg (Ultralytics, AGPL-3.0)
- Training data: annotated tiles from 6 labs (Basel, Amsterdam, Bayreuth, Coimbra, Aarhus, Cairo) — CC-BY-4.0
- Weights: [Zenodo DOI: 10.5281/zenodo.17987886](https://doi.org/10.5281/zenodo.17987886)
- Inference: tiled 576×576 px, 50% overlap, graph-cut polygon fusion (detections with IoU > 0.101 merged, as in the base repository)

---

## Citation

> Wehrli M\*, Meyer A\*, Souza da Silva É\*, van Loon S, van Hall BG, van Gestel CAM, Natal-da-Luz T, Döring MVR, Feldhaar H, Mair M, Jordan D, Langer M (2026). COLLEMBOT: AI-based counting of Collembola for OECD 232 tests. *Environmental Toxicology and Chemistry*. DOI: 10.1093/etojnl/vgag068

---

## Links

- 📄 [Paper](https://doi.org/10.1093/etojnl/vgag068) · 💾 [Model weights](https://doi.org/10.5281/zenodo.17987886) · 🐙 [Source code](https://github.com/waldstrom/collembot)

**License:** AGPL-3.0 (code & weights) · CC-BY-4.0 (training data)