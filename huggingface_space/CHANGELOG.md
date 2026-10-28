# CollemBot — changelog

## v0.2.2 (2026-10-28) — desktop version
Changes AI-assisted (Claude Opus 5.5); maintainer review pending.

- New **desktop version** (`desktop/` folder of the repository): the same `gradio_app.py` runs on the user's own computer (double-click launchers `start_collembot.bat` for Windows and `start_collembot.command` for macOS 13+; Docker alternative), with length measurement, hand correction, calibration and ZIP export. See `desktop/README.md`.
- `launch_app` has two new options: `title_page` (default: shown; off with `title_page=False` or environment variable `COLLEMBOT_TITLE_PAGE=0`) and `inbrowser`. Without them the Space behaves as in v0.2.1.
- `run_settings.json` now also records `runtime` (`huggingface-space`, `desktop` or `local`) and `device` (e.g. `cuda:0`, `cpu`).
- Title page, reminder and README now point to the desktop version for large or routine analyses.
- Detection, tiling, merging, threshold and CSV contents are unchanged.

## v0.2.1 (2026-10-28) — interface text only
Changes AI-assisted (Claude Opus 5.5); maintainer review pending.

- New title page: before the tabs appear, visitors see a short notice that the online GPU is paid for by the developers per hour, that usage has increased sharply, and how to use the Space considerately (run only what is needed, test on 2–3 images first, download results straight away, run large or routine analyses locally from GitHub), that the app sleeps after 15 minutes without use, and a contact address for help, next to the CollemBot title graphic (`collembot_title.jpg`, 1200 px wide; page works without it). The tabs open after "I understand — continue to CollemBot"; a one-line reminder then stays under the title.
- README: same notice near the top.
- Detection, tiling, merging, threshold and CSV/ZIP contents are unchanged; `run_settings.json` now records app version 0.2.1.

---

# CollemBot v0.2.0 — changelog

Date: 2026-10-20 · Changes AI-assisted (Claude Opus 5.5); maintainer review and validation pending.

**This release changes counts.** Validate before using v0.2.0 counts in a publication (see "Validation" below).

## What changed
Overlapping detections from the two tile grids are now merged **exactly as in the base pipeline** used for the paper (`waldstrom/collembot`, `scripts/run_inference.py`, commit a437598):

1. Two detections are merged when their polygon IoU > 0.101 (`fuse_graphcut`). `polygon_iou` and `safe_polygon` are copied verbatim from the base file. Only intersecting pairs are tested (STRtree pre-filter); non-intersecting pairs have IoU 0, so the result equals the base all-pairs loop.
2. Removed the v0.1 merge code: its IoU step never ran with Shapely ≥ 2.0 (`STRtree.query_bulk` no longer exists, so the check silently returned nothing), and its centroid-distance step is not part of the base pipeline. This was the main cause of multi-counting at tile edges.
3. Tile polygons are cleaned with the base `safe_polygon`.
4. Detections are filtered by the dish circle before merging only, as in the base pipeline (v0.1 filtered again after merging).
5. The app stops with a clear error if Shapely < 2.0; Dockerfile requires `shapely>=2.0`.

Unchanged: model, tiling (576 px, two grids), confidence threshold (default 0.732 from `configs/inference.yaml`), IoU threshold 0.101, interface, CSV/ZIP format.

## Base-pipeline steps deliberately not copied
- **Tile skipping** (`skip_bright`, `skip_blue` in the base config): on an external test set (4 images), 33–34 of 52 tiles per image would be skipped as "blue", including nearly all tiles containing animals.
- **Full-resolution dish detection**: ~52 s per 12 MP image in testing (CPU), and on the same test images it returned a circle larger than the jar. The Space keeps its downscaled version (same algorithm on a 1600 px copy). Known difference: on 2 of 4 test images the downscaled version picked a smaller circle (r ≈ 1210 vs 1470 px); no animal was affected there, but animals near the dish edge could be.

## Tests (no YOLO model in the test environment)
- Equivalence: v0.2.0 merging = base `fuse_graphcut` in 74/74 cases (simulated tile detections and random overlapping polygons): same objects, order, confidence and outlines.
- End-to-end on 4 external test images (4272×2848) with a stand-in detector that returns cut-off pieces at tile edges like YOLO: total 318 objects vs 327 true blobs (v0.1.1: 402).
- Interface, CSV and ZIP identical to v0.1.1 for identical detections.
- Caveat: when distinct animals' outlines overlap strongly, IoU > 0.101 merges them (inherent to the published method). In simulations with heavily overlapping animals this undercounted; real-image performance must come from validation.

## Validation (to do on the Space)
1. The 4 external test images (14 animals each); v0.1.1 gave 25/25/19/18.
2. Difenoconazole set (46 replicates, manual counts): counts, error and EC50 vs manual and vs v0.1 results.
3. If time allows, the paper's test images (manual annotations exist).
4. Re-check the threshold: 0.80 performed best with v0.1, possibly compensating for the edge overcount; with base merging, the paper's 0.732 may be right again (to be tested, not assumed).

## Known issues (unchanged)
- Tiles do not cover the whole image (e.g. 15.5 % of a 4272×2848 image), same as the base pipeline.
- Calibration sweep error metric; unpinned dependencies (except Shapely); Docker build clones `waldstrom/collembot` main.
- Dockerfile: `gradio>=4.0` is not quoted, so the shell reads `>` as a redirect and Gradio is effectively unpinned.

---

# CollemBot v0.1.1 — changelog

Date: 2026-10-20 · Base: Hugging Face Space commit d23dc86 (`gradio_app.py` sha256 `c2519d0f…9f15`)
Patched `gradio_app.py` sha256: see `test_report_v0.1.1.md` · Code changes AI-assisted (Claude Opus 5.5), to be reviewed by the maintainer before deployment.

**Detection, tiling, fusion, confidence threshold and all CSV columns/values are unchanged.** This release changes the interface only.

## Changes (2026-10-20)

1. **Removed the Single image tab.** Compare groups handles one or more images; Hand correction and Calibrate threshold are unchanged.
2. **Removed ruler auto-detection** (function `detect_ruler_scale` and the three "📏 Auto-detect" buttons in Single image, Compare groups and Calibrate threshold). The px/mm field remains, now with an ImageJ how-to tip under it.
3. **New: "Download all results (ZIP)"** in Compare groups:
   - `annotated_images/NNN_<image>_annotated.jpg` at full original resolution, JPEG quality 95 (4:4:4), with a detection outline and **ID identical to the `id` column of the CSV**, the length axis used for `length_px`, the dish circle, and a footer strip below the image (file, group, count, review flags, threshold, scale, model and app version, UTC time). Original pixels are not covered by the footer.
   - `collembot_batch_results.csv` (same file as the CSV download)
   - `run_settings.json`: app/model version, threshold, IoU, alpha, tile size/shift, scale and its source, per-image counts and flags, software versions (python, torch, ultralytics, shapely, numpy, gradio, Pillow, OpenCV).
4. **Adult/juvenile only with a px/mm scale.** Without a scale, Compare groups shows `N/A*` for adults **and juveniles**, the bar chart shows the total count only, and the `adults>10` flag is not evaluated.
   - Note: hiding juveniles too goes one step beyond the agreed wording ("adults and the adults>10 flag"); it follows the same reasoning. Easy to revert.
   - Consequence: without a scale, fewer images are marked `needs_review` (the adults>10 flag no longer fires on GMM-based adult counts).
5. App title shows the version (`CollemBot v0.1.1`).

## Not changed (known issues for v0.2.0)

- Tile-seam multi-counting: the IoU merge step in `_build_adjacency` never runs with Shapely 2.x (`STRtree.query_bulk` absent); the paper pipeline `run_inference.py` is not affected (all-pairs IoU). See `analysis/`.
- Tiles do not cover the full image (e.g. 15.5 % of a 4272×2848 image is never analysed; images with a side < 576 px give 0 without warning).
- Calibration sweep error metric is dominated by the zero-count image; sweep stops at 0.80.
- Unpinned dependencies; the Docker build clones GitHub `waldstrom/collembot` main, whose `configs/inference.yaml` sets the default threshold to **0.732** (the 0.45 in `gradio_app.py` is overridden).
- Hand correction still shows a GMM-based size column without scale; the px/mm fields in Calibrate threshold are not used by the sweep.
- Leftover unused code (`default_img`/`examples`, three unused helpers).

## Deploying

1. In the Space's Files tab, upload `gradio_app.py` and `README.md` from this folder (replacing the current ones). Keep a copy of the old files (`../original_space_files_v0.1.0/`).
2. The upload triggers a rebuild that installs the latest versions of unpinned packages. Check the build log; `run_settings.json` of the first run records the versions actually used.
3. Smoke test: Compare groups with two images (a) without scale, (b) with scale → check table, plot, gallery, ZIP (open one JPEG at 100 % zoom).

## Documentation to update

- SOP §4 Image scale: replace "Auto-detection (recommended)" steps 1–2 by the manual ImageJ method only (text in `SOP_text_updates_v0.1.1.md`).
- SOP §5: "Default threshold is 0.45" does not match the deployed Space (0.732, see above).
- SOP §3 imaging requirements: the ruler is still needed to measure px/mm, but no longer "for auto-scale detection".
