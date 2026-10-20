# CollemBot v0.1.1 — changelog

Date: 2026-10-20 · Base: Hugging Face Space `MichaWehrli/COLLEMBOT`, commit d23dc86 (`gradio_app.py` sha256 `c2519d0f…9f15`, as supplied by Micha)
Patched `gradio_app.py` sha256: see `test_report_v0.1.1.md` · Code changes AI-assisted (Claude Opus 5.5), to be reviewed by the maintainer before deployment.

**Detection, tiling, fusion, confidence threshold and all CSV columns/values are unchanged.** This release changes the interface only.

## Changes (decided by Micha, 2026-10-20)

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
