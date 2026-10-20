"""
CollemBot — HuggingFace Spaces entry point
==========================================
Downloads or updates the YOLO model weights from Zenodo on boot.

Update check logic:
  - If weights don't exist → download immediately.
  - If weights exist → compare local mtime against the remote
    Last-Modified header from Zenodo. Re-download only if the
    remote file is newer. Falls back to using the cached file
    if the network check fails (offline / Zenodo unreachable).

ZeroGPU note: the @spaces.GPU decorator is applied to
infer_image inside gradio_app.py at import time.
"""

import importlib.util
import sys
import time
import urllib.request
import urllib.error
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT       = Path(__file__).parent
MODEL_DIR  = ROOT / "model"
MODEL_FILE = MODEL_DIR / "best.pt"
META_FILE  = MODEL_DIR / "best.pt.meta"   # stores remote Last-Modified timestamp
GRADIO_APP = ROOT / "gradio_app.py"

ZENODO_URL = (
    "https://zenodo.org/records/17987887/files/"
    "collembot_yolo11xseg_v_0_1_0.pt?download=1"
)

# How often to re-check for updates even if weights are present.
# Set to 0 to check on every cold start (adds ~1s HEAD request).
UPDATE_CHECK_INTERVAL_HOURS = 24


def _remote_last_modified() -> float | None:
    """Return the remote Last-Modified timestamp as a Unix float, or None."""
    try:
        req = urllib.request.Request(ZENODO_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            lm = resp.headers.get("Last-Modified")
            if lm:
                return parsedate_to_datetime(lm).timestamp()
    except Exception as e:
        print(f"[CollemBot] Update check failed (will use cached weights): {e}")
    return None


def _cached_remote_mtime() -> float | None:
    """Read the previously stored remote mtime from the meta file."""
    if META_FILE.exists():
        try:
            return float(META_FILE.read_text().strip())
        except ValueError:
            pass
    return None


def _save_remote_mtime(ts: float):
    META_FILE.write_text(str(ts))


def _last_check_age_hours() -> float:
    """Hours since the meta file was last written (proxy for last check time)."""
    if META_FILE.exists():
        return (time.time() - META_FILE.stat().st_mtime) / 3600
    return float("inf")


ZENODO_FILENAME = "collembot_yolo11xseg_v_0_1_0.pt"

def download_weights():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Check for locally uploaded weights first ──────────────────────────────
    # If the file was uploaded to the Space under the Zenodo filename,
    # copy it to best.pt so we don't need to download anything.
    local_zenodo = ROOT / ZENODO_FILENAME
    local_model_dir = MODEL_DIR / ZENODO_FILENAME
    for candidate in [local_zenodo, local_model_dir]:
        if candidate.exists() and not MODEL_FILE.exists():
            print(f"[CollemBot] Found local weights at {candidate} — copying to {MODEL_FILE}")
            import shutil
            shutil.copy2(candidate, MODEL_FILE)
            _save_remote_mtime(time.time())
            return

    # ── Case 1: no local weights at all ──────────────────────────────────────
    if not MODEL_FILE.exists():
        print("[CollemBot] No local weights found — downloading from Zenodo …")
        _do_download()
        return

    # ── Case 2: weights exist, check if update check is due ──────────────────
    if _last_check_age_hours() < UPDATE_CHECK_INTERVAL_HOURS:
        age_h = _last_check_age_hours()
        print(f"[CollemBot] Weights present, last checked {age_h:.1f}h ago "
              f"(threshold {UPDATE_CHECK_INTERVAL_HOURS}h) — skipping update check.")
        return

    # ── Case 3: weights exist, update check is due ────────────────────────────
    print("[CollemBot] Checking Zenodo for updated weights …")
    remote_ts = _remote_last_modified()

    if remote_ts is None:
        # Network unavailable — use what we have
        print("[CollemBot] Could not reach Zenodo — using cached weights.")
        return

    cached_ts = _cached_remote_mtime()

    if cached_ts is not None and remote_ts <= cached_ts:
        print(f"[CollemBot] Weights are up to date (remote: "
              f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(remote_ts))}).")
        _save_remote_mtime(remote_ts)   # refresh check timestamp
        return

    # Remote is newer (or we've never stored the remote mtime)
    if cached_ts is not None:
        print(f"[CollemBot] New weights available on Zenodo "
              f"(remote: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(remote_ts))} "
              f"> local record: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(cached_ts))}) "
              f"— re-downloading …")
    else:
        print("[CollemBot] No previous version record — downloading weights …")

    _do_download(remote_ts)


def _do_download(remote_ts: float | None = None):
    """Perform the actual download and update the meta file."""
    tmp = MODEL_FILE.with_suffix(".pt.tmp")
    try:
        urllib.request.urlretrieve(ZENODO_URL, tmp)
        tmp.replace(MODEL_FILE)
        size_mb = MODEL_FILE.stat().st_size / 1e6
        print(f"[CollemBot] Weights saved ({size_mb:.1f} MB) → {MODEL_FILE}")
        # Record the remote mtime we just downloaded
        ts = remote_ts or time.time()
        _save_remote_mtime(ts)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        if MODEL_FILE.exists():
            print(f"[CollemBot] Download failed ({e}) — using existing cached weights.")
        else:
            raise RuntimeError(f"[CollemBot] Download failed and no cached weights: {e}")


def load_gradio_app():
    spec   = importlib.util.spec_from_file_location("gradio_app", GRADIO_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gradio_app"] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    download_weights()
    sys.path.insert(0, str(ROOT))
    gradio_app = load_gradio_app()
    gradio_app.launch_app(
        model_path=str(MODEL_FILE),
        fuse=False,
        preferred_device="auto",
        predict_batch_size=8,
        use_half=False,
    )