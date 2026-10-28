"""CollemBot desktop: the CollemBot app on your own computer.

Runs the same app as the online version (huggingface_space/gradio_app.py in this
repository) with the same settings, so counts and lengths match the online version.
Your images are processed on this computer and are not uploaded anywhere.

Normally started by double-clicking start_collembot.bat (Windows) or
start_collembot.command (macOS).
Manual start:  python collembot_desktop.py [--model PATH] [--cpu] [--port N] [--no-browser]

AI-assisted: Claude (Opus 5.5), 2026-10-28; reviewed by: <pending>.
"""
import argparse
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_FILE = HERE.parent / "huggingface_space" / "gradio_app.py"
CONFIG_FILE = HERE / "configs" / "inference.yaml"


def _data_dir() -> Path:
    """Where the model is kept: COLLEMBOT_HOME, else %LOCALAPPDATA%\\CollemBot (Windows),
    ~/Library/Application Support/CollemBot (macOS) or ~/.collembot (Linux). Outside the
    download folder, so it is not synced by OneDrive/iCloud and survives updates."""
    if os.environ.get("COLLEMBOT_HOME"):
        return Path(os.environ["COLLEMBOT_HOME"])
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "CollemBot"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "CollemBot"
    return Path.home() / ".collembot"


MODEL_DIR = _data_dir() / "model"
MODEL_NAME = "collembot_yolo11xseg_v_0_1_0.pt"
MODEL_FILE = MODEL_DIR / MODEL_NAME
ZENODO_URL = f"https://zenodo.org/records/17987887/files/{MODEL_NAME}?download=1"
MODEL_URL = os.environ.get("COLLEMBOT_MODEL_URL", ZENODO_URL)   # override: mirror/testing
MIN_MODEL_BYTES = 10_000_000   # anything smaller is a failed download


def _stop(message: str) -> None:
    print("\nERROR: " + message)
    sys.exit(1)


def _ssl_context():
    """Certificates from the 'certifi' package (installed with CollemBot). The python.org
    Python on macOS has no certificates until 'Install Certificates.command' is run."""
    try:
        import ssl
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _download_model() -> None:
    """Download the model weights from Zenodo once (same file as the online version)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    part = MODEL_FILE.with_suffix(".pt.part")
    print("Downloading the CollemBot model from Zenodo (once; please wait) ...")
    print("  " + MODEL_URL)
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "CollemBot-desktop"})
        with urllib.request.urlopen(request, timeout=60, context=_ssl_context()) as resp, \
                open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                size = f" of {total / 1e6:.0f}" if total else ""
                print(f"\r  {done / 1e6:6.0f}{size} MB", end="", flush=True)
    except Exception as exc:
        part.unlink(missing_ok=True)
        _stop(f"Model download failed ({exc}).\n"
              f"Download it in your browser from {ZENODO_URL}\n"
              f"and save it as: {MODEL_FILE}")
    print()
    if part.stat().st_size < MIN_MODEL_BYTES:
        part.unlink(missing_ok=True)
        _stop("The downloaded model file is too small (incomplete download). Please try again.")
    part.replace(MODEL_FILE)
    print(f"  saved: {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="CollemBot desktop")
    parser.add_argument("--model", help="use this model weights file (.pt) instead of the download")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU even if a GPU is available")
    parser.add_argument("--port", type=int, help="port for the local web page (default: first free from 7860)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = parser.parse_args()

    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        print(f"WARNING: Python {sys.version.split()[0]} is not tested; please use Python 3.10-3.12.")
    if not APP_FILE.is_file():
        _stop(f"App file not found: {APP_FILE}\n"
              "Keep the folders 'desktop' and 'huggingface_space' together, as in the download.")
    if not CONFIG_FILE.is_file():
        _stop(f"Settings file not found: {CONFIG_FILE}")

    if args.model:
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.is_file():
            _stop(f"Model file not found: {model_path}")
    else:
        if not MODEL_FILE.is_file():
            _download_model()
        model_path = MODEL_FILE

    # Same settings as the online version; local only; no title page.
    os.chdir(HERE)                                  # app reads configs/inference.yaml from here
    os.environ["COLLEMBOT_RUNTIME"] = "desktop"
    os.environ["COLLEMBOT_TITLE_PAGE"] = "0"
    os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"   # reachable from this computer only
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple GPU: CPU for missing ops
    if args.port:
        os.environ["GRADIO_SERVER_PORT"] = str(args.port)

    print("Loading CollemBot ...")
    spec = importlib.util.spec_from_file_location("gradio_app", APP_FILE)
    app = importlib.util.module_from_spec(spec)
    sys.modules["gradio_app"] = app
    spec.loader.exec_module(app)

    device = "cpu" if args.cpu else "auto"
    resolved = app._resolve_device(device)
    print(f"CollemBot desktop v{app.APP_VERSION}")
    print(f"  model:     {model_path.name}")
    print(f"  device:    {resolved}" + ("   (no GPU found: works, but slower)" if resolved == "cpu" else ""))
    print(f"  threshold: {app.BEST_CONF} (default; can be changed in the app)")
    print("The app opens in your browser. Keep this window open while you work;")
    print("close it (or press Ctrl+C) to stop CollemBot.\n")

    app.launch_app(
        model_path=str(model_path),
        fuse=False,
        preferred_device=device,
        predict_batch_size=8,
        use_half=False,
        title_page=False,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
