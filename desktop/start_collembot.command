#!/bin/bash
# CollemBot desktop for macOS: double-click this file to start CollemBot (opens in Terminal).
# First start: installs the Python packages and downloads the model.
# Later starts are much faster.
# Everything is installed in ~/Library/Application Support/CollemBot (delete it to uninstall).
# AI-assisted: Claude (Opus 5.5), 2026-10-28; reviewed by: <pending>.

cd "$(dirname "$0")" || exit 1
export PYTHONUTF8=1
export COLLEMBOT_HOME="${COLLEMBOT_HOME:-$HOME/Library/Application Support/CollemBot}"
VENV="$COLLEMBOT_HOME/venv"
VPY="$VENV/bin/python"

setup_failed() {
    echo
    echo "Setup failed - see the messages above (internet connection? disk space?)."
    echo "Start this file again to retry. To start from scratch, delete the folder"
    echo "$VENV"
    exit 1
}

# Python 3.12, 3.11 or 3.10 is needed (PyTorch 2.2 does not support Python 3.13 or newer).
find_python() {
    for c in python3.12 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null 2>&1 &&
           "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1; then
            echo "$c"
            return 0
        fi
    done
    # python.org installs here, even if the PATH was not updated
    for v in 3.12 3.11 3.10; do
        p="/Library/Frameworks/Python.framework/Versions/$v/bin/python$v"
        if [ -x "$p" ]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

if [ ! -x "$VPY" ]; then
    echo
    echo "=== CollemBot desktop: first start ==="
    echo "Setting up CollemBot in $COLLEMBOT_HOME"
    echo "This can take 10-30 minutes and is done only once. Please keep this window open."
    echo
    if ! PY="$(find_python)"; then
        echo "Python 3.10, 3.11 or 3.12 was not found on this Mac."
        echo "Please install Python 3.12 from https://www.python.org/downloads/macos/"
        echo "and then start this file again."
        exit 1
    fi
    echo "Using Python: $PY"
    mkdir -p "$COLLEMBOT_HOME" || setup_failed
    "$PY" -m venv "$VENV" || setup_failed
    "$VPY" -m pip install --upgrade pip || setup_failed
fi

if [ ! -f "$VENV/torch-installed.txt" ]; then
    # On Apple-silicon Macs this PyTorch build includes GPU support (MPS).
    echo "Installing PyTorch ..."
    "$VPY" -m pip install torch==2.2.2 torchvision==0.17.2 || setup_failed
    echo "installed" > "$VENV/torch-installed.txt"
fi

# Install the other packages, and again whenever requirements-desktop.txt changes.
if ! cmp -s requirements-desktop.txt "$VENV/requirements-installed.txt"; then
    echo "Installing CollemBot packages ..."
    "$VPY" -m pip install -r requirements-desktop.txt || setup_failed
    cp requirements-desktop.txt "$VENV/requirements-installed.txt"
fi

"$VPY" collembot_desktop.py "$@"
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 130 ]; then   # 130 = stopped with Ctrl+C
    echo
    echo "CollemBot stopped with an error - see the messages above."
    exit 1
fi
