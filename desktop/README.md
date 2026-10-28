# 🪲 CollemBot desktop (Windows and macOS)

The CollemBot app on your own computer: the same app as the [online version](https://huggingface.co/spaces/MichaWehrli/COLLEMBOT), with counting, body length and adult/juvenile classes (with a px/mm scale), hand correction, threshold calibration and the full-resolution results ZIP.

- **Your images stay on your computer.** Nothing is uploaded.
- **Free and without limits.** Use it for large or routine analyses; the online version runs on a GPU that we pay for per hour.
- ⚠️ Beta: check counts and lengths for your own imaging setup before using them in a publication (Hand correction and Calibrate threshold tabs).

## What you need
| | Windows | macOS |
|---|---|---|
| System | Windows 10 or 11 (64-bit) | macOS 13 (Ventura) or newer; Apple-silicon (M-chip) or Intel Mac |
| Python | **Python 3.12 (64-bit)** from [python.org](https://www.python.org/downloads/windows/); in the installer tick **"Add python.exe to PATH"** | **Python 3.12** from [python.org](https://www.python.org/downloads/macos/) |
| Faster with | an NVIDIA graphics card | an Apple-silicon Mac (uses the Mac's GPU) |

- Python 3.10 and 3.11 also work; 3.13 or newer does not (not yet supported by the PyTorch version used).
- About 5 GB of free disk space and an internet connection for the first start.
- Without a supported GPU, CollemBot runs on the processor (CPU): it works, but each image takes longer.

## Install and start
1. On the [GitHub page of this repository](https://github.com/MichaWehrli/collembot_huggingface), click **Code → Download ZIP** and unzip it. Keep the folders `desktop` and `huggingface_space` together.
2. Open the folder `desktop` and double-click the start file:
   - **Windows:** `start_collembot.bat`. If Windows shows "Windows protected your PC", click **More info → Run anyway**.
   - **macOS:** `start_collembot.command` (it opens in the Terminal). If macOS says the file cannot be opened or could not be verified: open **System Settings → Privacy & Security**, scroll down to **Security** and click **Open Anyway** (shown for about an hour after the attempt), then enter your password.
     Alternative: open the Terminal app, type `bash ` (with a space), drag `start_collembot.command` into the Terminal window and press Enter.
3. **First start only:** CollemBot installs its Python packages and downloads the model from [Zenodo](https://doi.org/10.5281/zenodo.17987886). Depending on your internet connection this can take 10–30 minutes. Later starts are much faster.
4. CollemBot opens in your web browser (address `http://127.0.0.1:7860`). Keep the black window (Windows) or the Terminal window (macOS) open while you work; close it to stop CollemBot.

Python packages and model are installed outside the download folder:
- Windows: `%LOCALAPPDATA%\CollemBot`
- macOS: `~/Library/Application Support/CollemBot`

## Results
The same as online: CSV download and **Download all results (ZIP)** with every image annotated at full resolution, the CSV and `run_settings.json`. Files are saved to your browser's download folder. `run_settings.json` records that the run was made with the desktop version (`"runtime": "desktop"`) and on which device (`cpu`, `cuda:0` for NVIDIA, `mps` for an Apple-silicon GPU).

## Same results as online?
The desktop version uses the same app code, the same model and the same settings as the online version (default confidence threshold 0.732, 576 px tiles, IoU 0.101), with the same versions of the main packages. GPUs (NVIDIA or Apple) and CPUs calculate with slightly different rounding, so in rare cases a detection very close to the threshold can come out differently.

## Update
Download the new ZIP and start the start file from the new folder. Packages are updated automatically when needed; the model is kept.

## Uninstall
Delete the CollemBot download folder and the installation folder:
- Windows: `%LOCALAPPDATA%\CollemBot` (type `%LOCALAPPDATA%` into the address bar of the File Explorer).
- macOS: `~/Library/Application Support/CollemBot` (in the Finder: **Go → Go to Folder…**).

## Problems?
- **"Python 3.10, 3.11 or 3.12 was not found":** install Python 3.12 as described above, then start the start file again.
- **Setup failed:** usually the internet connection, a company proxy or disk space. Start again to retry. To start from scratch, delete the folder `venv` inside the installation folder (see above).
- **Model download fails** (e.g. firewall): download the model in your browser from [Zenodo](https://zenodo.org/records/17987887/files/collembot_yolo11xseg_v_0_1_0.pt?download=1) and save it as `collembot_yolo11xseg_v_0_1_0.pt` in the folder `model` inside the installation folder.
- **macOS: the start file does nothing or "permission denied":** use the Terminal alternative in step 2.
- **Port already in use:** CollemBot then uses the next free port and opens the right address itself.
- **Options:** start from a command window / Terminal with options after the start file, e.g. `start_collembot.bat --cpu` or `bash start_collembot.command --cpu`. Options: `--cpu` (do not use the GPU), `--port 7861`, `--no-browser`, `--model PATH` (use a model file stored elsewhere).
- Still stuck? Contact [micha.wehrli@eawag.ch](mailto:micha.wehrli@eawag.ch).

## Alternative: Docker (Windows, macOS, Linux)
For users who already work with Docker. This runs the exact container of the online version.
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. On the [CollemBot Space page](https://huggingface.co/spaces/MichaWehrli/COLLEMBOT), open **Run with Docker** (top right) and copy the command. Add `-e COLLEMBOT_TITLE_PAGE=0` to hide the cost notice of the online version, and use `-p 127.0.0.1:7860:7860` so the app is reachable from your computer only. The command then looks like this:
   ```
   docker run -it -p 127.0.0.1:7860:7860 --platform=linux/amd64 -e COLLEMBOT_TITLE_PAGE=0 registry.hf.space/michawehrli-collembot:latest python app.py
   ```
3. Open `http://127.0.0.1:7860` in your browser.

With an NVIDIA GPU (Linux, or Windows with WSL 2 and the NVIDIA Container Toolkit), add `--gpus all`; otherwise the container runs on the CPU (on Apple-silicon Macs under emulation, which is slow). The container downloads the model each time it starts.

## Citation
If CollemBot is useful for your work, please cite [doi:10.1093/etojnl/vgag068](https://doi.org/10.1093/etojnl/vgag068).
