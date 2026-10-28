@echo off
rem CollemBot desktop for Windows: double-click this file to start CollemBot.
rem First start: installs the Python packages (about 3-5 GB) and downloads the model.
rem Later starts are much faster.
rem Everything is installed in %LOCALAPPDATA%\CollemBot (delete that folder to uninstall).
rem AI-assisted: Claude (Opus 5.5), 2026-10-28; reviewed by: <pending>.

setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not defined COLLEMBOT_HOME set "COLLEMBOT_HOME=%LOCALAPPDATA%\CollemBot"
set "VENV=%COLLEMBOT_HOME%\venv"
set "VPY=%VENV%\Scripts\python.exe"

if not exist "%VPY%" goto :create_venv
if not exist "%VENV%\torch-installed.txt" goto :install_torch
goto :check_packages

:create_venv
echo.
echo === CollemBot desktop: first start ===
echo Setting up CollemBot in %COLLEMBOT_HOME%
echo This can take 10-30 minutes and is done only once. Please keep this window open.
echo.
rem Python 3.12, 3.11 or 3.10 is needed (PyTorch 2.2 does not support Python 3.13 or newer).
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY goto :no_python
echo Using Python: %PY%
if not exist "%COLLEMBOT_HOME%" mkdir "%COLLEMBOT_HOME%"
%PY% -m venv "%VENV%"
if errorlevel 1 goto :setup_failed
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :setup_failed

:install_torch
rem PyTorch: NVIDIA (CUDA) build if an NVIDIA driver is installed, otherwise the CPU build.
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :torch_cpu
echo NVIDIA GPU found: installing PyTorch with GPU support ...
"%VPY%" -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto :torch_done
echo The GPU version could not be installed; installing the CPU version instead.
:torch_cpu
echo Installing PyTorch (CPU version) ...
"%VPY%" -m pip install torch==2.2.2 torchvision==0.17.2
if errorlevel 1 goto :setup_failed
:torch_done
echo done> "%VENV%\torch-installed.txt"

:check_packages
rem Install the other packages, and again whenever requirements-desktop.txt changes.
fc /b "requirements-desktop.txt" "%VENV%\requirements-installed.txt" >nul 2>nul
if not errorlevel 1 goto :run
echo Installing CollemBot packages ...
"%VPY%" -m pip install -r requirements-desktop.txt
if errorlevel 1 goto :setup_failed
copy /y "requirements-desktop.txt" "%VENV%\requirements-installed.txt" >nul

:run
"%VPY%" collembot_desktop.py %*
if errorlevel 1 goto :run_failed
endlocal
exit /b 0

:no_python
echo.
echo Python 3.10, 3.11 or 3.12 was not found on this computer.
echo Please install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" in the installer. Then start this file again.
pause
exit /b 1

:setup_failed
echo.
echo Setup failed - see the messages above (internet connection? disk space?).
echo Start this file again to retry. To start from scratch, delete the folder
echo %VENV%
pause
exit /b 1

:run_failed
echo.
echo CollemBot stopped with an error - see the messages above.
pause
exit /b 1
