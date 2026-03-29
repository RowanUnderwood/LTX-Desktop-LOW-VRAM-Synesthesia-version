@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ========================================
echo   LTX Desktop - Install
echo ========================================
echo.

REM ---- Check Python --------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install Python 3.12+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo Found Python %PYTHON_VERSION%

REM ---- Check uv ------------------------------------------------------
where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: uv not found. Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: Failed to install uv.
        echo Install manually: https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv still not found after install. Please restart this script.
        pause
        exit /b 1
    )
)
for /f "tokens=1,2 delims= " %%a in ('uv --version') do echo Found uv %%b

REM ---- Check Node ----------------------------------------------------
set NODE_AVAILABLE=0
where node >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%v in ('node --version') do echo Found Node %%v
    set NODE_AVAILABLE=1
) else (
    echo WARNING: Node.js not found. Install from https://nodejs.org/
    echo          Skipping Node dependency install.
)

echo.
echo -- Step 1: Create Python virtual environment
cd backend

if not exist ".venv" goto :create_venv
echo Virtual environment already exists at backend\.venv
set /p RECREATE=Recreate it? [y/N]:
if /i "!RECREATE!"=="y" (
    echo Removing existing venv...
    rmdir /s /q .venv
    goto :create_venv
)
echo Keeping existing venv.
goto :sync_deps

:create_venv
echo Fetching Python 3.13 (project requires 3.12+)...
uv python install 3.13
if errorlevel 1 (
    echo ERROR: Failed to install required Python version.
    pause
    exit /b 1
)
echo.
echo Creating venv at backend\.venv ...
uv venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
echo Virtual environment created.

:sync_deps
echo.
echo -- Step 2: Install Python dependencies
echo This may take a while on first install (PyTorch is large)...
echo.
uv sync --frozen
if errorlevel 1 (
    echo NOTE: --frozen failed, retrying without lock enforcement...
    uv sync
    if errorlevel 1 (
        echo ERROR: Failed to install Python dependencies.
        pause
        exit /b 1
    )
)
echo Python dependencies installed.

echo.
echo -- Step 3: Verify PyTorch / CUDA
.venv\Scripts\python.exe -c "import torch; cuda=torch.cuda.is_available(); print('PyTorch:', torch.__version__); print('CUDA:', cuda); [print('GPU %d:' %% i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())] if cuda else print('WARNING: No CUDA GPU detected')" 2>nul
if errorlevel 1 (
    echo Could not verify PyTorch - this is OK if install is still in progress.
)

cd ..

echo.
echo -- Step 4: Install Node dependencies
if "!NODE_AVAILABLE!"=="0" goto :skip_node

where pnpm >nul 2>&1
if not errorlevel 1 (
    pnpm install
    goto :node_done
)
npm install

:node_done
if errorlevel 1 (
    echo ERROR: Node dependency install failed.
    pause
    exit /b 1
)
echo Node dependencies installed.
goto :check_ffmpeg

:skip_node
echo Skipping (Node.js not found).

:check_ffmpeg
echo.
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo NOTE: ffmpeg not found. Install with: winget install ffmpeg
    echo       (imageio-ffmpeg bundled binary will be used as fallback)
) else (
    echo Found ffmpeg.
)

echo.
echo ========================================
echo   Install complete!
echo.
echo   Run the app:  launch.bat
echo   Dev mode:     pnpm run dev
echo ========================================
echo.
pause
