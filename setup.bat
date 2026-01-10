@echo off
setlocal enabledelayedexpansion

echo ============================================
echo MoveSeg Installation Script
echo ============================================
echo.

:: Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Ask user which environment to install
echo Choose installation type:
echo   1. Base environment (moveseg) - GUI only
echo   2. Developer environment (moveseg-dev) - GUI + Model training
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    set ENV_NAME=moveseg
    set ENV_FILE=environment.yml
) else if "%choice%"=="2" (
    set ENV_NAME=moveseg-dev
    set ENV_FILE=environment-dev.yml
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

echo.
echo Installing %ENV_NAME% environment from %ENV_FILE%...
echo.

:: Check if environment already exists
call conda env list | findstr /C:"%ENV_NAME%" >nul 2>nul
if %errorlevel% equ 0 (
    echo Environment %ENV_NAME% already exists.
    set /p overwrite="Do you want to remove and reinstall it? (y/n): "
    if /i "!overwrite!"=="y" (
        echo Removing existing environment...
        call conda env remove -n %ENV_NAME% -y
    ) else (
        echo Keeping existing environment. Updating instead...
        call conda env update -n %ENV_NAME% -f %ENV_FILE%
        goto :create_shortcut
    )
)

:: Create conda environment
echo Creating conda environment...
call conda env create -f %ENV_FILE%
if %errorlevel% neq 0 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

:create_shortcut
echo.
echo Creating desktop shortcut...

:: Get desktop path and create shortcut using PowerShell
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -Command ^
    "$WshShell = New-Object -ComObject WScript.Shell; ^
    $Desktop = [Environment]::GetFolderPath('Desktop'); ^
    $Shortcut = $WshShell.CreateShortcut(\"$Desktop\MoveSeg.lnk\"); ^
    $Shortcut.TargetPath = 'cmd.exe'; ^
    $Shortcut.Arguments = '/k \"conda activate %ENV_NAME% && napari\"'; ^
    $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; ^
    $Shortcut.Description = 'Launch MoveSeg GUI'; ^
    $Shortcut.Save()"

if %errorlevel% equ 0 (
    echo Desktop shortcut created: MoveSeg.lnk
) else (
    echo Warning: Could not create desktop shortcut.
)

echo.
echo ============================================
echo Installation complete!
echo ============================================
echo.
echo To use MoveSeg:
echo   1. Open Anaconda Prompt
echo   2. Run: conda activate %ENV_NAME%
echo   3. Run: napari
echo.
echo Or use the desktop shortcut "MoveSeg"
echo.
pause
