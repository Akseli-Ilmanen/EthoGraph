@echo off
setlocal enabledelayedexpansion
set ENV_NAME=ethograph

echo ============================================
echo ethograph Installation Script
echo ============================================
echo.



:: Check if conda is available
where conda >nul 2>nulconda
if %errorlevel% neq 0 (
    echo ERROR: Conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://www.anaconda.com/download
    pause
    exit /b 1
)


:: Ask user which environment to install
echo Choose installation type:
echo   1. ethograph GUI 
echo   2. ethograph GUI + Tranformer segmentation model (in development)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    set ENV_FILE=environment.yml
) else if "%choice%"=="2" (
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
call conda env list | findstr /B /C:"%ENV_NAME% " >nul 2>nul
if %errorlevel% equ 0 (
    echo Environment %ENV_NAME% already exists.
    set /p overwrite="Do you want to remove and reinstall it? (y/n): "
    if /i "!overwrite!"=="y" (
        echo Removing existing environment...
        call conda env remove -n %ENV_NAME% -yn
    ) else (
        goto :finish
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

echo.
echo Environment created successfully!

echo.
set /p create_shortcut="Do you want to create a desktop shortcut for the GUI? (y/n): "
if /i "!create_shortcut!"=="y" (
    echo Creating desktop shortcut...
    call conda activate %ENV_NAME%
    call ethograph-shortcut
    if %errorlevel% neq 0 (
        echo WARNING: Failed to create desktop shortcut.
    )
) else (
    echo Skipping desktop shortcut creation.
)

echo.

:finish









