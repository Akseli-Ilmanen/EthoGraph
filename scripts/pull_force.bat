@echo off
set "PROJECT_PATH=E:\Your\Path\Here\ethograph"
E:
cd "%PROJECT_PATH%"
git fetch --all
git reset --hard origin/main
pip install -e .