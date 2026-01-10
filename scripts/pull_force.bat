@echo off
set "PROJECT_PATH=E:\Your\Path\Here\MoveSeg"
E:
cd "%PROJECT_PATH%"
git fetch --all
git reset --hard origin/main
pip install -e .