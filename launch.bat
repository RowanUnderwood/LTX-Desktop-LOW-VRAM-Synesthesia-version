@echo off
cd /d "%~dp0"
set ELECTRON_LOAD_DIST=1
npx electron .
