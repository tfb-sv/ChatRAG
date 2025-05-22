@echo off

set ENV_NAME=torch-lnn
set CONDA_DIR=C:\Users\nural\anaconda3
set MAIN_DIR=C:\Users\nural\gptRAG
set PYTHON_EXE=%CONDA_DIR%\envs\%ENV_NAME%\python.exe

CALL %CONDA_DIR%\Scripts\activate.bat %ENV_NAME%
cd %MAIN_DIR%

set PYTHONPATH=%MAIN_DIR%

%PYTHON_EXE% main.py

pause
