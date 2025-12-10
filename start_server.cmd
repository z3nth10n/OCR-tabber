@echo off

cd src
..\.venv\Scripts\python.exe -m uvicorn json2tab_api:app --port 9999 --reload