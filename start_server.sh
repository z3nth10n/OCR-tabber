#!/bin/bash

cd src
../.venv/bin/python -m uvicorn json2tab_api:app --port 9999 --reload
