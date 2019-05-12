#!/bin/bash

cd "$(dirname "$0")/.."

FLASK_APP=src/server.py flask run --host=0.0.0.0 -p 8085
