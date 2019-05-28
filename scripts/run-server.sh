#!/bin/bash

cd "$(dirname "$0")/.."

python src/server.py -m data/model
