#!/bin/bash

cd "$(dirname "$0")/../src"

python server.py -m ../data/model -s ../data/segs2/patches
