#!/bin/bash

cd "$(dirname "$0")/.."

pip freeze > requirements.txt
