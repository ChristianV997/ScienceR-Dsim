#!/usr/bin/env bash
set -euo pipefail

python main.py --mode external --config config/defaults.yaml --output results/live_sensors.csv --db data/runs.sqlite "$@"
