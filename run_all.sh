#!/usr/bin/env bash
# Reproduces every number, table and figure of the paper.
# Quick smoke test:  FAST=1 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figs
echo "== climapoll.py: main experiment, 5 replicates x 6 seasons"
python climapoll.py
echo "== frontier.py: risk-return frontier"
python frontier.py
echo "== extra.py: budget and component studies"
python extra.py
echo "== windcal.py: wind-constraint reliability"
python windcal.py
echo "== ts.py: heatwave time-series example"
python ts.py
echo "== ensize.py: ensemble-size study"
python ensize.py
echo "== analyze.py: statistics, tables and figures"
python analyze.py
echo "== ensize_an.py: ensemble-size figure"
python ensize_an.py
echo "== fig_arch.py: architecture figure"
python fig_arch.py
echo "Done. Numbers: numbers.json, figures: figs/, raw results: results/"
