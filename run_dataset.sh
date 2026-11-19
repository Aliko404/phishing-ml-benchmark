#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f data/external/uci327_phishing_websites.csv || ! -f data/external/uci327_provenance.json ]]; then
  python -m src.download_uci
else
  echo "Using existing UCI dataset; verify checksum in provenance manifest."
fi
python -m src.collect_inference --data data/external/uci327_phishing_websites.csv --out data/raw/test_run.jsonl --runs 120
python -m src.process_data --input data/raw/test_run.jsonl
python -m src.visualize --input data/raw/test_run.jsonl
echo 'Measured runs saved. Open notebook, Restart & Run All, then save.'
