"""Entry point: python -m src.main --config configs/config.yaml"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from .benchmark import run_benchmark


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phishing detection ML benchmark")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", default=None, help="Override data.path from config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.data:
        cfg["data"]["path"] = args.data

    setup_logging(cfg["experiment"]["log_file"])
    log = logging.getLogger("main")
    np.random.seed(cfg["experiment"]["seed"])
    log.info("Environment: python=%s sklearn=%s pandas=%s numpy=%s on %s",
             platform.python_version(), sklearn.__version__, pd.__version__,
             np.__version__, platform.platform())

    df = run_benchmark(cfg)

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "results.csv"
    df.to_csv(out, index=False)
    log.info("Saved %d rows to %s", len(df), out)
    print(df[["algorithm", "feature_selection", "accuracy", "f1",
              "inference_ms_per_sample", "guardrails_pass"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
