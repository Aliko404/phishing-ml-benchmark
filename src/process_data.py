"""Process inference telemetry, error rates, descriptive statistics and bootstrap confidence intervals."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"phase", "model_variant", "batch_size", "latency_ms", "status_code"}

def read_telemetry(path: str | Path) -> pd.DataFrame:
    rows=[]
    with open(path,encoding="utf-8") as source:
        for n,line in enumerate(source,1):
            if line.strip():
                try: rec=json.loads(line)
                except json.JSONDecodeError as exc: raise ValueError(f"Malformed JSON on line {n}") from exc
                if not isinstance(rec,dict): raise ValueError(f"Line {n} must contain an object")
                rows.append(rec)
    if not rows: raise ValueError(f"Empty telemetry: {path}")
    frame=pd.DataFrame(rows)
    absent=REQUIRED-set(frame.columns)
    if absent: raise ValueError(f"Missing fields {sorted(absent)} in {path}; old constant-latency dump cannot be used")
    if frame.phase.isna().any() or not frame.phase.isin(("warmup","benchmark")).all():
        raise ValueError("Each row must be marked warmup or benchmark")
    if frame.model_variant.isna().any(): raise ValueError("Missing model names")
    frame['batch_size']=pd.to_numeric(frame['batch_size'],errors='raise').astype(int)
    if (frame.batch_size<=0).any(): raise ValueError("Batch size must be positive")
    frame['status_code']=pd.to_numeric(frame.status_code,errors='raise').astype(int)
    frame['latency_ms']=pd.to_numeric(frame.latency_ms,errors='coerce')
    return frame

def load_and_clean_data(filepath: str | Path) -> pd.DataFrame:
    """Drop warmup rows only; retain errors as guardrail events (not latency samples)."""
    return read_telemetry(filepath).loc[lambda x:x.phase.eq('benchmark')].copy()

def calculate_statistics(df: pd.DataFrame, seed: int=42, n_boot: int=2000) -> pd.DataFrame:
    if df.empty: raise ValueError('No benchmark rows after warmup exclusion')
    if n_boot < 100: raise ValueError('At least 100 bootstrap replicates needed')
    rng=np.random.default_rng(seed)
    records=[]
    for (model,batch), group in df.groupby(['model_variant','batch_size'],sort=True):
        ok=group.status_code.eq(200)&group.latency_ms.notna()&np.isfinite(group.latency_ms)&(group.latency_ms>=0)
        vals=group.loc[ok,'latency_ms'].to_numpy(dtype=float)
        rec={'model_variant':model,'batch_size':int(batch),'n_total':int(len(group)),
             'n_success':int(len(vals)),'n_error':int(len(group)-len(vals)),
             'error_rate_pct':float(100*(len(group)-len(vals))/len(group))}
        measures=['mean_ms','sd_ms','median_ms','p25_ms','p75_ms','p95_ms','p99_ms','median_ci_lower_ms','median_ci_upper_ms']
        rec.update({field:float('nan') for field in measures})
        if len(vals):
            boots=np.median(rng.choice(vals,size=(n_boot,len(vals)),replace=True),axis=1)
            rec.update(mean_ms=float(vals.mean()),sd_ms=float(vals.std(ddof=1)) if len(vals)>1 else float('nan'),
                median_ms=float(np.median(vals)),p25_ms=float(np.percentile(vals,25)),p75_ms=float(np.percentile(vals,75)),
                p95_ms=float(np.percentile(vals,95)),p99_ms=float(np.percentile(vals,99)),
                median_ci_lower_ms=float(np.percentile(boots,2.5)),median_ci_upper_ms=float(np.percentile(boots,97.5)))
        records.append(rec)
    return pd.DataFrame.from_records(records)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='data/raw/test_run.jsonl')
    parser.add_argument('--output',default='data/processed/metrics_summary.csv')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    raw=read_telemetry(args.input)
    benchmark=raw.loc[raw.phase.eq('benchmark')].copy()
    print(f"Warm-up excluded: {(raw.phase=='warmup').sum()} | benchmark rows: {len(benchmark)}")
    stats=calculate_statistics(benchmark,seed=args.seed)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    stats.to_csv(target,index=False)
    print(stats.to_string(index=False));print('Saved:',target)
if __name__=='__main__':main()
