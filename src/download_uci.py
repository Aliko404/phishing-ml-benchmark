"""Download actual UCI Phishing Websites dataset #327; no generated or synthetic observations."""
from __future__ import annotations
import argparse, json, hashlib, platform
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='data/external/uci327_phishing_websites.csv')
    args=ap.parse_args()
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit('Install dependencies: pip install -r requirements.txt') from exc
    item=fetch_ucirepo(id=327)
    X=item.data.features.copy()
    y=item.data.targets.iloc[:,0].copy()
    if len(X)!=11055: raise RuntimeError(f'Unexpected UCI dataset size: {len(X)}; expected 11055')
    if X.shape[1]!=30: raise RuntimeError(f'Unexpected feature count: {X.shape[1]}; expected 30')
    classes=set(pd.to_numeric(y).dropna().unique())
    if classes!={-1,1}:raise RuntimeError(f'Unexpected target classes: {classes}')
    numeric=X.apply(pd.to_numeric,errors='raise')
    if numeric.isna().any().any(): raise RuntimeError('Missing feature values; no imputation configured')
    # UCI -1 means phishing; our project label 1 means phishing.
    numeric['label']=(pd.to_numeric(y)==-1).astype(int).to_numpy()
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    numeric.to_csv(dest,index=False)
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    meta={'dataset':'UCI Phishing Websites','uci_id':327,
          'source_url':'https://archive.ics.uci.edu/dataset/327/phishing+websites',
          'citation':'https://doi.org/10.24432/C51W2X',
          'rows':len(numeric),'features':X.shape[1],
          'labels':{'0':'legitimate','1':'phishing'},
          'label_mapping':{'UCI -1':'phishing -> 1','UCI +1':'legitimate -> 0'},
          'dataset_csv_sha256':digest,'python':platform.python_version(),
          'note':'Original source records; no synthetic observations. CSV is locally re-encoded.'}
    (dest.parent/'uci327_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'Downloaded {len(numeric)} real website observations, {X.shape[1]} features -> {dest}')
    print('CSV SHA256:',digest)
if __name__=='__main__':main()
