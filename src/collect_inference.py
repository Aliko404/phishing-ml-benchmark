"""Collect real measured prediction times. Batch sizes are workload proxies, NOT actual RPS."""
from __future__ import annotations
import argparse,json,platform,time,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .models import build_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/external/uci327_phishing_websites.csv')
    p.add_argument('--out',default='data/raw/inference_telemetry.jsonl')
    p.add_argument('--runs',type=int,default=120)
    p.add_argument('--min-rows',type=int,default=1000)
    p.add_argument('--allow-small',action='store_true',help='Only for testing; never use as evidence')
    a=p.parse_args()
    df=pd.read_csv(a.data)
    if 'label' not in df.columns: raise ValueError('Missing label')
    X=df.drop(columns=['label']);y=df.label
    if not set(y.unique()).issubset({0,1}):raise ValueError('Expected 0/1 labels')
    if len(df)<a.min_rows and not a.allow_small: raise ValueError(f'{len(df)} records is too small for a real experiment. Download UCI dataset with python -m src.download_uci first')
    if df.isna().any().any():raise ValueError('Missing values; clean the dataset before benchmark')
    dataset_hash=hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
    train,test,ytrain,ytest=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
    variants=[('LogisticRegression','logistic_regression',{'max_iter':1000}),('RandomForest','random_forest',{'n_estimators':100,'n_jobs':1})]
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for name,algo,params in variants:
            pipe=Pipeline([('scale',StandardScaler()),('clf',build_model(algo,params,42))]);pipe.fit(train,ytrain)
            for batch in [1, 2, 4, 6, 9]:
                indices=np.resize(np.arange(len(test)),batch)
                inputs=test.iloc[indices]
                for i in range(a.runs):
                    phase='warmup' if i<max(3,a.runs//10) else 'benchmark'
                    start=time.perf_counter()
                    try:
                        pred=pipe.predict(inputs); status=200 if len(pred)==batch else 500;err=None
                    except Exception as exc:
                        status=500;err=str(exc)
                    ms=(time.perf_counter()-start)*1000
                    rec={'phase':phase,'model_variant':name,'batch_size':batch,
                         'status_code':status,'latency_ms':ms,'dataset':str(a.data),'seed':42,'error':err,'timing_scope':'predict_only',
                         'python':platform.python_version(),'sklearn':sklearn.__version__,
                         'dataset_sha256':dataset_hash,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         'sample_origin':'UCI 327' if 'uci327' in a.data.lower() else 'user_supplied',
                         'data_rows':len(df), 'observations':'measured_model_predict_calls'}
                    f.write(json.dumps(rec)+'\n')
    print('Saved',out, '| Source:',a.data,'| Batch sizes are measured, not RPS')
if __name__=='__main__':main()
