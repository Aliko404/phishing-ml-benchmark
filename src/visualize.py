"""Scientific charts (PDF vector + 300 DPI PNG), with uncertainty and B/W styles."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .process_data import load_and_clean_data,calculate_statistics

def generate_plots(df:pd.DataFrame, output_dir:str|Path,seed:int=42):
    stats=calculate_statistics(df,seed=seed)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    markers=['o','s','^','D'];styles=['-','--',':','-.']
    colors=plt.get_cmap('tab10').colors
    fig,ax=plt.subplots(figsize=(8,5))
    for i,(model,g) in enumerate(stats.groupby('model_variant',sort=True)):
        g=g.sort_values('batch_size');good=g.dropna(subset=['median_ms'])
        if good.empty:continue
        x=good.batch_size.to_numpy(dtype=float)
        ax.errorbar(x,good.median_ms,yerr=np.vstack((good.median_ms-good.median_ci_lower_ms,
                    good.median_ci_upper_ms-good.median_ms)),label=model,linestyle=styles[i%4],
                    marker=markers[i%4],color=colors[i%len(colors)],capsize=3,linewidth=1.5)
    ax.set(xlabel='Input batch size (samples per predict call)',ylabel='Predict latency (ms)',
           title='Inference latency by batch size (points: median; bars: bootstrap 95% CI)')
    ax.legend(loc='best');fig.tight_layout()
    for fmt in ('pdf','png'):fig.savefig(out/f'fig1_latency_vs_load.{fmt}',dpi=300)
    plt.close(fig)
    good=df[df.status_code.eq(200)&df.latency_ms.notna()&(df.latency_ms>=0)]
    models=sorted(good.model_variant.unique())
    if not models:raise ValueError('Cannot plot: no successful predictions')
    arrays=[good.loc[good.model_variant.eq(m),'latency_ms'].to_numpy(dtype=float) for m in models]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.boxplot(arrays,tick_labels=models,showfliers=True,patch_artist=False)
    ax.set(xlabel='Model variant',ylabel='Predict latency (ms)',
           title='Successful prediction latency (warm-up excluded; mixed batch sizes)')
    ax.tick_params(axis='x',rotation=12);fig.tight_layout()
    for fmt in ('pdf','png'):fig.savefig(out/f'fig2_distribution_boxplot.{fmt}',dpi=300)
    plt.close(fig)
    return stats

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='data/raw/test_run.jsonl')
    parser.add_argument('--output-dir',default='reports/figures')
    args=parser.parse_args()
    result=generate_plots(load_and_clean_data(args.input),args.output_dir)
    print(f'Saved 4 figures under {args.output_dir}; {len(result)} groups')
if __name__=='__main__': main()
