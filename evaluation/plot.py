"""Save run-metric plots into the run directory."""
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_metrics(run_dir):
 run=Path(run_dir); records=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line]
 if not records: raise ValueError('No metrics to plot.')
 plots=run/'plots'; plots.mkdir(exist_ok=True); steps=[r['step'] for r in records]
 for key,label in [('loss','Training loss'),('learning_rate','Learning rate'),('tokens_per_second','Tokens/sec')]:
  fig,ax=plt.subplots(); ax.plot(steps,[r[key] for r in records]); ax.set(xlabel='Step',ylabel=label,title=f'{run.name} — {label}'); fig.savefig(plots/f'{key}.png',dpi=150,bbox_inches='tight'); plt.close(fig)
 return plots
