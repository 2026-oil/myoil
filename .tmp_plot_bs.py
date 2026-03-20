from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

base = Path('/home/sonet/.openclaw/workspace/research/neuralforecast')
csv_path = base / 'data' / 'df.csv'
out_dir = base / 'artifacts'
out_dir.mkdir(exist_ok=True)
out_path = out_dir / 'bs_columns_plot.png'

# Font setup if available
font_candidates = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
for fp in font_candidates:
    if Path(fp).exists():
        plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(csv_path, encoding='utf-8-sig')
if 'dt' not in df.columns:
    raise SystemExit('dt column not found')

bs_cols = [c for c in df.columns if c.startswith('BS_')]
if not bs_cols:
    raise SystemExit('No BS_ columns found')

# Parse dates robustly
try:
    dt = pd.to_datetime(df['dt'])
except Exception:
    dt = pd.to_datetime(df['dt'], errors='coerce')

plot_df = df[bs_cols].copy()
plot_df.index = dt
plot_df = plot_df.dropna(how='all')

fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
colors = ['#2563eb', '#dc2626', '#16a34a', '#7c3aed', '#ea580c', '#0891b2']
for i, col in enumerate(bs_cols):
    ax.plot(plot_df.index, plot_df[col], label=col.replace('BS_', ''), linewidth=2.0, color=colors[i % len(colors)])

ax.set_title('BS_* Series from df.csv', fontsize=16, pad=14)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='best', frameon=True)
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig(out_path, bbox_inches='tight')
print(out_path)
print('columns:', ', '.join(bs_cols))
