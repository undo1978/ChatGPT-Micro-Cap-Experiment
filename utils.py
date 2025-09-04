from __future__ import annotations
import time, os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
import pandas as pd
from typing import Callable, Iterable, Any, List, Dict


# ---- retry/backoff ----


def retry(times: int = 3, base_delay: float = 0.8):
def deco(fn: Callable):
@wraps(fn)
def inner(*args, **kwargs):
last = None
for i in range(times):
try:
return fn(*args, **kwargs)
except Exception as e:
last = e
time.sleep(base_delay * (2 ** i))
if last:
raise last
return inner
return deco


# ---- atomiline CSV ----


def write_csv_atomic(path: Path, df: pd.DataFrame) -> None:
path = Path(path)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
df.to_csv(tmp, index=False)
os.replace(tmp, path)


# ---- RECO HTML ----


def recos_to_html(recos: List[Dict[str, Any]]) -> str:
if not recos:
return "<p>(täna ettepanekuid ei tekkinud)</p>"
rows = []
for r in recos:
rows.append(
f"<tr><td>{r['action']}</td><td>{r['ticker']}</td><td style='text-align:right'>{r.get('target_shares','')}</td>"
f"<td style='text-align:right'>{r.get('stop','')}</td><td>{r.get('reason','')}</td></tr>"
)
return (
"<table border='1' cellpadding='6' cellspacing='0'><thead><tr>"
"<th>Action</th><th>Ticker</th><th>Kogus</th><th>Stop</th><th>Põhjus</th>"
"</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
)




def df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
if df is None or df.empty:
return "<p>(tühi)</p>"
return df.head(max_rows).to_html(index=False, border=1)


# ---- dedupe ----


def dedupe_recos(recos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
seen = set(); out = []
for r in recos:
key = (r.get('action'), r.get('ticker'), int(r.get('target_shares', 0)), float(r.get('stop', 0.0)))
if key in seen: continue
seen.add(key)
if int(r.get('target_shares', 0)) > 0:
out.append(r)
return out
