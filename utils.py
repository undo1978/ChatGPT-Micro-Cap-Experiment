import functools
import time
import pandas as pd
from typing import Callable, Any, List, Dict


def retry(tries: int = 3, delay: float = 2.0, backoff: float = 1.0):
    """
    Retry decorator with optional exponential backoff.
    Example: @retry(tries=3, delay=1.0, backoff=2.0)
    will wait 1s, 2s, 4s between attempts.
    """
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _delay = delay
            last_exc = None
            for attempt in range(tries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < tries - 1:
                        time.sleep(_delay)
                        _delay *= backoff
            raise last_exc
        return wrapper
    return deco


def write_csv_atomic(path: str, df: pd.DataFrame) -> None:
    """
    Write a DataFrame to CSV atomically (safe write).
    """
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False)
    import os
    os.replace(tmp_path, path)


def recos_to_html(recos: List[Dict[str, Any]]) -> str:
    """
    Format recommendations (RECOMMENDATIONS list) into an HTML table.
    """
    if not recos:
        return "<p>(täna ettepanekuid ei tekkinud)</p>"

    rows = []
    for r in recos:
        rows.append(
            f"<tr><td>{r['action']}</td>"
            f"<td>{r['ticker']}</td>"
            f"<td>{r.get('target_shares','')}</td>"
            f"<td>{r.get('stop','')}</td>"
            f"<td>{r.get('reason','')}</td></tr>"
        )

    table = (
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>Action</th><th>Ticker</th><th>Shares</th><th>Stop</th><th>Reason</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return table


def df_to_html(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame into a nice HTML table.
    """
    if df is None or df.empty:
        return "<p>(tabel tühi)</p>"
    return df.to_html(index=False, border=1, justify="center")


def dedupe_recos(recos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate recommendations by ticker+action, keeping the last one.
    """
    seen = {}
    for r in recos:
        key = (r["ticker"], r["action"])
        seen[key] = r
    return list(seen.values())
