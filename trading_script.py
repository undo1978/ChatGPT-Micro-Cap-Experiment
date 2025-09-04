from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import argparse
import io
import logging
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import CFG
from email_utils import send_email_html
from utils import (
    retry,
    write_csv_atomic,
    recos_to_html,
    df_to_html,
    dedupe_recos,
)

# ---------------- Optional: market calendar (XNYS) ----------------
_HAS_MCAL = False
_XNYS = None
try:
    import pandas_market_calendars as mcal  # optional dependency (declared in extra-requirements.txt)

    _XNYS = mcal.get_calendar("XNYS")
    _HAS_MCAL = True
except Exception:  # pragma: no cover
    _HAS_MCAL = False
    _XNYS = None


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Run without interactive prompts")
    p.add_argument("--allocation", type=float, default=0.10, help="Max osakaal ühest tickerist (0.10=10%)")
    p.add_argument("--stop_pct", type=float, default=CFG.STOP_PCT, help="Stop-loss protsent (0.1=10%)")
    p.add_argument(
        "--sizing",
        choices=["allocation", "risk"],
        default="allocation",
        help="Position sizing meetod",
    )
    p.add_argument("--risk_perc", type=float, default=CFG.RISK_PERC, help="Risk per trade % portfellist")
    p.add_argument(
        "--max_spread_pct",
        type=float,
        default=CFG.MAX_SPREAD_PCT,
        help="Max intraday spread % (High-Low)/Close*100",
    )
    p.add_argument("--trailing_window", type=int, default=CFG.TRAILING_WINDOW, help="Trailing stop aken (päevi)")
    return p.parse_args()


_args: Optional[argparse.Namespace] = None


# ---------------- Date helpers ----------------

ASOF_DATE: Optional[pd.Timestamp] = None


def set_asof(date: Optional[str]) -> None:
    global ASOF_DATE
    if not date:
        ASOF_DATE = None
        return
    ASOF_DATE = pd.Timestamp(date).normalize()


def _now() -> datetime:
    return ASOF_DATE.to_pydatetime() if ASOF_DATE is not None else datetime.now()


def last_trading_date(ts: Optional[datetime] = None) -> pd.Timestamp:
    d = pd.Timestamp(ts or _now())
    # weekend -> Friday
    if d.weekday() == 5:  # Sat
        return (d - pd.Timedelta(days=1)).normalize()
    if d.weekday() == 6:  # Sun
        return (d - pd.Timedelta(days=2)).normalize()
    # optional holiday check
    if _HAS_MCAL and _XNYS is not None:
        # If market closed today (holiday), take previous session
        sched = _XNYS.schedule(start_date=d.date(), end_date=d.date())
        if sched.empty:
            prev = _XNYS.valid_days(start_date=d - pd.Timedelta(days=10), end_date=d)
            if len(prev) > 0:
                return pd.Timestamp(prev[-1]).normalize()
    return d.normalize()


def trading_day_window(target: Optional[datetime] = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = last_trading_date(target)
    return d, d + pd.Timedelta(days=1)


# ---------------- Data access ----------------

@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str  # "yahoo" | "stooq-csv" | "yahoo:proxy" | "empty"


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


@retry(tries=2, delay=0.5, backoff=2.0)
def _yahoo_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    import requests as _req

    sess = _req.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    kwargs.setdefault("session", sess)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            df = cast(pd.DataFrame, yf.download(ticker, **kwargs))
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _stooq_csv_download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # daily CSV: https://stooq.com/q/d/l/?s=spy.us&i=d
    t = ticker
    if not t.startswith("^"):
        sym = t.lower()
        if not sym.endswith(".us"):
            sym = f"{sym}.us"
    else:
        sym = t.lower()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # filter [start, end)
        df = df.loc[(df.index >= start.normalize()) & (df.index < end.normalize())]
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


def _weekend_safe_range(period: Optional[str], start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start or end:
        e = pd.Timestamp(end) if end else last_trading_date() + pd.Timedelta(days=1)
        s = pd.Timestamp(start) if start else (e - pd.Timedelta(days=5))
        return s.normalize(), e.normalize()
    if isinstance(period, str) and period.endswith("d"):
        days = int(period[:-1])
    else:
        days = 1
    end_tr = last_trading_date()
    s = (end_tr - pd.Timedelta(days=days)).normalize()
    e = (end_tr + pd.Timedelta(days=1)).normalize()
    return s, e


def download_price_data(ticker: str, **kwargs: Any) -> FetchResult:
    period = kwargs.pop("period", None)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)

    s, e = _weekend_safe_range(period, start, end)

    df_y = _yahoo_download(ticker, start=s, end=e, **kwargs)
    if isinstance(df_y, pd.DataFrame) and not df_y.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_y)), "yahoo")

    # fallback: Stooq CSV (works for many US tickers / ETFs)
    df_s = _stooq_csv_download(ticker, s, e)
    if isinstance(df_s, pd.DataFrame) and not df_s.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_s)), "stooq-csv")

    # proxy for indices commonly missing on Stooq
    proxy_map = {"^GSPC": "SPY", "^RUT": "IWM"}
    if ticker in proxy_map:
        df_p = _yahoo_download(proxy_map[ticker], start=s, end=e, **kwargs)
        if isinstance(df_p, pd.DataFrame) and not df_p.empty:
            return FetchResult(_normalize_ohlcv(_to_datetime_index(df_p)), "yahoo:proxy")

    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])  # for safety
    return FetchResult(empty, "empty")


# ---------------- Portfolio sizing helpers ----------------

def get_last_close_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period="5d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def shares_by_allocation(ticker: str, allocation: float, cash: float) -> int:
    price = get_last_close_price(ticker)
    if price <= 0:
        return 0
    target = max(0.0, allocation) * (cash)
    return int(target // price)


def shares_by_risk(ticker: str, risk_perc: float, stop_pct: float, equity: float, cash: float) -> int:
    """Risk-based sizing: position_value * stop_pct = equity * risk_perc."""
    price = get_last_close_price(ticker)
    if price <= 0 or stop_pct <= 0:
        return 0
    risk_cash = max(0.0, risk_perc) * equity
    pos_value = risk_cash / max(stop_pct, 1e-9)
    pos_value = min(pos_value, cash)  # never exceed available cash in headless path
    return int(pos_value // price)


# ---------------- Global paths ----------------

SCRIPT_DIR = Path(__file__).resolve().parent
PORTFOLIO_CSV = SCRIPT_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = SCRIPT_DIR / "chatgpt_trade_log.csv"


# ---------------- Recommendations capture ----------------

RECOMMENDATIONS: List[Dict[str, Any]] = []


def add_reco(action: str, ticker: str, reason: str = "", target_shares: int = 0, stop: float = 0.0) -> None:
    RECOMMENDATIONS.append(
        {
            "action": action.upper(),
            "ticker": ticker,
            "reason": reason,
            "target_shares": int(target_shares),
            "stop": float(stop),
        }
    )


# ---------------- Core processing ----------------

def process_portfolio(
    portfolio: pd.DataFrame | Dict[str, List[Any]] | List[Dict[str, Any]],
    cash: float,
    interactive: bool,
) -> tuple[pd.DataFrame, float]:
    """Price holdings for the last trading day, execute stops, append CSV rows."""
    today_iso = last_trading_date().date().isoformat()
    dfp = portfolio.copy() if isinstance(portfolio, pd.DataFrame) else pd.DataFrame(portfolio)
    if dfp.empty:
        dfp = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])

    results: List[Dict[str, Any]] = []
    total_value = 0.0
    total_pnl = 0.0

    s, e = trading_day_window()

    for _, row in dfp.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        shares = float(row.get("shares", 0) or 0)
        stop = float(row.get("stop_loss", 0) or 0)
        buy_price = float(row.get("buy_price", 0) or 0)
        cost_basis = float(row.get("cost_basis", buy_price * shares) or 0)

        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
        data = fetch.df
        if data.empty:
            results.append(
                {
                    "Date": today_iso,
                    "Ticker": ticker,
                    "Shares": shares,
                    "Buy Price": buy_price,
                    "Cost Basis": cost_basis,
                    "Stop Loss": stop,
                    "Current Price": "",
                    "Total Value": "",
                    "PnL": "",
                    "Action": "NO DATA",
                    "Cash Balance": "",
                    "Total Equity": "",
                }
            )
            continue

        o = float(data.get("Open", data["Close"]).iloc[-1])
        h = float(data["High"].iloc[-1])
        l = float(data["Low"].iloc[-1])
        c = float(data["Close"].iloc[-1])

        if stop and l <= stop:
            # stop triggered -> assume fill at min(o, stop)
            exec_price = round(o if o <= stop else stop, 2)
            value = round(exec_price * shares, 2)
            pnl = round((exec_price - buy_price) * shares, 2)
            cash += value
            # remove position in dfp
            dfp = dfp[dfp["ticker"].str.upper() != ticker]
            action = "SELL - Stop Loss Triggered"
        else:
            exec_price = round(c, 2)
            value = round(exec_price * shares, 2)
            pnl = round((exec_price - buy_price) * shares, 2)
            total_value += value
            total_pnl += pnl
            action = "HOLD"

        results.append(
            {
                "Date": today_iso,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": buy_price,
                "Cost Basis": cost_basis,
                "Stop Loss": stop,
                "Current Price": exec_price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": "",
            }
        )

    total_row = {
        "Date": today_iso,
        "Ticker": "TOTAL",
        "Shares": "",
        "Buy Price": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2),
        "Action": "",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    out = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != today_iso]
        out = pd.concat([existing, out], ignore_index=True)
    write_csv_atomic(PORTFOLIO_CSV, out)

    return dfp.reset_index(drop=True), cash


# ---------------- Load/save ----------------

def load_latest_portfolio_state(file: str) -> tuple[pd.DataFrame | List[Dict[str, Any]], float]:
    p = Path(file)
    if not p.exists():
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return portfolio, float(CFG.STARTING_CASH)

    df = pd.read_csv(p)
    if df.empty:
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return portfolio, float(CFG.STARTING_CASH)

    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"], format="mixed", errors="coerce")
    latest_date = non_total["Date"].max()
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    sold_mask = latest_tickers["Action"].astype(str).str.startswith("SELL")
    latest_tickers = latest_tickers[~sold_mask].copy()
    latest_tickers.drop(
        columns=[
            "Date",
            "Cash Balance",
            "Total Equity",
            "Action",
            "Current Price",
            "PnL",
            "Total Value",
        ],
        inplace=True,
        errors="ignore",
    )
    latest_tickers.rename(
        columns={
            "Cost Basis": "cost_basis",
            "Buy Price": "buy_price",
            "Shares": "shares",
            "Ticker": "ticker",
            "Stop Loss": "stop_loss",
        },
        inplace=True,
    )
    latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient="records")

    df_total = df[df["Ticker"] == "TOTAL"].copy()
    df_total["Date"] = pd.to_datetime(df_total["Date"], format="mixed", errors="coerce")
    latest = df_total.sort_values("Date").iloc[-1]
    cash = float(latest["Cash Balance"]) if "Cash Balance" in latest else float(CFG.STARTING_CASH)
    return latest_tickers, cash


# ---------------- Main ----------------

def main() -> None:
    global _args
    _args = parse_args()

    asof_env = os.environ.get("ASOF_DATE")
    if asof_env:
        set_asof(asof_env)

    portfolio, cash = load_latest_portfolio_state(str(PORTFOLIO_CSV))
    if isinstance(portfolio, (list, dict)):
        portfolio = pd.DataFrame(portfolio)

    # Optional: headless demo BUY if HEADLESS_TICKER provided via env
    if _args.headless:
        ticker = os.getenv("HEADLESS_TICKER", "").strip().upper()
        if ticker:
            s, e = trading_day_window()
            fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
            df = fetch.df
            if not df.empty:
                close = float(df["Close"].iloc[-1])
                high = float(df["High"].iloc[-1])
                low = float(df["Low"].iloc[-1])
                vol = float(df["Volume"].iloc[-1])
                spread_pct = (high - low) / close * 100 if close else 999.0
                if spread_pct <= _args.max_spread_pct and vol > 0:
                    # approximate equity (cash + book value of holdings)
                    equity = cash
                    for _, r in portfolio.iterrows():
                        s_ = float(r.get("shares", 0) or 0)
                        bp = float(r.get("buy_price", 0) or 0)
                        equity += s_ * bp
                    if _args.sizing == "risk":
                        shares = shares_by_risk(ticker, _args.risk_perc, _args.stop_pct, equity, cash)
                    else:
                        shares = shares_by_allocation(ticker, _args.allocation, cash)
                    stop = round(close * (1.0 - _args.stop_pct), 4)
                    if shares > 0:
                        add_reco("BUY", ticker, reason=f"headless auto-buy {ticker}", target_shares=shares, stop=stop)
                else:
                    add_reco(
                        "HOLD",
                        ticker,
                        reason=f"skipped: spread {spread_pct:.2f}%>{_args.max_spread_pct}% või volume=0",
                    )

    portfolio, cash = process_portfolio(portfolio, cash, interactive=(not _args.headless))

    # Build email (HTML) and send
    recos = dedupe_recos(RECOMMENDATIONS)
    html = (
        f"<h3>Päevane portfellisoovitus</h3>"
        f"<p>Kuupäev: {last_trading_date().date().isoformat()}</p>"
        + recos_to_html(recos)
        + "<h4>Portfelli snapshot</h4>"
        + df_to_html(portfolio)
        + "<p>Märkused: micro-cap volatiilsus, slippage, likviidsus; info on hariduslik, mitte investeerimisnõu.</p>"
    )
    try:
        if _args.headless:
            send_email_html("Päevane portfellisoovitus", html, text=None, attachments=[str(PORTFOLIO_CSV)])
    except Exception as e:  # pragma: no cover
        logging.exception("Email send failed: %s", e)


if __name__ == "__main__":
    main()
