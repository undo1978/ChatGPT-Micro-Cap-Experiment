"""Utilities for maintaining the ChatGPT micro-cap portfolio.

This module rewrites the original script to:
- Centralize market data fetching with a robust Yahoo->Stooq fallback
- Ensure ALL price requests go through the same accessor
- Handle empty Yahoo frames (no exception) so fallback actually triggers
- Normalize Stooq output to Yahoo-like columns
- Make weekend handling consistent and testable
- Keep behavior and CSV formats compatible with prior runs

Notes:
- Some tickers/indices are not available on Stooq (e.g., ^RUT). These stay on Yahoo.
- Stooq end date is exclusive; we add +1 day for ranges.
- "Adj Close" is set equal to "Close" for Stooq to match downstream expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf


warnings.simplefilter("ignore", category=FutureWarning)
try:
    import pandas_datareader.data as pdr
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

# ------------------------------
# Globals / file locations
# ------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # Save files alongside this script by default
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"


# ------------------------------
# Date helpers
# ------------------------------

def last_trading_date(today: datetime | None = None) -> pd.Timestamp:
    """Return last trading date (M-F), mapping Sat->Fri and Sun->Fri.

    This function does *not* know market holidays; it specifically handles weekends.
    """
    dt = pd.Timestamp(today or datetime.now())
    # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    if dt.weekday() == 5:
        return (dt - pd.Timedelta(days=1)).normalize()
    if dt.weekday() == 6:
        return (dt - pd.Timedelta(days=2)).normalize()
    return dt.normalize()


def check_weekend() -> str:
    """Backwards-compatible wrapper returning ISO date string for last trading day."""
    return last_trading_date().date().isoformat()


# ------------------------------
# Data access layer
# ------------------------------

# Known Stooq symbol remaps for common indices
STOOQ_MAP = {
    "^GSPC": "^SPX",  # S&P 500
    "^DJI": "^DJI",   # Dow Jones
    "^IXIC": "^IXIC", # Nasdaq Composite
    # "^RUT": not on Stooq; keep Yahoo
}

# Symbols we should *not* attempt on Stooq
STOOQ_BLOCKLIST = {"^RUT"}


@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str  # "yahoo" or "stooq" or "empty"


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        # yfinance returns DatetimeIndex; pdr.stooq returns DatetimeIndex too
        # but be defensive
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    # Ensure all expected columns exist
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    if "Adj Close" not in df.columns:
        # For Stooq we set Adj Close = Close (no adjustments)
        df["Adj Close"] = df["Close"]
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def _yahoo_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    """Call yfinance.download with a real UA and silence all chatter."""
    import io, logging, requests
    from contextlib import redirect_stderr, redirect_stdout

    # Robust session with UA to avoid 403/999
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    kwargs.setdefault("session", sess)

    # Silence yfinance logger + stdout/stderr
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                df = cast(pd.DataFrame, yf.download(ticker, **kwargs))
        except Exception:
            return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _stooq_csv_download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV from Stooq CSV endpoint (daily). Good for US tickers and many ETFs."""
    import requests
    # Map ^GSPC -> ^SPX (Stooq naming), block ^RUT
    if ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()
    t = STOOQ_MAP.get(ticker, ticker)

    # Stooq daily CSV wants lowercase symbols; US equities/ETFs use .us suffix.
    if not t.startswith("^"):
        sym = t.lower()
        if not sym.endswith(".us"):
            sym = f"{sym}.us"
    else:
        # Indices keep caret, in lowercase (e.g., ^spx)
        sym = t.lower()

    # Stooq uses inclusive start and exclusive end – we’ll filter after download
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(pd.compat.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()
        # Normalize columns + index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        # Rename to Yahoo-like
        df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"}, inplace=True)
        # Keep within [start, end)
        df = df.loc[(df.index >= start.normalize()) & (df.index < end.normalize())]
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return df[["Open","High","Low","Close","Adj Close","Volume"]]
    except Exception:
        return pd.DataFrame()



def _stooq_download(
    ticker: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Fetch OHLCV from Stooq via pandas-datareader; returns empty DF on failure."""
    if not _HAS_PDR:
        return pd.DataFrame()
    if ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()

    # Map common indices and lowercase regular tickers for Stooq
    t = STOOQ_MAP.get(ticker, ticker)
    if not t.startswith("^"):
        t = t.lower()

    try:
        df = cast(pd.DataFrame, pdr.DataReader(t, "stooq", start=start, end=end))
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()



def _period_to_range(period: str | None, start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Translate a Yahoo-style period (e.g., '2d') or explicit start/end to a date range.

    Stooq uses inclusive start and exclusive end; we add +1 day to end to cover the last bar.
    """
    if period and not start and not end and isinstance(period, str) and period.endswith("d"):
        days = int(period[:-1])
        end_ts = pd.Timestamp(datetime.now())
        start_ts = end_ts - pd.Timedelta(days=days)
    else:
        end_ts = pd.Timestamp(end) if end else pd.Timestamp(datetime.now())
        start_ts = pd.Timestamp(start) if start else (end_ts - pd.Timedelta(days=5))
    # Make Stooq end exclusive by adding one day
    return start_ts.normalize(), (end_ts + pd.Timedelta(days=1)).normalize()


def download_price_data(ticker: str, **kwargs: Any) -> FetchResult:
    """
    Robust OHLCV fetch with multi-stage fallbacks:

    Order:
      1) Yahoo Finance via yfinance (with UA & silenced; see _yahoo_download)
      2) Stooq via pandas-datareader (see _stooq_download)
      3) Stooq direct CSV (see _stooq_csv_download)
      4) Index proxies (e.g., ^GSPC->SPY, ^RUT->IWM) via Yahoo

    Accepts Yahoo-like kwargs (period/start/end/auto_adjust/progress/etc).
    Returns: FetchResult(df, source) where df has columns:
      [Open, High, Low, Close, Adj Close, Volume]
    """
    # Normalize kwargs for yfinance
    period = kwargs.pop("period", None)
    start = kwargs.get("start")
    end = kwargs.get("end")
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)

    # ---------- 1) Yahoo ----------
    df_y = _yahoo_download(ticker, period=period, **kwargs)
    if isinstance(df_y, pd.DataFrame) and not df_y.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_y)), "yahoo")

    # Build a concrete [start, end) window for Stooq fallbacks
    s, e = _period_to_range(period, start, end)

    # ---------- 2) Stooq via pandas-datareader ----------
    df_s = _stooq_download(ticker, start=s, end=e)
    if isinstance(df_s, pd.DataFrame) and not df_s.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_s)), "stooq-pdr")

    # ---------- 3) Stooq direct CSV ----------
    df_csv = _stooq_csv_download(ticker, s, e)
    if isinstance(df_csv, pd.DataFrame) and not df_csv.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_csv)), "stooq-csv")

    # ---------- 4) Proxies for common indices when all else fails ----------
    proxy_map = {
        "^GSPC": "SPY",
        "^RUT":  "IWM",
    }
    proxy = proxy_map.get(ticker)
    if proxy:
        df_proxy = _yahoo_download(proxy, period=period, **kwargs)
        if isinstance(df_proxy, pd.DataFrame) and not df_proxy.empty:
            return FetchResult(_normalize_ohlcv(_to_datetime_index(df_proxy)), f"yahoo:{proxy}-proxy")

    # ---------- Nothing worked ----------
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    return FetchResult(empty, "empty")



# ------------------------------
# File path configuration
# ------------------------------

def set_data_dir(data_dir: Path) -> None:
    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    DATA_DIR = Path(data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"


# ------------------------------
# Portfolio operations
# ------------------------------

def _ensure_df(portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]]) -> pd.DataFrame:
    if isinstance(portfolio, pd.DataFrame):
        return portfolio.copy()
    if isinstance(portfolio, (dict, list)):
        return pd.DataFrame(portfolio)
    raise TypeError("portfolio must be a DataFrame, dict, or list[dict]")


def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    Update daily price information, execute stop-loss sells, and (optionally) prompt for trades.

    Additions:
      - BUY (MOO): market-on-open buy that executes at today's Open (falls back to Close if Open missing).
      - BUY (LIMIT): unchanged (delegates to log_manual_buy).
      - SELL (LIMIT): unchanged (delegates to log_manual_sell).
    """
    today_iso = last_trading_date().date().isoformat()
    portfolio_df = _ensure_df(portfolio)

    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    # ------- Interactive trade entry (now supports MOO) -------
    if interactive:
        while True:
            print(portfolio_df)
            action = input(
                f""" You have {cash} in cash.
Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: """
            ).strip().lower()

            if action == "b":
                ticker = input("Enter ticker symbol: ").strip().upper()
                order_type = input("Order type? 'm' = market-on-open, 'l' = limit: ").strip().lower()

                try:
                    shares = float(input("Enter number of shares: "))
                    if shares <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid share amount. Buy cancelled.")
                    continue

                # ---- MOO buy (executes at Open) ----
                if order_type == "m":
                    try:
                        stop_loss = float(input("Enter stop loss (or 0 to skip): "))
                        if stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid stop loss. Buy cancelled.")
                        continue

                    # Fetch today's bar and fill at Open
                    fetch = download_price_data(ticker, period="1d", auto_adjust=False, progress=False)
                    data = fetch.df
                    if data.empty:
                        print(f"MOO buy for {ticker} failed: no market data available (source={fetch.source}).")
                        continue

                    o = float(data["Open"].iloc[-1]) if "Open" in data else float(data["Close"].iloc[-1])
                    exec_price = round(o, 2)
                    notional = exec_price * shares
                    if notional > cash:
                        print(f"MOO buy for {ticker} failed: cost {notional:.2f} exceeds cash {cash:.2f}.")
                        continue

                    # ---- Log trade to trade log CSV ----
                    log = {
                        "Date": today_iso,
                        "Ticker": ticker,
                        "Shares Bought": shares,
                        "Buy Price": exec_price,
                        "Cost Basis": notional,
                        "PnL": 0.0,
                        "Reason": "MANUAL BUY MOO - Filled",
                    }
                    if os.path.exists(TRADE_LOG_CSV):
                        df_log = pd.read_csv(TRADE_LOG_CSV)
                        df_log = pd.concat([df_log, pd.DataFrame([log])], ignore_index=True)
                    else:
                        df_log = pd.DataFrame([log])
                    df_log.to_csv(TRADE_LOG_CSV, index=False)

                    # ---- Update portfolio (weighted avg if adding) ----
                    rows = portfolio_df.loc[portfolio_df["ticker"].astype(str).str.upper() == ticker.upper()]
                    if rows.empty:
                        new_trade = {
                            "ticker": ticker,
                            "shares": float(shares),
                            "stop_loss": float(stop_loss),
                            "buy_price": float(exec_price),
                            "cost_basis": float(notional),
                        }
                        portfolio_df = pd.concat([portfolio_df, pd.DataFrame([new_trade])], ignore_index=True)
                    else:
                        idx = rows.index[0]
                        cur_shares = float(portfolio_df.at[idx, "shares"])
                        cur_cost = float(portfolio_df.at[idx, "cost_basis"])
                        new_shares = cur_shares + float(shares)
                        new_cost = cur_cost + float(notional)
                        avg_price = new_cost / new_shares if new_shares else 0.0
                        portfolio_df.at[idx, "shares"] = new_shares
                        portfolio_df.at[idx, "cost_basis"] = new_cost
                        portfolio_df.at[idx, "buy_price"] = avg_price
                        portfolio_df.at[idx, "stop_loss"] = float(stop_loss)

                    cash -= notional
                    print(f"Manual BUY MOO for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
                    continue

                # ---- LIMIT buy (existing behavior) ----
                elif order_type == "l":
                    try:
                        buy_price = float(input("Enter buy LIMIT price: "))
                        stop_loss = float(input("Enter stop loss (or 0 to skip): "))
                        if buy_price <= 0 or stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid input. Limit buy cancelled.")
                        continue

                    cash, portfolio_df = log_manual_buy(
                        buy_price, shares, ticker, stop_loss, cash, portfolio_df
                    )
                    continue

                else:
                    print("Unknown order type. Use 'm' or 'l'.")
                    continue

            if action == "s":
                try:
                    ticker = input("Enter ticker symbol: ").strip().upper()
                    shares = float(input("Enter number of shares to sell (LIMIT): "))
                    sell_price = float(input("Enter sell LIMIT price: "))
                    if shares <= 0 or sell_price <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Manual sell cancelled.")
                    continue

                cash, portfolio_df = log_manual_sell(
                    sell_price, shares, ticker, cash, portfolio_df
                )
                continue

            # empty => proceed to pricing loop
            break

    # ------- Daily pricing + stop-loss execution -------
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock["ticker"]).upper()
        shares = int(stock["shares"]) if not pd.isna(stock["shares"]) else 0
        cost = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
        cost_basis = float(stock["cost_basis"]) if not pd.isna(stock["cost_basis"]) else cost * shares
        stop = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

        fetch = download_price_data(ticker, period="1d", auto_adjust=False, progress=False)
        data = fetch.df

        if data.empty:
            print(f"No data for {ticker} (source={fetch.source}).")
            row = {
                "Date": today_iso, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost, "Cost Basis": cost_basis, "Stop Loss": stop,
                "Current Price": "", "Total Value": "", "PnL": "",
                "Action": "NO DATA", "Cash Balance": "", "Total Equity": "",
            }
            results.append(row)
            continue

        # Extract OHLC defensively
        o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
        h = float(data["High"].iloc[-1])
        l = float(data["Low"].iloc[-1])
        c = float(data["Close"].iloc[-1])
        if np.isnan(o):
            o = c

        # --- Stop-loss (market) with gap handling ---
        if stop and l <= stop:
            exec_price = round(o if o <= stop else stop, 2)
            value = round(exec_price * shares, 2)
            pnl = round((exec_price - cost) * shares, 2)
            action = "SELL - Stop Loss Triggered"
            cash += value
            portfolio_df = log_sell(ticker, shares, exec_price, cost, pnl, portfolio_df)
            row = {
                "Date": today_iso, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost, "Cost Basis": cost_basis, "Stop Loss": stop,
                "Current Price": exec_price, "Total Value": value, "PnL": pnl,
                "Action": action, "Cash Balance": "", "Total Equity": "",
            }
        else:
            price = round(c, 2)
            value = round(price * shares, 2)
            pnl = round((price - cost) * shares, 2)
            action = "HOLD"
            total_value += value
            total_pnl += pnl
            row = {
                "Date": today_iso, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost, "Cost Basis": cost_basis, "Stop Loss": stop,
                "Current Price": price, "Total Value": value, "PnL": pnl,
                "Action": action, "Cash Balance": "", "Total Equity": "",
            }

        results.append(row)

    # ------- Append TOTAL row & persist -------
    total_row = {
        "Date": today_iso, "Ticker": "TOTAL", "Shares": "", "Buy Price": "",
        "Cost Basis": "", "Stop Loss": "", "Current Price": "",
        "Total Value": round(total_value, 2), "PnL": round(total_pnl, 2),
        "Action": "", "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    df_out = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != str(today_iso)]
        print("Saving results to CSV...")
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(PORTFOLIO_CSV, index=False)

    return portfolio_df, cash



# ------------------------------
# Trade logging
# ------------------------------

def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    today = check_weekend()
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }
    print(f"{ticker} stop loss was met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    return portfolio


def log_manual_buy(
    buy_price: float,   # interpreted as LIMIT price
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    """IOC buy-limit:
    Fill at Open if Open <= limit,
    else at limit if Low <= limit,
    else no fill.
    """

    today = check_weekend()

    # Confirm trade if interactive
    if interactive:
        check = input(
            f"You are placing a BUY LIMIT for {shares} {ticker} at ${buy_price:.2f}.\n"
            f"If this is a mistake, type '1': "
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    # Ensure portfolio has required structure
    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

    # Get today’s price data
    fetch = download_price_data(ticker, period="1d", auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual buy for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    o = float(data.get("Open", [np.nan])[-1])
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])  # fallback

    # --- IOC buy-limit fill logic ---
    if o <= buy_price:
        exec_price = o  # better fill at open
    elif l <= buy_price:
        exec_price = buy_price  # touched intraday
    else:
        print(f"Buy limit ${buy_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
        return cash, chatgpt_portfolio

    cost_amt = exec_price * shares
    if cost_amt > cash:
        print(f"Manual buy for {ticker} failed: cost {cost_amt:.2f} exceeds cash balance {cash:.2f}.")
        return cash, chatgpt_portfolio

    # Log trade
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": exec_price,
        "Cost Basis": cost_amt,
        "PnL": 0.0,
        "Reason": "MANUAL BUY LIMIT - Filled",
    }
    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    # Update portfolio
    rows = chatgpt_portfolio.loc[chatgpt_portfolio["ticker"].str.upper() == ticker.upper()]
    if rows.empty:
        chatgpt_portfolio = pd.concat(
            [chatgpt_portfolio, pd.DataFrame([{
                "ticker": ticker,
                "shares": float(shares),
                "stop_loss": float(stoploss),
                "buy_price": float(exec_price),
                "cost_basis": float(cost_amt),
            }])],
            ignore_index=True
        )
    else:
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])
        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(cost_amt)
        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = new_cost / new_shares if new_shares else 0.0
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss)

    cash -= cost_amt
    print(f"Manual BUY LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
    return cash, chatgpt_portfolio




def log_manual_sell(
    sell_price: float,  # interpreted as LIMIT price
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    reason: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    """IOC sell-limit: fill at Open if Open>=limit, else at limit if High>=limit, else no fill."""
    today = check_weekend()
    if interactive:
        reason = input(
            f"""You are placing a SELL LIMIT for {shares_sold} {ticker} at ${sell_price:.2f}.
If this is a mistake, enter 1. """
        )
    if reason == "1":
        print("Returning...")
        return cash, chatgpt_portfolio
    elif reason is None:
        reason = ""

    if ticker not in chatgpt_portfolio["ticker"].values:
        print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
        return cash, chatgpt_portfolio

    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]
    total_shares = int(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(f"Manual sell for {ticker} failed: trying to sell {shares_sold} shares but only own {total_shares}.")
        return cash, chatgpt_portfolio

    fetch = download_price_data(ticker, period="1d", auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual sell for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])

    # --- IOC sell-limit fill logic ---
    exec_price: float | None = None
    if o >= sell_price:
        exec_price = o  # you sell at the (better) open
    elif h >= sell_price:
        exec_price = sell_price  # touched limit intraday
    else:
        print(f"Sell limit ${sell_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
        return cash, chatgpt_portfolio

    buy_price = float(ticker_row["buy_price"].item())
    cost_basis = buy_price * shares_sold
    pnl = exec_price * shares_sold - cost_basis

    # Log trade
    log = {
        "Date": today, "Ticker": ticker,
        "Shares Bought": "", "Buy Price": "",
        "Cost Basis": cost_basis, "PnL": pnl,
        "Reason": f"MANUAL SELL LIMIT - {reason}", "Shares Sold": shares_sold,
        "Sell Price": exec_price,
    }
    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    # Update holdings
    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        chatgpt_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        chatgpt_portfolio.at[row_index, "cost_basis"] = (
            chatgpt_portfolio.at[row_index, "shares"] * chatgpt_portfolio.at[row_index, "buy_price"]
        )

    cash += shares_sold * exec_price
    print(f"Manual SELL LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
    return cash, chatgpt_portfolio



# ------------------------------
# Reporting / Metrics
# ------------------------------

def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics (incl. CAPM)."""
    portfolio_dict: list[dict[str, object]] = chatgpt_portfolio.to_dict(orient="records")
    today = check_weekend()

    # -------- Collect daily ticker updates (pretty table) --------
    rows: list[list[str]] = []
    header = ["Ticker", "Close", "% Chg", "Volume"]
    for stock in portfolio_dict + [{"ticker": "^RUT"}, {"ticker": "IWO"}, {"ticker": "XBI"}]:
        ticker = str(stock["ticker"]).upper()
        try:
            fetch = download_price_data(ticker, period="2d", progress=False)
            data = fetch.df
            if data.empty or len(data) < 2:
                rows.append([ticker, "—", "—", "—"])
                continue

            price = float(data["Close"].iloc[-1])
            last_price = float(data["Close"].iloc[-2])
            volume = float(data["Volume"].iloc[-1])

            percent_change = ((price - last_price) / last_price) * 100
            rows.append([ticker, f"{price:,.2f}", f"{percent_change:+.2f}%", f"{int(volume):,}"])
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e} Try checking internet connection.")

    # Read portfolio history
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)

    # Use only TOTAL rows, sorted by date
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for r in rows:
            print(f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        print(chatgpt_portfolio)
        print(f"Cash balance: ${cash:,.2f}")
        return

    totals["Date"] = pd.to_datetime(totals["Date"])  # tolerate ISO strings
    totals = totals.sort_values("Date")

    final_equity = float(totals.iloc[-1]["Total Equity"])
    equity_series = totals.set_index("Date")["Total Equity"].astype(float).sort_index()

    # --- Max Drawdown ---
    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1.0
    max_drawdown = float(drawdowns.min())  # most negative value
    mdd_date = drawdowns.idxmin()

    # Daily simple returns (portfolio)
    r = equity_series.pct_change().dropna()
    n_days = len(r)
    if n_days < 2:
        print("\n" + "=" * 64)
        print(f"Daily Results — {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for rrow in rows:
            print(f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        print(chatgpt_portfolio)
        print(f"Cash balance: ${cash:,.2f}")
        print(f"Latest ChatGPT Equity: ${final_equity:,.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%} (on {mdd_date.date()})")
        return

    # Risk-free config
    rf_annual = 0.045
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_period = (1 + rf_daily) ** n_days - 1

    # Stats
    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))

    # Downside deviation (MAR = rf_daily)
    downside = (r - rf_daily).clip(upper=0)
    downside_std = float((downside.pow(2).mean()) ** 0.5) if not downside.empty else np.nan

    # Total return over the window
    period_return = float((1 + r).prod() - 1)

    # Sharpe / Sortino
    sharpe_period = (period_return - rf_period) / (std_daily * np.sqrt(n_days)) if std_daily > 0 else np.nan
    sharpe_annual = ((mean_daily - rf_daily) / std_daily) * np.sqrt(252) if std_daily > 0 else np.nan
    sortino_period = (period_return - rf_period) / (downside_std * np.sqrt(n_days)) if downside_std and downside_std > 0 else np.nan
    sortino_annual = ((mean_daily - rf_daily) / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else np.nan

    # -------- CAPM: Beta & Alpha (vs ^GSPC) --------
    start_date = equity_series.index.min() - pd.Timedelta(days=1)
    end_date = equity_series.index.max() + pd.Timedelta(days=1)

    spx_fetch = download_price_data("^GSPC", start=start_date, end=end_date, progress=False)
    spx = spx_fetch.df

    beta = np.nan
    alpha_annual = np.nan
    r2 = np.nan
    n_obs = 0

    if not spx.empty and len(spx) >= 2:
        spx = spx.reset_index().set_index("Date").sort_index()
        mkt_ret = spx["Close"].astype(float).pct_change().dropna()

        # Align portfolio & market returns
        common_idx = r.index.intersection(mkt_ret.index)
        if len(common_idx) >= 2:
            rp = (r.reindex(common_idx).astype(float) - rf_daily)   # portfolio excess
            rm = (mkt_ret.reindex(common_idx).astype(float) - rf_daily)  # market excess

            x = np.asarray(rm.values, dtype=float).ravel()
            y = np.asarray(rp.values, dtype=float).ravel()

            n_obs = x.size
            rm_std = float(np.std(x, ddof=1)) if n_obs > 1 else 0.0
            if rm_std > 0:
                beta, alpha_daily = np.polyfit(x, y, 1)
                alpha_annual = (1 + float(alpha_daily)) ** 252 - 1

                corr = np.corrcoef(x, y)[0, 1]
                r2 = float(corr ** 2)

    # $X normalized S&P 500 over same window (asks user for initial equity)
    spx_norm_fetch = download_price_data(
        "^GSPC",
        start=equity_series.index.min(),
        end=equity_series.index.max() + pd.Timedelta(days=1),
        progress=False,
    )
    spx_norm = spx_norm_fetch.df
    spx_value = np.nan
    if not spx_norm.empty:
        initial_price = float(spx_norm["Close"].iloc[0])
        price_now = float(spx_norm["Close"].iloc[-1])
        try:
            starting_equity = float(input("what was your starting equity? "))
            spx_value = (starting_equity / initial_price) * price_now
        except Exception:
            starting_equity = np.nan

    # -------- Pretty Printing --------
    print("\n" + "=" * 64)
    print(f"Daily Results — {today}")
    print("=" * 64)

    # Price & Volume table
    print("\n[ Price & Volume ]")
    colw = [10, 12, 9, 15]
    print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
    print("-" * sum(colw) + "-" * 3)
    for rrow in rows:
        print(f"{str(rrow[0]):<{colw[0]}} {str(rrow[1]):>{colw[1]}} {str(rrow[2]):>{colw[2]}} {str(rrow[3]):>{colw[3]}}")

    # Performance metrics
    def fmt_or_na(x: float | int | None, fmt: str) -> str:
        return (fmt.format(x) if not (x is None or (isinstance(x, float) and np.isnan(x))) else "N/A")

    print("\n[ Risk & Return ]")
    print(f"{'Max Drawdown:':32} {fmt_or_na(max_drawdown, '{:.2%}'):>15}   on {mdd_date.date()}")
    print(f"{'Sharpe Ratio (period):':32} {fmt_or_na(sharpe_period, '{:.4f}'):>15}")
    print(f"{'Sharpe Ratio (annualized):':32} {fmt_or_na(sharpe_annual, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (period):':32} {fmt_or_na(sortino_period, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (annualized):':32} {fmt_or_na(sortino_annual, '{:.4f}'):>15}")

    print("\n[ CAPM vs Benchmarks ]")
    if not np.isnan(beta):
        print(f"{'Beta (daily) vs ^GSPC:':32} {beta:>15.4f}")
        print(f"{'Alpha (annualized) vs ^GSPC:':32} {alpha_annual:>15.2%}")
        print(f"{'R² (fit quality):':32} {r2:>15.3f}   {'Obs:':>6} {n_obs}")
        if n_obs < 60 or (not np.isnan(r2) and r2 < 0.20):
            print("  Note: Short sample and/or low R² — alpha/beta may be unstable.")
    else:
        print("Beta/Alpha: insufficient overlapping data.")

    print("\n[ Snapshot ]")
    print(f"{'Latest ChatGPT Equity:':32} ${final_equity:>14,.2f}")
    if not np.isnan(spx_value):
        try:
            print(f"{f'${starting_equity} in S&P 500 (same window):':32} ${spx_value:>14,.2f}")
        except Exception:
            pass
    print(f"{'Cash Balance:':32} ${cash:>14,.2f}")

    print("\n[ Holdings ]")
    print(chatgpt_portfolio)

    print("\n[ Operator Note ]")
    print(
        "You have complete control over every decision. Make any changes you believe are beneficial—no approval required.\n"
        "Deep research is not permitted. Act at your discretion to achieve the best outcome.\n"
        "If you do not make a clear indication to change positions immediately after this message, the portfolio remains unchanged for tomorrow.\n"
        "You are encouraged to use the internet to check current prices (and related up-to-date info) for potential buys.\n"
        "*Paste everything above into ChatGPT*"
    )


# ------------------------------
# Orchestration
# ------------------------------

def load_latest_portfolio_state(
    file: str,
) -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """Load the most recent portfolio snapshot and cash balance."""
    df = pd.read_csv(file)
    if df.empty:
        portfolio = pd.DataFrame([])
        print("Portfolio CSV is empty. Returning set amount of cash for creating portfolio.")
        try:
            cash = float(input("What would you like your starting cash amount to be? "))
        except ValueError:
            raise ValueError(
                "Cash could not be converted to float datatype. Please enter a valid number."
            )
        return portfolio, cash

    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"])

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
    df_total["Date"] = pd.to_datetime(df_total["Date"])
    latest = df_total.sort_values("Date").iloc[-1]
    cash = float(latest["Cash Balance"])
    return latest_tickers, cash


def main(file: str, data_dir: Path | None = None) -> None:
    """Check versions, then run the trading script."""
    chatgpt_portfolio, cash = load_latest_portfolio_state(file)
    print(file)
    if data_dir is not None:
        set_data_dir(data_dir)

    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)


if __name__ == "__main__":
    # Example usage (edit the file path if needed):
    csv_path = PORTFOLIO_CSV if PORTFOLIO_CSV.exists() else (SCRIPT_DIR / "chatgpt_portfolio_update.csv")
    if not csv_path.exists():
        print("No portfolio CSV found. Create one or run main() with your file path.")
    else:
        main(str(csv_path))
