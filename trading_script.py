from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import os, argparse, io, logging, warnings
from contextlib import redirect_stdout


import numpy as np
import pandas as pd
import yfinance as yf


from config import CFG
from email_utils import send_email_html
from utils import retry, write_csv_atomic, recos_to_html, df_to_html, dedupe_recos


# ---- valikuline: turupäevad ----
try:
import pandas_market_calendars as mcal
_HAS_MCAL = True
_XNYS = mcal.get_calendar('XNYS')
except Exception:
_HAS_MCAL = False


# ---------------- Args ----------------


def parse_args():
p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true")
p.add_argument("--allocation", type=float, default=0.10)
p.add_argument("--stop_pct", type=float, default=CFG.STOP_PCT)
p.add_argument("--sizing", choices=["allocation", "risk"], default="allocation")
p.add_argument("--risk_perc", type=float, default=CFG.RISK_PERC)
p.add_argument("--max_spread_pct", type=float, default= CFG.MAX_SPREAD_PCT)
p.add_argument("--trailing_window", type=int, default=CFG.TRAILING_WINDOW)
return p.parse_args()


_args = None

# ---------------- Date helpers ----------------
if ticker in STOOQ_BLOCKLIST:
return pd.DataFrame()
t = STOOQ_MAP.get(ticker, ticker)
if not t.startswith("^"):
t = t.lower()
try:
df = cast(pd.DataFrame, pdr.DataReader(t, "stooq", start=start, end=end))
return df.sort_index()
except Exception:
return pd.DataFrame()




def _weekend_safe_range(period: str | None, start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
if start or end:
end_ts = pd.Timestamp(end) if end else last_trading_date() + pd.Timedelta(days=1)
start_ts = pd.Timestamp(start) if start else (end_ts - pd.Timedelta(days=5))
return start_ts.normalize(), pd.Timestamp(end_ts).normalize()
days = int(period[:-1]) if isinstance(period, str) and period.endswith("d") else 1
end_trading = last_trading_date()
start_ts = (end_trading - pd.Timedelta(days=days)).normalize()
end_ts = (end_trading + pd.Timedelta(days=1)).normalize()
return start_ts, end_ts




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


df_s = _stooq_download(ticker, start=s, end=e)
if isinstance(df_s, pd.DataFrame) and not df_s.empty:
return FetchResult(_normalize_ohlcv(_to_datetime_index(df_s)), "stooq-pdr")


df_csv = _stooq_csv_download(ticker, s, e)
if isinstance(df_csv, pd.DataFrame) and not df_csv.empty:
return FetchResult(_normalize_ohlcv(_to_datetime_index(df_csv)), "stooq-csv")


proxy_map = {"^GSPC": "SPY", "^RUT": "IWM"}
proxy = proxy_map.get(ticker)
if proxy:
df_proxy = _yahoo_download(proxy, start=s, end=e, **kwargs)
if isinstance(df_proxy, pd.DataFrame) and not df_proxy.empty:
return FetchResult(_normalize_ohlcv(_to_datetime_index(df_proxy)), f"yahoo:{proxy}-proxy")


empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
return FetchResult(empty, "empty")

# ---------------- Paths ----------------
shares = float(input("Enter number of shares: "))
if shares <= 0:
raise ValueError
except ValueError:
print("Invalid share amount. Buy cancelled.")
continue
if order_type == "m":
try:
stop_loss = float(input("Enter stop loss (or 0 to skip): "))
if stop_loss < 0:
raise ValueError
except ValueError:
print("Invalid stop loss. Buy cancelled.")
continue
add_reco("BUY", ticker, reason=f"interactive: order={order_type}", target_shares=int(shares), stop=stop_loss)
# MOO täitmine (lühendatult)
s, e = trading_day_window()
fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
data = fetch.df
if data.empty:
print(f"MOO buy for {ticker} failed: no market data available (source={fetch.source}).")
continue
o = float(data.get("Open", data["Close"]).iloc[-1])
exec_price = round(o, 2)
notional = exec_price * float(shares)
if notional > cash:
print(f"MOO buy for {ticker} failed: cost {notional:.2f} exceeds cash {cash:.2f}.")
continue
# append position
rows = portfolio_df.loc[portfolio_df.get("ticker", pd.Series(dtype=str)).astype(str).str.upper() == ticker.upper()]
if rows.empty:
new_trade = {"ticker": ticker, "shares": float(shares), "stop_loss": float(stop_loss), "buy_price": float(exec_price), "cost_basis": float(notional)}
portfolio_df = pd.concat([portfolio_df, pd.DataFrame([new_trade])], ignore_index=True) if not portfolio_df.empty else pd.DataFrame([new_trade])
else:
idx = rows.index[0]
cur_shares = float(portfolio_df.at[idx, "shares"]) if "shares" in portfolio_df.columns else 0.0
cur_cost = float(portfolio_df.at[idx, "cost_basis"]) if "cost_basis" in portfolio_df.columns else 0.0
new_shares = cur_shares + float(shares)
new_cost = cur_cost + float(notional)
avg_price = new_cost / new_shares if new_shares else 0.0
portfolio_df.at[idx, "shares"] = new_shares
portfolio_df.at[idx, "cost_basis"] = new_cost
portfolio_df.at[idx, "buy_price"] = avg_price
portfolio_df.at[idx, "stop_loss"] = float(stop_loss)
cash -= notional
continue
if action == "s":
# jäta müük interaktiivseks (võime hiljem headlessiks teha)
...
# while lõpp


# ------- Daily pricing + stop-loss + trailing stop -------

s, e = trading_day_window()
total_value += value
total_pnl += pnl
row = {"Date": today_iso, "Ticker": ticker, "Shares": shares, "Buy Price": cost, "Cost Basis": cost_basis, "Stop Loss": stop, "Current Price": price, "Total Value": value, "PnL": pnl, "Action": action, "Cash Balance": "", "Total Equity": ""}
results.append(row)


total_row = {"Date": today_iso, "Ticker": "TOTAL", "Shares": "", "Buy Price": "", "Cost Basis": "", "Stop Loss": "", "Current Price": "", "Total Value": round(total_value, 2), "PnL": round(total_pnl, 2), "Action": "", "Cash Balance": round(cash, 2), "Total Equity": round(total_value + cash, 2)}
results.append(total_row)


df_out = pd.DataFrame(results)
if Path(PORTFOLIO_CSV).exists():
existing = pd.read_csv(PORTFOLIO_CSV)
existing = existing[existing["Date"] != str(today_iso)]
df_out = pd.concat([existing, df_out], ignore_index=True)
write_csv_atomic(PORTFOLIO_CSV, df_out)


return portfolio_df, cash


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
latest_tickers.drop(columns=["Date", "Cash Balance", "Total Equity", "Action", "Current Price", "PnL", "Total Value"], inplace=True, errors="ignore")
latest_tickers.rename(columns={"Cost Basis": "cost_basis", "Buy Price": "buy_price", "Shares": "shares", "Ticker": "ticker", "Stop Loss": "stop_loss"}, inplace=True)
latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient="records")
df_total = df[df["Ticker"] == "TOTAL"].copy()
df_total["Date"] = pd.to_datetime(df_total["Date"], format="mixed", errors="coerce")
latest = df_total.sort_values("Date").iloc[-1]
cash = float(latest["Cash Balance"])
return latest_tickers, cash

# ---------------- Main ----------------
_args = parse_args()
asof_env = os.environ.get("ASOF_DATE")
if asof_env:
set_asof(asof_env)


SCRIPT_DIR = Path(__file__).resolve().parent
PORTFOLIO_CSV = SCRIPT_DIR / "chatgpt_portfolio_update.csv"


chatgpt_portfolio, cash = load_latest_portfolio_state(str(PORTFOLIO_CSV))
if isinstance(chatgpt_portfolio, (list, dict)):
chatgpt_portfolio = pd.DataFrame(chatgpt_portfolio)


# HEADLESS sisseost (demoks): kui on HEADLESS_TICKER, proovi osta mõistliku kontrolliga
if _args.headless:
ticker = os.getenv("HEADLESS_TICKER", "").strip().upper()
if ticker:
# likviidsus/spread filter
s, e = trading_day_window()
fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
df = fetch.df
if not df.empty:
close = float(df["Close"].iloc[-1]); high = float(df["High"].iloc[-1]); low = float(df["Low"].iloc[-1]); vol = float(df["Volume"].iloc[-1])
spread_pct = (high - low) / close * 100 if close else 999.0
if spread_pct <= _args.max_spread_pct and vol > 0:
if _args.sizing == "risk":
# equity = cash + lahtiste positsioonide väärtus (ligikaudne)
equity = cash
for _, row in chatgpt_portfolio.iterrows():
equity += float(row.get("shares", 0) or 0) * float(row.get("buy_price", 0) or 0)
shares = shares_by_risk(ticker, _args.risk_perc, _args.stop_pct, equity, cash)
else:
shares = shares_by_allocation(ticker, _args.allocation, cash)
stop = round(close * (1.0 - _args.stop_pct), 4)
if shares > 0:
add_reco("BUY", ticker, reason=f"headless auto-buy {ticker}", target_shares=shares, stop=stop)
else:
add_reco("HOLD", ticker, reason=f"skipped: spread {spread_pct:.2f}% > {_args.max_spread_pct}% või volume=0")


# Põhiloogika
chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash, interactive=(not _args.headless))


# Meil: HTML kokkuvõte + manusena CSV
recos = dedupe_recos(RECOMMENDATIONS)
html = (
f"<h3>Päevane portfellisoovitus</h3>"
f"<p>Kuupäev: {last_trading_date().date().isoformat()}</p>"
+ recos_to_html(recos)
+ "<h4>Portfelli snapshot</h4>" + df_to_html(chatgpt_portfolio)
+ "<p>Märkused: micro-cap volatiilsus, slippage, likviidsus; info on hariduslik, mitte investeerimisnõu.</p>"
)
try:
if _args.headless:
send_email_html("Päevane portfellisoovitus", html, text=None, attachments=[str(PORTFOLIO_CSV)])
except Exception as e:
logging.exception("Email send failed: %s", e)
