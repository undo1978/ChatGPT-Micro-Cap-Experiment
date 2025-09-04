"""Headless + e‑kiri versioon ChatGPT Micro‑Cap skriptist.


- Lisab CLI lipud: --headless, --allocation, --stop_pct
- Väldib interaktiivseid input()‑e headless‑režiimis
- Koostab jooksu lõpus soovituste kokkuvõtte ja saadab e‑kirja


NB! See fail sisaldab hinnatõmbamist (yfinance + Stooq fallback), portfelli
protsessimist, päevatulemuste printimist ning e‑kirja saatmist.
"""
from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import os
import warnings
import argparse
import json
import logging
import io
from contextlib import redirect_stdout

import numpy as np
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"


# ---------------- Recommendations ----------------
RECOMMENDATIONS: List[Dict[str, Any]] = []


def add_reco(action: str, ticker: str, reason: str = "", target_shares: int = 0, stop: float = 0.0):
RECOMMENDATIONS.append({
"action": action.upper(),
"ticker": ticker,
"reason": reason,
"target_shares": int(target_shares),
"stop": float(stop),
})


def format_reco(recos: List[Dict[str, Any]]) -> str:
if not recos:
return "(täna ettepanekuid ei tekkinud)"
lines = []
for r in recos:
lines.append(f"{r['action']:<5} {r['ticker']} kogus≈{r.get('target_shares','?')} SL={r.get('stop','-')} — {r.get('reason','')}")
return "\n".join(lines)


# ---------------- Helpers ----------------


def get_last_close_price(ticker: str) -> float:
try:
data = yf.Ticker(ticker).history(period="5d")
if not data.empty:
return float(data["Close"].iloc[-1])
except Exception:
pass
return 0.0




def compute_stop_default(ticker: str, stop_pct: float) -> float:
px = get_last_close_price(ticker)
return round(px * (1.0 - stop_pct), 4) if px > 0 else 0.0




def get_target_shares(ticker: str, allocation: float) -> int:
total = float(os.getenv("TOTAL_PORTFOLIO_EUR", "0") or "0")
budget = float(os.getenv("TRADE_BUDGET_EUR", "0") or "0")
if total > 0:
target_value = total * allocation
elif budget > 0:
target_value = budget
else:
target_value = 200.0
px = get_last_close_price(ticker)
if px <= 0:
return 0
return max(0, int(target_value // px))

# ---------------- Portfolio ops ----------------
if data.empty:
print(f"Manual sell for {ticker} failed: no market data available (source={fetch.source}).")
continue
o = float(data.get("Open", data["Close"]).iloc[-1])
h = float(data["High"].iloc[-1])
l = float(data["Low"].iloc[-1])
if o >= sell_price:
exec_price = o
elif h >= sell_price:
exec_price = sell_price
else:
print(f"Sell limit ${sell_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
continue
# Update portfolio
rows = portfolio_df.loc[portfolio_df.get("ticker", pd.Series(dtype=str)).astype(str).str.upper() == ticker.upper()]
if rows.empty:
print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
continue
idx = rows.index[0]
cur_shares = float(portfolio_df.at[idx, "shares"]) if "shares" in portfolio_df.columns else 0.0
if shares > cur_shares:
print(f"Manual sell for {ticker} failed: trying to sell {shares} > {cur_shares}.")
continue
buy_price = float(portfolio_df.at[idx, "buy_price"]) if "buy_price" in portfolio_df.columns else 0.0
pnl = (exec_price - buy_price) * shares
cash += exec_price * shares
new_shares = cur_shares - shares
if new_shares == 0:
portfolio_df = portfolio_df.drop(index=idx)
else:
portfolio_df.at[idx, "shares"] = new_shares
portfolio_df.at[idx, "cost_basis"] = new_shares * buy_price
log = {"Date": today_iso, "Ticker": ticker, "Shares Sold": shares, "Sell Price": exec_price, "Cost Basis": buy_price * shares, "PnL": pnl, "Reason": "MANUAL SELL LIMIT"}
if os.path.exists(TRADE_LOG_CSV):
df_log = pd.read_csv(TRADE_LOG_CSV)
df_log = pd.concat([df_log, pd.DataFrame([log])], ignore_index=True) if not df_log.empty else pd.DataFrame([log])
else:
df_log = pd.DataFrame([log])
df_log.to_csv(TRADE_LOG_CSV, index=False)
print(f"Manual SELL LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
continue
# while lõpp


# ------- Daily pricing + stop-loss execution -------

s, e = trading_day_window()
df_out.to_csv(PORTFOLIO_CSV, index=False)


return portfolio_df, cash


# ---------------- Reporting (lühem) ----------------


def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
today = check_weekend()
print("\n" + "=" * 64)
print(f"Daily Results — {today}")
print("=" * 64)
print("\n[ Holdings ]")
print(chatgpt_portfolio)
print(f"\nCash balance: ${cash:,.2f}")


# ---------------- Load/save ----------------


def load_latest_portfolio_state(file: str) -> tuple[pd.DataFrame | List[Dict[str, Any]], float]:
if not Path(file).exists():
# alusta tühjalt 10k sularahaga (või küsi interaktiivselt kui mitte headless)
start_cash = float(os.getenv("STARTING_CASH", "10000"))
portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
return portfolio, start_cash
df = pd.read_csv(file)
if df.empty:
start_cash = float(os.getenv("STARTING_CASH", "10000"))
portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
return portfolio, start_cash
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
if __name__ == "__main__":
_args = parse_args()
asof_env = os.environ.get("ASOF_DATE")
if asof_env:
set_asof(asof_env)


csv_path = PORTFOLIO_CSV if PORTFOLIO_CSV.exists() else (SCRIPT_DIR / "chatgpt_portfolio_update.csv")
chatgpt_portfolio, cash = load_latest_portfolio_state(str(csv_path))
if isinstance(chatgpt_portfolio, (list, dict)):
chatgpt_portfolio = _ensure_df(chatgpt_portfolio)
chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash, interactive=(not _args.headless))
daily_results(chatgpt_portfolio, cash)


if _args.headless:
buf = io.StringIO()
with redirect_stdout(buf):
print("Soovitatud tehingud:")
print(format_reco(RECOMMENDATIONS))
body = buf.getvalue()
body += "\n\nMärkused: micro-cap volatiilsus, slippage, likviidsus; käsitle infot õppematerjalina.\n"
send_email("Päevane portfellisoovitus", body)
