"""Headless + e‑kiri versioon ChatGPT Micro‑Cap skriptist.
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
