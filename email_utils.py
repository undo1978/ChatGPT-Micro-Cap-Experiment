# Micro-cap agent: tasuta GitHub Actions + e-kiri (täisjuhend)

**Eesmärk:** käivitame GitHubis (tasuta) igapäevase töö, mis jooksutab repo *ChatGPT-Micro-Cap-Experiment* skripti "headless"-režiimis ja **saadab sulle e-kirja** portfelli soovitustega.

---

## 0) Eeldused

* **E‑post:** Gmail/Google Workspace (soovituslik). Vaja on **App Password**‑it (kui 2‑astmeline kinnitamine on sees). Alternatiiv: SendGrid/Postmark vms (tasuta tase olemas).
* **OpenAI API võti** (kui skript seda kasutab).
* GitHubi konto (teeme all samm‑sammult).

> Kui App Password on uus teema, vaata juhendi lõpus lisasid (Gmail App Password).

---

## 1) Loo GitHubi konto

1. Mine **github.com** ja vali **Sign up**.
2. Lõpeta registreerimine (e‑posti kinnitamine jne).

---

## 2) Tee repo „fork” (enda koopia)

1. Ava repo: `https://github.com/LuckyOne7777/ChatGPT-Micro-Cap-Experiment`.
2. Üleval paremal klõpsa **Fork** → loo fork oma kasutaja alla.
3. Pärast fork’i oled **oma koopia** vaates (URL algab `https://github.com/<sinu_kasutaja>/ChatGPT-Micro-Cap-Experiment`).

---

## 3) Lisa e‑posti abifail `email_utils.py`

1. Oma forkis klõpsa **Add file → Create new file**.
2. Failinimi: `email_utils.py`
3. Kleepi sisse:

```python
# email_utils.py
import os, smtplib
from email.mime.text import MIMEText

def send_email(subject: str, body: str):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    sender = os.getenv("EMAIL_FROM")
    to = os.getenv("EMAIL_TO")

    if not all([host, port, user, pwd, sender, to]):
        raise RuntimeError("Email env muutujad puuduvad")

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(sender, [to], msg.as_string())
```

4. All **Commit changes** → jäta vaikimisi „Commit directly to the `main` branch” → **Commit**.

---

## 4) Muuda skripti „headless“ + koosta kokkuvõte ja saada meil

> Mõte: lülitame välja interaktiivsed `input()` küsimused ja saadame jooksu lõpus kokkuvõtte e‑posti. All on turvaline ja minimaalne muudatus, mille saab hiljem täpsemaks timmida.

### 4.1 Ava põhiskript

1. Repo failide loendis leia `trading_script.py` (või analoogne põhikäivitusfail).
2. Ava ja klõpsa paremal üleval **Edit** (pliiatsi ikoon).

### 4.2 Lisa argumendid ja abifunktsioonid faili algusesse

Kleebi ülaossa (otse importide järele):

```python
import os, argparse, builtins, math
from contextlib import redirect_stdout
import io
from email_utils import send_email

try:
    import yfinance as yf
except Exception:
    yf = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Run without interactive prompts")
    p.add_argument("--allocation", type=float, default=0.10,
                   help="Max osakaal ühest tickerist (nt 0.10 = 10% portfellist)")
    p.add_argument("--stop_pct", type=float, default=float(os.getenv("STOP_PCT", "0.1")),
                   help="Stop-loss protsent, nt 0.1 = 10%")
    return p.parse_args()

_args = None

def get_last_close_price(ticker: str) -> float:
    if yf is None:
        return 0.0
    try:
        data = yf.Ticker(ticker).history(period="5d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0

def compute_stop_default(ticker: str, stop_pct: float) -> float:
    price = get_last_close_price(ticker)
    return round(price * (1.0 - stop_pct), 4) if price > 0 else 0.0

def get_target_shares(ticker: str, allocation: float) -> int:
    # Eelistus: TOTAL_PORTFOLIO_EUR või TRADE_BUDGET_EUR keskkonnamuutuja
    total = float(os.getenv("TOTAL_PORTFOLIO_EUR", "0") or "0")
    budget = float(os.getenv("TRADE_BUDGET_EUR", "0") or "0")
    if total > 0:
        target_value = total * allocation
    elif budget > 0:
        target_value = budget
    else:
        target_value = 200.0  # vaikimisi
    price = get_last_close_price(ticker)
    if price <= 0:
        return 0
    return max(0, int(target_value // price))

# Kogume soovitused siia; lisa sinna, kus otsused tekivad
RECOMMENDATIONS = []  # elemendid: {action, ticker, reason, target_shares, stop}

def add_reco(action: str, ticker: str, reason: str = "", target_shares: int = 0, stop: float = 0.0):
    RECOMMENDATIONS.append({
        "action": action.upper(),
        "ticker": ticker,
        "reason": reason,
        "target_shares": target_shares,
        "stop": stop,
    })


def format_reco(recos):
    lines = []
    for r in recos:
        line = f"{r['action']:<5} {r['ticker']}  kogus≈{r.get('target_shares','?')}  SL={r.get('stop','-')} — {r.get('reason','')}"
        lines.append(line)
    return "\n".join(lines) if lines else "(täna ettepanekuid ei tekkinud)"
```

### 4.3 Tee `input()`-kohad mitteinteraktiivseks

Leia failist kohad, kus küsitakse nt:

```python
shares = int(input("Enter number of shares: "))
stop = float(input("Enter stop loss (or 0 to skip): "))
```

Asenda need plokiga (kasuta tegelikku muutuja `ticker` nime, mis on sinu koodis saadaval):

```python
if _args is None:
    _args = parse_args()

if _args.headless:
    shares = get_target_shares(ticker, allocation=_args.allocation)
    stop = compute_stop_default(ticker, stop_pct=_args.stop_pct)
else:
    shares = int(input("Enter number of shares: "))
    stop = float(input("Enter stop loss (or 0 to skip): "))
```

**Lisa soovituse salvestamine** sinna, kus teed ostu/müügi otsuse (või kohe pärast ülaltoodud `shares/stop` määramist):

```python
add_reco("BUY", ticker, reason="signaal vastab kriteeriumile", target_shares=shares, stop=stop)
# või SELL/HOLD vastavalt sinu loogikale
```

### 4.4 Saada jooksu lõpus meil

Enne kui skript lõpetab, lisa lõppu (või pärast peamist analüüsi):

```python
if _args is None:
    _args = parse_args()

if _args.headless:
    # püüame ka tavalise väljundi kinni, et lisada e‑kirja
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Soovitatud tehingud:")
        print(format_reco(RECOMMENDATIONS))
    body = buf.getvalue()

    # lisa vabatekst: riskid jm
    body += "\n\nMärkused: micro-cap volatiilsus, slippage, likviidsus; käsitle infot õppematerjalina.\n"

    send_email("Päevane portfellisoovitus", body)
```

**Commit** muudatused (green "Commit changes").

> Kui `trading_script.py` nimes erineb, kohanda vastavalt.

---

## 5) Lisa salajased võtmed (Secrets)

1. Repo ülaosas **Settings** → vasakult **Secrets and variables → Actions**.
2. **New repository secret** ja sisesta allolevad (üks haaval):

   * `OPENAI_API_KEY` *(kui skript seda kasutab)*
   * `SMTP_HOST` = `smtp.gmail.com`
   * `SMTP_PORT` = `587`
   * `SMTP_USER` = *sinu e‑post* (nt `andreas@grata.ee`)
   * `SMTP_PASS` = *Gmail App Password* (või SMTP teenuse API võti)
   * `EMAIL_FROM` = *sama mis kasutad saatjana*
   * `EMAIL_TO` = *kuhu soovid kokkuvõtte saada* (võid sama panna)
   * **Valikuline:** `TOTAL_PORTFOLIO_EUR` (nt `10000`) või `TRADE_BUDGET_EUR` (nt `200`) ja `STOP_PCT` (nt `0.1`)

---

## 6) Loo töövoog `.github/workflows/daily.yml`

1. Repo vaates **Add file → Create new file**.
2. Failinimi: `.github/workflows/daily.yml` (tähele: kaustad kaasaarvatud).
3. Kleepi sisse:

```yaml
name: daily-microcap
on:
  schedule:
    - cron: "5 15 * * *"   # 18:05 EEST = 15:05 UTC (kohanda soovi järgi)
  workflow_dispatch:        # käsitsi käivitamiseks

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m venv .venv
          . .venv/bin/activate
          pip install -r requirements.txt

      - name: Run headless + email summary
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_PORT: ${{ secrets.SMTP_PORT }}
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
          EMAIL_FROM: ${{ secrets.EMAIL_FROM }}
          EMAIL_TO:   ${{ secrets.EMAIL_TO }}
          TOTAL_PORTFOLIO_EUR: ${{ secrets.TOTAL_PORTFOLIO_EUR }}
          TRADE_BUDGET_EUR: ${{ secrets.TRADE_BUDGET_EUR }}
          STOP_PCT: ${{ secrets.STOP_PCT }}
        run: |
          . .venv/bin/activate
          python trading_script.py --headle
```
