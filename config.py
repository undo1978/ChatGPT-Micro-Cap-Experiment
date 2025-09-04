from dataclasses import dataclass
import os


def _get_env_str(key: str, default: str | None = None) -> str | None:
v = os.getenv(key, default)
return v if v is not None and str(v).strip() != "" else None


@dataclass
class Settings:
EMAIL_FROM: str | None
EMAIL_TO: str | None
SMTP_HOST: str
SMTP_PORT: int
SMTP_USER: str | None
SMTP_PASS: str | None


TOTAL_PORTFOLIO_EUR: float | None
TRADE_BUDGET_EUR: float | None
STOP_PCT: float
STARTING_CASH: float


RISK_PERC: float
MAX_SPREAD_PCT: float
TRAILING_WINDOW: int


@classmethod
def load(cls) -> "Settings":
def _f(key: str, default: float) -> float:
v = os.getenv(key)
try:
return float(v) if v is not None else default
except Exception:
return default
def _i(key: str, default: int) -> int:
v = os.getenv(key)
try:
return int(v) if v is not None else default
except Exception:
return default


return cls(
EMAIL_FROM=_get_env_str("EMAIL_FROM"),
EMAIL_TO=_get_env_str("EMAIL_TO"),
SMTP_HOST=_get_env_str("SMTP_HOST", "smtp.gmail.com") or "smtp.gmail.com",
SMTP_PORT=_i("SMTP_PORT", 587),
SMTP_USER=_get_env_str("SMTP_USER"),
SMTP_PASS=_get_env_str("SMTP_PASS"),
TOTAL_PORTFOLIO_EUR=_f("TOTAL_PORTFOLIO_EUR", 0.0) or None,
TRADE_BUDGET_EUR=_f("TRADE_BUDGET_EUR", 0.0) or None,
STOP_PCT=_f("STOP_PCT", 0.10),
STARTING_CASH=_f("STARTING_CASH", 10000.0),
RISK_PERC=_f("RISK_PERC", 0.01),
MAX_SPREAD_PCT=_f("MAX_SPREAD_PCT", 5.0), # protsent
TRAILING_WINDOW=_i("TRAILING_WINDOW", 5),
)


CFG = Settings.load()
