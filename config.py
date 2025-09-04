import os


def _as_float(name: str, default: float) -> float:
    val = os.getenv(name, "").strip()
    try:
        return float(val) if val else default
    except ValueError:
        return default


def _as_int(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    try:
        return int(val) if val else default
    except ValueError:
        return default


class CFG:
    STOP_PCT = _as_float("STOP_PCT", 0.1)            # 10% stop-loss
    RISK_PERC = _as_float("RISK_PERC", 0.01)         # 1% risk
    MAX_SPREAD_PCT = _as_float("MAX_SPREAD_PCT", 5.0)
    TRAILING_WINDOW = _as_int("TRAILING_WINDOW", 5)
    STARTING_CASH = _as_float("STARTING_CASH", 10000.0)


def get(key: str, default=None):
    """Abifunktsioon env-muutujate lugemiseks"""
    val = os.getenv(key)
    return val if val not in (None, "") else default
