import os


class CFG:
    # Vaikimisi väärtused (loe keskkonnast või kasuta default’i)
    STOP_PCT = float(os.getenv("STOP_PCT", "0.1"))              # 10% stop-loss
    RISK_PERC = float(os.getenv("RISK_PERC", "0.01"))           # 1% risk per trade
    MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "5.0"))  # max 5% intraday spread
    TRAILING_WINDOW = int(os.getenv("TRAILING_WINDOW", "5"))    # trailing stop akna pikkus
    STARTING_CASH = float(os.getenv("STARTING_CASH", "10000.0"))


def get(key: str, default=None):
    """Abifunktsioon env-muutujate lugemiseks"""
    return os.getenv(key, default)
