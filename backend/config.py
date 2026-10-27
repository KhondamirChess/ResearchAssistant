"""Settings loaded from .env (see .env.example)."""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001").strip()
PRICE_IN = float(os.getenv("PRICE_PER_MTOK_INPUT", "0") or 0)
PRICE_OUT = float(os.getenv("PRICE_PER_MTOK_OUTPUT", "0") or 0)

DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"
HISTORY_FILE = DATA_DIR / "history.json"
LOG_DIR = ROOT / "logs"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()


def setup_logging() -> None:
    """Настраивает логгер "backend": вывод в консоль и в logs/app.log."""
    logger = logging.getLogger("backend")
    if logger.handlers:  # уже настроен (например, повторный импорт при --reload)
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    # файл ротируется: максимум 1 МБ, храним 3 старые копии
    file_h = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000,
                                 backupCount=3, encoding="utf-8")
    console_h = logging.StreamHandler()
    for h in (file_h, console_h):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False  # не дублируем сообщения через root-логгер uvicorn
