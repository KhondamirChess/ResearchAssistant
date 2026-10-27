"""FastAPI backend. Run from the project root:  uvicorn backend.app:app --reload"""
import json
import logging
import threading
import time
import uuid
from typing import Iterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .agent import run_agent

# логирование настраиваем до создания приложения
config.setup_logging()
log = logging.getLogger("backend.api")

app = FastAPI(title="LLM-Based Agentic Research Assistant", version="0.1")

# ------------------------------------------------------------- history storage
_lock = threading.Lock()
MAX_HISTORY = 50


def _load() -> list[dict]:
    try:
        return json.loads(config.HISTORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return []


def _save(items: list[dict]) -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    config.HISTORY_FILE.write_text(json.dumps(items[:MAX_HISTORY], ensure_ascii=False),
                                   encoding="utf-8")


# ------------------------------------------------------------------- endpoints
class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["demo", "live"] = "demo"
    max_steps: int = Field(6, ge=1, le=10)
    lang: Literal["en", "ko"] = "en"


@app.get("/api/health")
def health():
    return {"live_available": bool(config.ANTHROPIC_API_KEY),
            "search_available": bool(config.TAVILY_API_KEY),
            "model": config.MODEL}


@app.post("/api/research")
def research(req: ResearchRequest):
    if req.mode == "live":
        if not config.ANTHROPIC_API_KEY:
            log.warning("live mode rejected: no ANTHROPIC_API_KEY")
            raise HTTPException(400, "Live mode needs ANTHROPIC_API_KEY in .env. "
                                     "Use Demo mode or add the key and restart the server.")
        if not config.TAVILY_API_KEY:
            log.warning("live mode rejected: no TAVILY_API_KEY")
            raise HTTPException(400, "Live mode needs TAVILY_API_KEY in .env (free key at "
                                     "app.tavily.com). Restart the server after adding it.")

    # вопрос обрезаем, чтобы не раздувать лог
    log.info("research start mode=%s max_steps=%d lang=%s q=%r",
             req.mode, req.max_steps, req.lang, req.question.strip()[:80])

    def stream() -> Iterator[str]:
        steps, report, metrics, failed = [], None, None, False
        for ev in run_agent(req.question.strip(), req.mode, req.max_steps, req.lang):
            if ev["type"] == "step":
                steps.append(ev)
            elif ev["type"] == "report":
                report = ev["report"]
            elif ev["type"] == "metrics":
                metrics = ev
            elif ev["type"] == "error":
                failed = True
                log.error("research failed: %s", ev.get("message"))
            yield json.dumps(ev, ensure_ascii=False) + "\n"
        # итог запроса: успех, число шагов, токены, время
        log.info("research done ok=%s steps=%d metrics=%s",
                 bool(report and not failed), len(steps), metrics)
        if report and not failed:
            item = {"id": uuid.uuid4().hex[:10], "question": req.question.strip(),
                    "mode": req.mode, "lang": req.lang, "created_at": int(time.time()), "steps": steps,
                    "report": report, "metrics": metrics}
            with _lock:
                _save([item] + _load())

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/history")
def history():
    return [{k: i[k] for k in ("id", "question", "mode", "created_at")} for i in _load()]


@app.get("/api/history/{item_id}")
def history_item(item_id: str):
    for i in _load():
        if i["id"] == item_id:
            return i
    raise HTTPException(404, "not found")


@app.delete("/api/history")
def clear_history():
    with _lock:
        _save([])
    log.info("history cleared")
    return {"ok": True}


# the web page itself (must be mounted last)
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
