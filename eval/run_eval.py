"""Прогон агента по eval/questions.csv и подсчёт метрик.

Запуск из корня проекта:
    .venv/bin/python -m eval.run_eval              # все 30 вопросов
    .venv/bin/python -m eval.run_eval --limit 3    # первые 3 (быстрая проверка)

Результаты пишутся в eval/results/<время>/:
    runs.json    - полные данные по каждому вопросу (шаги, отчёт, оценка)
    runs.csv     - одна строка на вопрос
    summary.md   - итоговая таблица по категориям
"""
import argparse
import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import anthropic

from backend import config
from backend.agent import run_agent

log = logging.getLogger("backend.eval")

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.csv"

# цены модели в $ за 1M токенов: claude-haiku-4-5 = $1/$5, у Gemini бесплатный уровень = 0;
# можно переопределить аргументами --price-in / --price-out
DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT = (0.0, 0.0) if config.PROVIDER == "gemini" else (1.0, 5.0)

JUDGE_TOOL = {
    "name": "grade",
    "description": "Report which expected points are covered by the findings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "covered": {
                "type": "array",
                "description": "One boolean per expected point, in the same order.",
                "items": {"type": "boolean"},
            }
        },
        "required": ["covered"],
    },
}

JUDGE_PROMPT = """You are grading a research report.

Question: {question}

Findings in the report:
{findings}

Expected points:
{points}

For each expected point, decide whether the findings state it (same meaning is enough, \
exact wording is not needed). Call the grade tool with one boolean per expected point."""


def load_questions(limit: int | None) -> list[dict]:
    with QUESTIONS.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def run_one(row: dict, max_steps: int) -> dict:
    """Запускает агента на одном вопросе и собирает события."""
    steps, report, metrics, error = [], None, None, None
    for ev in run_agent(row["question"], max_steps, "en"):
        if ev["type"] == "step":
            steps.append(ev)
        elif ev["type"] == "report":
            report = ev["report"]
        elif ev["type"] == "metrics":
            metrics = ev
        elif ev["type"] == "error":
            error = ev["message"]
    return {**row, "steps": steps, "report": report, "metrics": metrics, "error": error,
            "budget_hit": any(s["phase"] == "reflect" and s["title"] == "Step budget reached"
                              for s in steps)}


def judge(client, run: dict) -> dict:
    """LLM-судья: какие из expected_points есть в отчёте."""
    points = [p.strip() for p in run["expected_points"].split(";") if p.strip()]
    findings = (run["report"] or {}).get("findings", [])
    if not findings:
        return {"covered": [False] * len(points), "judge_tokens": 0}
    prompt = JUDGE_PROMPT.format(
        question=run["question"],
        findings="\n".join(f"- {f}" for f in findings),
        points="\n".join(f"{i + 1}. {p}" for i, p in enumerate(points)))
    if config.PROVIDER == "gemini":
        # формат OpenAI: инструмент оборачиваем в function, ответ - строка JSON
        resp = client.chat.completions.create(
            model=config.MODEL, max_tokens=500, tool_choice="required",
            tools=[{"type": "function", "function": {
                "name": JUDGE_TOOL["name"], "description": JUDGE_TOOL["description"],
                "parameters": JUDGE_TOOL["input_schema"]}}],
            messages=[{"role": "user", "content": prompt}])
        tc = resp.choices[0].message.tool_calls[0]
        covered = json.loads(tc.function.arguments).get("covered", [])
        tokens = resp.usage.total_tokens if resp.usage else 0
    else:
        resp = client.messages.create(
            model=config.MODEL, max_tokens=500, tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "grade"},
            messages=[{"role": "user", "content": prompt}])
        covered = next(b.input for b in resp.content if b.type == "tool_use").get("covered", [])
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
    # судья мог вернуть список другой длины: выравниваем по числу пунктов
    covered = (list(covered) + [False] * len(points))[:len(points)]
    return {"covered": covered, "judge_tokens": tokens}


def summarize(runs: list[dict], price_in: float, price_out: float) -> str:
    """Таблица по категориям в markdown."""
    def stats(rs):
        ok = [r for r in rs if r["report"] and not r["error"]]
        m = [r["metrics"] for r in ok if r["metrics"]]
        cost = [(x["input_tokens"] * price_in + x["output_tokens"] * price_out) / 1e6 for x in m]
        cov = [r["coverage"] for r in rs]
        return {
            "n": len(rs),
            "success": f"{len(ok)}/{len(rs)}",
            "coverage": f"{mean(cov) * 100:.0f}%" if cov else "-",
            "latency": f"{mean(x['latency_s'] for x in m):.1f} s" if m else "-",
            "calls": f"{mean(x['steps'] for x in m):.1f}" if m else "-",
            "tokens": f"{mean(x['input_tokens'] + x['output_tokens'] for x in m):,.0f}" if m else "-",
            "cost": f"${mean(cost):.4f}" if cost else "-",
            "total_cost": sum(cost),
            "findings": f"{mean(len(r['report']['findings']) for r in ok):.1f}" if ok else "-",
            "budget": sum(r["budget_hit"] for r in rs),
        }

    groups = {}
    for r in runs:
        groups.setdefault(r["category"], []).append(r)
    cols = list(groups) + ["all"]
    data = {c: stats(groups[c]) for c in groups}
    data["all"] = stats(runs)
    rows = [("Questions", "n"), ("Successful reports", "success"),
            ("Coverage of expected points", "coverage"), ("Avg findings per report", "findings"),
            ("Avg latency", "latency"), ("Avg tool calls", "calls"), ("Avg tokens", "tokens"),
            ("Avg cost per question", "cost"), ("Step budget reached", "budget")]
    out = [f"# Evaluation results\n\nModel: `{config.MODEL}` · {len(runs)} questions · "
           f"total agent cost ≈ ${data['all']['total_cost']:.2f}\n",
           "| Metric | " + " | ".join(cols) + " |",
           "|---|" + "---|" * len(cols)]
    for label, key in rows:
        out.append(f"| {label} | " + " | ".join(str(data[c][key]) for c in cols) + " |")
    # вопросы с самым низким покрытием - полезно разобрать на защите
    worst = sorted(runs, key=lambda r: r["coverage"])[:5]
    out += ["", "## Lowest coverage", "", "| id | category | coverage | question |", "|---|---|---|---|"]
    out += [f"| {r['id']} | {r['category']} | {r['coverage'] * 100:.0f}% | {r['question']} |"
            for r in worst]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="run only the first N questions")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3, help="questions run in parallel")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN)
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT)
    args = ap.parse_args()

    config.setup_logging()
    if not (config.LLM_API_KEY and config.TAVILY_API_KEY):
        raise SystemExit(f"{config.LLM_KEY_NAME} and TAVILY_API_KEY must be set in .env")

    questions = load_questions(args.limit)
    out_dir = EVAL_DIR / "results" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    log.info("eval start: %d questions, max_steps=%d", len(questions), args.max_steps)

    # агентов запускаем параллельно, чтобы прогон занял минуты, а не полчаса
    done = 0

    def task(row):
        nonlocal done
        r = run_one(row, args.max_steps)
        done += 1
        print(f"[{done}/{len(questions)}] {row['id']} "
              f"{'OK' if r['report'] and not r['error'] else 'FAIL: ' + str(r['error'])}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(task, questions))

    if config.PROVIDER == "gemini":
        from openai import OpenAI
        client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)
    else:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    for r in runs:
        try:
            g = judge(client, r)
        except Exception as e:  # судья упал - считаем покрытие нулевым, но не теряем прогон
            log.exception("judge failed for %s", r["id"])
            g = {"covered": [], "judge_tokens": 0, "judge_error": str(e)}
        r.update(g)
        r["coverage"] = sum(g["covered"]) / len(g["covered"]) if g["covered"] else 0.0

    (out_dir / "runs.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "difficulty", "success", "coverage", "findings",
                    "latency_s", "tool_calls", "input_tokens", "output_tokens", "budget_hit",
                    "error", "question"])
        for r in runs:
            m = r["metrics"] or {}
            w.writerow([r["id"], r["category"], r["difficulty"], bool(r["report"] and not r["error"]),
                        round(r["coverage"], 2), len((r["report"] or {}).get("findings", [])),
                        m.get("latency_s"), m.get("steps"), m.get("input_tokens"),
                        m.get("output_tokens"), r["budget_hit"], r["error"] or "", r["question"]])
    summary = summarize(runs, args.price_in, args.price_out)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
