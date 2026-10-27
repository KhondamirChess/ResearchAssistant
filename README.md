# LLM-Based Agentic Research Assistant (prototype v0.2-0.3)

A FastAPI backend with an agent loop (Plan → Act → Observe → Reflect → Write) and a
single-page web UI (English, Korean, Chinese, Uzbek). Matches the design slides: two tools,
findings-list report, step limit, logging of time / tokens.

## Run on Ubuntu

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
cd research-assistant
chmod +x run.sh && ./run.sh          # creates .venv, installs packages, starts the server
```

Open http://localhost:8000 in your browser. **Demo mode works with no keys** and uses the mock data in `backend/mock_data.json`.

Manual version of the same thing:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # then edit .env
uvicorn backend.app:app --reload --port 8000
```

## Turn on Live mode

1. Create an API key in the Anthropic Console, and a free key at https://app.tavily.com
2. Put them in `.env`:
   ```
   ANTHROPIC_API_KEY=...
   TAVILY_API_KEY=...
   ```
3. Restart the server. The badge at the top changes to "live ready" and the **Live** button unlocks.

Live mode spends API credits. Start with Max steps = 3-4.
Never commit `.env` (it is already in `.gitignore`).

## Tests

```bash
source .venv/bin/activate
pytest -q
```

The tests need no keys or network: tool safety checks, mock data and matching, and the
agent loop driven by a scripted fake Claude client.

## What is where

| Path | What it does |
|---|---|
| `backend/app.py` | FastAPI endpoints: `/api/research` (streams steps), `/api/health`, `/api/history` |
| `backend/agent.py` | Agent loop, demo mode (matches your question to the mock data), `normalize_report` |
| `backend/mock_data.json` | Mock questions and findings in en / ko / zh / uz. Add your own entries here |
| `backend/tools.py` | `web_search` (Tavily) and `read_document` (PDF + web pages), blocks private URLs |
| `frontend/index.html` | The whole UI: language switcher, live trace, findings list, history |
| `data/history.json` | Saved runs (created automatically) |
| `tests/` | pytest tests |

## Language switcher

The buttons at the top right switch the interface between English, Korean, Chinese and
Uzbek. The choice is remembered in the browser. The selected language is also sent to the
backend, so the agent trace and the findings come back in that language (Demo reads the
matching text from `mock_data.json`; Live asks Claude to write in that language).

## Mock questions (Demo mode)

Type one of these (in any of the four languages; small typos are fine). Anything else
shows a short "no mock entry" result.

1. What are the main limitations of retrieval-augmented generation?
2. How do LLM agents decide which tool to call?
3. What methods exist to reduce hallucinated citations in LLMs?
4. Compare ReAct and plan-and-execute agent designs
5. How can research agents defend against prompt injection?
6. How should LLM agents be evaluated?

To add more, copy an item in `backend/mock_data.json` and fill the `questions` and
`findings` for `en`, `ko`, `zh`, `uz`. Restart the server.

## Guardrails built in

- Only the two listed tools can be called; any other tool name is blocked.
- Step limit (Max steps); when it is reached the agent must write the report anyway.
- The prompt tells the model that page text is untrusted data, not instructions.
- `read_document` refuses localhost / private network addresses (including after redirects).

## Not done yet (next phases)

Session state / follow-up questions (v0.4), retries and result caching (v0.5),
evaluation runner over `questions.csv` (v1.0).
