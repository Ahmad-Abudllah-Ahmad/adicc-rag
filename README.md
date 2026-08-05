# ADICC Volume 4 RAG API

FastAPI retrieval + OpenAI chat over the SQLite knowledge base (`data/adicc.db`).

## Live (Render)

- Service: `https://adicc-rag.onrender.com`
- Health: `https://adicc-rag.onrender.com/health`
- Dashboard: `https://dashboard.render.com/web/srv-d9peb56417fc73dnt6e0`
- Repo: `https://github.com/Ahmad-Abudllah-Ahmad/adicc-rag`

Frontend (`https://adicc.onrender.com`) uses build env `VITE_RAG_URL=https://adicc-rag.onrender.com`.

## Local

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY
# ADICC_DB should point at ./data/adicc.db
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

OpenTakeoff Vite proxies `/rag` → this service when `VITE_RAG_URL` is unset.

## Knowledge base

| Environment | Path |
|-------------|------|
| Local | `backend/data/adicc.db` (hard-linked to workspace `adicc.db`) |
| Render / Docker | `/app/data/adicc.db` via `ADICC_DB` |
