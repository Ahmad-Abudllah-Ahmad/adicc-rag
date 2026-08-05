# ADICC Volume 4 RAG API

Serves Drawings Q&A from the local SQLite knowledge base (`adicc.db`).

```bash
cd backend
export ADICC_DB="../adicc.db"
# optional — PDF corpus for citation open / page preview
export ADICC_CORPUS="$HOME/Downloads/VOLUME 4 - DRAWINGS"
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Health: http://127.0.0.1:8001/health  
Vite proxies `/rag` → this service (see `opentakeoff/web/vite.config.js`).
