"""ADICC Volume 4 RAG API — serves Drawings Q&A from local adicc.db.

Uses OpenAI chat for structured answers. Point ADICC_DB at the SQLite KB.
Optional ADICC_CORPUS points at the Volume 4 drawings folder for citation files.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT = BACKEND_DIR.parent  # .../cursor to detect
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_DB = ROOT / "adicc.db"
DEFAULT_CORPUS = Path(
    os.environ.get(
        "ADICC_CORPUS",
        str(Path.home() / "Downloads" / "VOLUME 4 - DRAWINGS"),
    )
)

DB_PATH = Path(os.environ.get("ADICC_DB", str(DEFAULT_DB))).expanduser().resolve()
CORPUS_ROOT = Path(os.environ.get("ADICC_CORPUS", str(DEFAULT_CORPUS))).expanduser().resolve()

OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY") or "").strip()
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
MAX_CITATIONS = 2

SYSTEM_PROMPT = """You are the ADICC Volume 4 drawings assistant.
Answer ONLY from the provided source excerpts.
Write a clear, professional response for a construction takeoff chat.

Formatting rules (strict):
- Start with a short bold title line using this exact pattern: TITLE: Your Title Here
- Then use section headings with this exact pattern: SECTION: Heading
- Under each section write plain sentences only.
- Do not use markdown symbols, bullets, dashes as bullets, asterisks, hashes, backticks, emojis, or special characters for decoration.
- Do not invent sheet numbers or finishes that are not in the sources.
- If sources are insufficient, say so plainly and set confidence low.
- Keep the answer concise and well structured.
"""

app = FastAPI(title="ADICC Volume 4 RAG", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class FinishForRoomRequest(BaseModel):
    room: str = Field(..., min_length=1)


def connect() -> sqlite3.Connection:
    if not DB_PATH.is_file():
        raise HTTPException(503, f"Knowledge base not found: {DB_PATH}")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def normalize_room(s: str) -> str:
    s = (s or "").upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fts_query(raw: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+\-]{1,}", raw or "")
    tokens = [t for t in tokens if len(t) > 1][:12]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def parse_bbox(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
            return parse_bbox(data)
        except json.JSONDecodeError:
            return None
    return None


def citation_from_row(row: sqlite3.Row, quote: str | None = None) -> dict[str, Any]:
    text = quote if quote is not None else (row["text"] or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 480:
        text = text[:477] + "..."
    bbox = None
    keys = row.keys()
    if "bbox" in keys:
        bbox = parse_bbox(row["bbox"])
    return {
        "id": f"c{row['id']}",
        "chunk_id": int(row["id"]),
        "doc_path": row["rel_path"] or row["path"] or "",
        "page_no": int(row["page_no"] or 0),
        "sheet_id": row["sheet_id"],
        "sheet_title": row["sheet_title"],
        "discipline": row["discipline"],
        "bbox": bbox,
        "quote": text,
        "source": row["source"] or "text",
        "verified": True,
    }


def resolve_source_path(doc_path: str | None, rel_path: str | None = None) -> Path | None:
    candidates: list[Path] = []
    for p in (doc_path, rel_path):
        if not p:
            continue
        cleaned = str(p).replace("\\", "/").strip()
        candidates.append(Path(cleaned))
        m = re.search(r"(?:VOLUME\s*4\s*-\s*DRAWINGS[/\\])?(.+)$", cleaned, re.I)
        if m:
            candidates.append(CORPUS_ROOT / m.group(1).lstrip("/"))
        if not Path(cleaned).is_absolute():
            candidates.append(CORPUS_ROOT / cleaned)
        base = Path(cleaned).name
        if base:
            candidates.append(CORPUS_ROOT / base)

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.is_file():
                return c
        except OSError:
            continue

    if rel_path or doc_path:
        base = Path(str(rel_path or doc_path).replace("\\", "/")).name
        if base and CORPUS_ROOT.is_dir():
            for hit in CORPUS_ROOT.rglob(base):
                if hit.is_file():
                    return hit
    return None


def search_chunks(con: sqlite3.Connection, question: str, limit: int = 8) -> list[sqlite3.Row]:
    q = fts_query(question)
    sql = """
    SELECT c.id, c.text, c.sheet_id, c.sheet_title, c.discipline, c.source,
           d.path, d.rel_path, p.page_no, r.bbox AS bbox
    FROM chunks_fts
    JOIN chunks c ON c.id = chunks_fts.rowid
    JOIN documents d ON d.id = c.doc_id
    JOIN pages p ON p.id = c.page_id
    LEFT JOIN regions r ON r.id = c.region_id
    WHERE chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts)
    LIMIT ?
    """
    try:
        rows = list(con.execute(sql, (q, limit)))
    except sqlite3.OperationalError:
        rows = []
    if rows:
        return rows
    like = f"%{(question or '').strip()[:80]}%"
    return list(
        con.execute(
            """
            SELECT c.id, c.text, c.sheet_id, c.sheet_title, c.discipline, c.source,
                   d.path, d.rel_path, p.page_no, r.bbox AS bbox
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            JOIN pages p ON p.id = c.page_id
            LEFT JOIN regions r ON r.id = c.region_id
            WHERE c.text LIKE ?
            LIMIT ?
            """,
            (like, limit),
        )
    )


def strip_decorative_symbols(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[#*`>•●○◆▪︎■□★☆✓✔✕✖⚠️🔧📌]+", "", text)
    text = re.sub(r"^\s*[-–—]\s+", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_fallback_answer(question: str, rows: list[sqlite3.Row]) -> tuple[str, bool]:
    if not rows:
        return (
            "TITLE: No matching source found\n\n"
            "SECTION: Result\n"
            "I could not find matching text in the Volume 4 knowledge base for that question.",
            True,
        )
    lines = [
        "TITLE: Drawings answer",
        "",
        "SECTION: Findings",
    ]
    for i, row in enumerate(rows[:MAX_CITATIONS], 1):
        label = row["sheet_id"] or row["sheet_title"] or Path(row["rel_path"] or "").name or f"Source {i}"
        snippet = re.sub(r"\s+", " ", (row["text"] or "")).strip()
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."
        lines.append(f"{i}) {label}. {snippet}")
    lines.extend(["", "SECTION: Question", question.strip()])
    return "\n".join(lines), False


def openai_answer(question: str, rows: list[sqlite3.Row]) -> tuple[str, bool]:
    if not rows:
        return build_fallback_answer(question, rows)
    if not OPENAI_API_KEY:
        return build_fallback_answer(question, rows)

    sources = []
    for i, row in enumerate(rows[:6], 1):
        sources.append(
            {
                "n": i,
                "sheet_id": row["sheet_id"],
                "sheet_title": row["sheet_title"],
                "doc_path": row["rel_path"] or row["path"],
                "page": int(row["page_no"] or 0) + 1,
                "text": (row["text"] or "")[:1200],
            }
        )
    user_prompt = (
        f"Question: {question.strip()}\n\n"
        f"Source excerpts (JSON):\n{json.dumps(sources, ensure_ascii=False)}\n\n"
        "Produce the structured answer now."
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        answer = strip_decorative_symbols(answer)
        if not answer:
            return build_fallback_answer(question, rows)
        return answer, False
    except Exception as e:
        fallback, abstained = build_fallback_answer(question, rows)
        return (
            f"{fallback}\n\nSECTION: Note\nLive model answer unavailable ({type(e).__name__}). Showing source excerpts instead.",
            abstained,
        )


@app.get("/health")
def health() -> dict[str, Any]:
    ok = DB_PATH.is_file()
    chunks = docs = None
    if ok:
        with connect() as con:
            chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    return {
        "status": "ok" if ok else "missing_db",
        "db": str(DB_PATH),
        "corpus": str(CORPUS_ROOT),
        "chunks": chunks,
        "documents": docs,
        "openai": bool(OPENAI_API_KEY),
        "chat_model": OPENAI_CHAT_MODEL,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "max_citations": MAX_CITATIONS,
    }


@app.post("/query")
def query(body: QueryRequest) -> dict[str, Any]:
    with connect() as con:
        rows = search_chunks(con, body.query, limit=8)
    cite_rows = rows[:MAX_CITATIONS]
    answer, abstained = openai_answer(body.query, rows)
    citations = [citation_from_row(r) for r in cite_rows]
    return {
        "answer": answer,
        "citations": citations,
        "abstained": abstained,
        "candidates": None,
    }


@app.post("/query/stream")
def query_stream(body: QueryRequest) -> StreamingResponse:
    result = query(body)

    def gen():
        yield f"data: {json.dumps({'type': 'answer', 'answer': result['answer']})}\n\n"
        yield f"data: {json.dumps({'type': 'citations', 'citations': result['citations']})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/finish-for-room")
def finish_for_room(body: FinishForRoomRequest) -> dict[str, Any]:
    room_raw = body.room.strip()
    room_n = normalize_room(room_raw)
    if not room_n:
        return {
            "room": room_raw,
            "matched_room": None,
            "finish_codes": [],
            "citations": [],
            "abstained": True,
        }

    with connect() as con:
        candidates = list(
            con.execute(
                """
                SELECT DISTINCT room_or_key
                FROM schedule_rows
                WHERE schedule_type = 'finish_schedule'
                  AND room_or_key IS NOT NULL
                  AND trim(room_or_key) != ''
                """
            )
        )
        matched = None
        best_score = -1
        for (name,) in candidates:
            nn = normalize_room(name)
            if not nn:
                continue
            score = 0
            if nn == room_n:
                score = 100
            elif room_n in nn or nn in room_n:
                score = 80
            else:
                a, b = set(room_n.split()), set(nn.split())
                if a and b and (a <= b or b <= a):
                    score = 60
                elif a & b:
                    score = 40 + len(a & b)
            if score > best_score:
                best_score = score
                matched = name

        if best_score < 40 or not matched:
            tag = con.execute(
                """
                SELECT room_name, normalized FROM room_tags
                WHERE normalized = ? OR normalized LIKE ? OR room_name LIKE ?
                LIMIT 1
                """,
                (room_n, f"%{room_n}%", f"%{room_raw}%"),
            ).fetchone()
            if tag:
                matched = tag["room_name"]
                best_score = 50
            else:
                return {
                    "room": room_raw,
                    "matched_room": None,
                    "finish_codes": [],
                    "citations": [],
                    "abstained": True,
                }

        rows = list(
            con.execute(
                """
                SELECT category, code, description, row_json, doc_id, page_id
                FROM schedule_rows
                WHERE schedule_type = 'finish_schedule'
                  AND room_or_key = ?
                ORDER BY id
                """,
                (matched,),
            )
        )

        finish_codes: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for r in rows:
            cat = (r["category"] or "").strip() or "FINISH"
            code = (r["code"] or "").strip()
            desc = (r["description"] or "").strip()
            key = (cat, code or desc)
            if key in seen:
                continue
            seen.add(key)
            finish_codes.append(
                {
                    "category": cat,
                    "code": code or "-",
                    "description": desc,
                    "material": desc or None,
                }
            )

        if not finish_codes and rows:
            try:
                payload = json.loads(rows[0]["row_json"] or "{}")
                for cat in payload.get("categories") or []:
                    finish_codes.append(
                        {
                            "category": cat.get("category") or "FINISH",
                            "code": cat.get("code") or "-",
                            "description": cat.get("description") or "",
                            "material": cat.get("description"),
                        }
                    )
            except json.JSONDecodeError:
                pass

        citations: list[dict[str, Any]] = []
        if rows:
            doc_id = rows[0]["doc_id"]
            page_id = rows[0]["page_id"]
            hit = con.execute(
                """
                SELECT c.id, c.text, c.sheet_id, c.sheet_title, c.discipline, c.source,
                       d.path, d.rel_path, p.page_no, r.bbox AS bbox
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                JOIN pages p ON p.id = c.page_id
                LEFT JOIN regions r ON r.id = c.region_id
                WHERE c.doc_id = ? AND c.page_id = ?
                LIMIT ?
                """,
                (doc_id, page_id, MAX_CITATIONS),
            ).fetchall()
            if not hit:
                hit = search_chunks(con, f"finish schedule {matched}", limit=MAX_CITATIONS)
            citations = [citation_from_row(h) for h in hit[:MAX_CITATIONS]]

    return {
        "room": room_raw,
        "matched_room": matched,
        "finish_codes": finish_codes,
        "citations": citations,
        "abstained": not bool(finish_codes),
    }


def _lookup_doc(con: sqlite3.Connection, chunk_id: int) -> sqlite3.Row:
    row = con.execute(
        """
        SELECT c.id, c.text, d.path, d.rel_path, p.page_no, r.bbox AS bbox
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        JOIN pages p ON p.id = c.page_id
        LEFT JOIN regions r ON r.id = c.region_id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if not row:
        raise HTTPException(404, f"chunk {chunk_id} not found")
    return row


def _highlight_needles(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    needles = [cleaned[:90]]
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-_/]{2,}", cleaned)
    if len(words) >= 4:
        needles.append(" ".join(words[:6]))
    if len(words) >= 8:
        needles.append(" ".join(words[3:9]))
    # unique preserve order
    out: list[str] = []
    seen: set[str] = set()
    for n in needles:
        key = n.lower()
        if key in seen or len(n) < 8:
            continue
        seen.add(key)
        out.append(n)
    return out[:4]


@app.get("/citation/{chunk_id}/file")
def citation_file(chunk_id: int, download: bool = False):
    with connect() as con:
        row = _lookup_doc(con, chunk_id)
    path = resolve_source_path(row["path"], row["rel_path"])
    if not path:
        raise HTTPException(
            404,
            f"Source file not on disk for chunk {chunk_id} "
            f"(rel_path={row['rel_path']}). Set ADICC_CORPUS to Volume 4 drawings root.",
        )
    media = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"
    return FileResponse(
        path,
        media_type=media,
        filename=path.name if download else None,
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/citation/{chunk_id}/image")
def citation_image(chunk_id: int):
    with connect() as con:
        row = _lookup_doc(con, chunk_id)
    path = resolve_source_path(row["path"], row["rel_path"])
    if not path or path.suffix.lower() != ".pdf":
        raise HTTPException(404, "No PDF page preview for this citation")
    try:
        import fitz
    except ImportError as e:
        raise HTTPException(503, "pymupdf not installed") from e
    try:
        doc = fitz.open(path)
        page_no = int(row["page_no"] or 0)
        if page_no < 0 or page_no >= doc.page_count:
            page_no = 0
        page = doc.load_page(page_no)

        # Highlight matching quote spans on the page
        for needle in _highlight_needles(row["text"] or ""):
            try:
                for rect in page.search_for(needle, quads=False)[:6]:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.92, 0.2))
                    annot.update()
            except Exception:
                continue

        # Fallback: draw region bbox if text search missed
        bbox = parse_bbox(row["bbox"] if "bbox" in row.keys() else None)
        existing = list(page.annots() or [])
        if bbox and not existing:
            try:
                x0, y0, x1, y1 = bbox
                pw, ph = page.rect.width, page.rect.height
                if (x1 - x0) * (y1 - y0) < 0.85 * pw * ph:
                    rect = fitz.Rect(x0, y0, x1, y1)
                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=(1, 0.75, 0.1))
                    annot.set_border(width=2)
                    annot.set_opacity(0.35)
                    annot.update()
            except Exception:
                pass

        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        png = pix.tobytes("png")
        doc.close()
    except Exception as e:
        raise HTTPException(500, f"Could not render page: {e}") from e
    return Response(png, media_type="image/png")


@app.get("/file")
def serve_file(path: str = Query(...), download: bool = False):
    resolved = resolve_source_path(path, path)
    if not resolved:
        raise HTTPException(404, f"File not found: {path}")
    media = "application/pdf" if resolved.suffix.lower() == ".pdf" else "application/octet-stream"
    return FileResponse(
        resolved,
        media_type=media,
        filename=resolved.name if download else None,
        content_disposition_type="attachment" if download else "inline",
    )
