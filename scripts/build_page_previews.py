#!/usr/bin/env python3
"""Build compact JPEG page previews for citation images (deploy without full PDF corpus)."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("ADICC_DB", ROOT / "data" / "adicc.db"))
CORPUS = Path(
    os.environ.get(
        "ADICC_CORPUS",
        str(Path.home() / "Downloads" / "VOLUME 4 - DRAWINGS"),
    )
)
OUT = Path(os.environ.get("ADICC_PREVIEWS", ROOT / "data" / "previews"))
SCALE = float(os.environ.get("ADICC_PREVIEW_SCALE", "0.55"))
QUALITY = int(os.environ.get("ADICC_PREVIEW_QUALITY", "45"))


def resolve(path: str | None, rel: str | None) -> Path | None:
    candidates: list[Path] = []
    for p in (path, rel):
        if not p:
            continue
        cleaned = str(p).replace("\\", "/").strip()
        candidates.append(Path(cleaned))
        if not Path(cleaned).is_absolute():
            candidates.append(CORPUS / cleaned)
        # strip volume prefix if present
        marker = "VOLUME 4 - DRAWINGS/"
        if marker.lower() in cleaned.lower():
            idx = cleaned.lower().index(marker.lower()) + len(marker)
            candidates.append(CORPUS / cleaned[idx:].lstrip("/"))
        base = Path(cleaned).name
        if base:
            candidates.append(CORPUS / base)
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
    if rel or path:
        base = Path(str(rel or path).replace("\\", "/")).name
        if base and CORPUS.is_dir():
            for hit in CORPUS.rglob(base):
                if hit.is_file():
                    return hit
    return None


def main() -> int:
    if not DB.is_file():
        print(f"DB missing: {DB}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = list(
        con.execute(
            """
            SELECT DISTINCT p.id AS page_id, p.page_no, d.path, d.rel_path
            FROM pages p
            JOIN documents d ON d.id = p.doc_id
            JOIN chunks c ON c.page_id = p.id
            WHERE lower(d.rel_path) LIKE '%.pdf'
            ORDER BY p.id
            """
        )
    )
    con.close()
    ok = skip = miss = 0
    for i, (page_id, page_no, path, rel) in enumerate(rows, 1):
        dest = OUT / f"{page_id}.jpg"
        if dest.is_file() and dest.stat().st_size > 500:
            skip += 1
            continue
        src = resolve(path, rel)
        if not src:
            miss += 1
            continue
        try:
            doc = fitz.open(src)
            pn = int(page_no or 0)
            if pn < 0 or pn >= doc.page_count:
                pn = 0
            page = doc.load_page(pn)
            pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
            dest.write_bytes(pix.tobytes("jpeg", jpg_quality=QUALITY))
            doc.close()
            ok += 1
        except Exception as e:
            miss += 1
            print(f"fail page_id={page_id}: {e}", file=sys.stderr)
        if i % 100 == 0:
            print(f"... {i}/{len(rows)} ok={ok} skip={skip} miss={miss}")
    print(f"done total={len(rows)} wrote={ok} skipped={skip} missing={miss} out={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
