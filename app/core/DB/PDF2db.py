import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from sentence_transformers import SentenceTransformer
from psycopg.types.json import Jsonb

from app.core.DB.connect_db import get_conn


BASE_DIR = Path(__file__).resolve().parents[1]  # app/core
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dim
model = SentenceTransformer(MODEL_NAME)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_pdf_text(pdf_path: Path, x_tol: float = 2, y_tol: float = 2) -> str:
    """pdfplumber로 텍스트 추출"""
    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text(x_tolerance=x_tol, y_tolerance=y_tol) or ""
            texts.append(f"\n\n--- page {i+1} ---\n{t}")
    return "".join(texts).strip()


def clean_pdf_text(text: str) -> str:
    """PDF 추출 텍스트 정리 (줄바꿈/공백/빈줄)"""
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)   # 줄 끝 공백 제거
    text = re.sub(r"\n{3,}", "\n\n", text)   # 과도한 빈줄 축소

    # 페이지 헤더는 유지하면서 본문 단일 줄바꿈은 공백으로 (문장 끊김 완화)
    parts = text.split("\n\n--- page ")
    if len(parts) > 1:
        head = parts[0]
        pages = []
        for p in parts[1:]:
            idx = p.find("---\n")
            if idx == -1:
                pages.append(p)
                continue
            page_header = p[:idx + 4]      # "N ---"
            page_body = p[idx + 4:]

            page_body = re.sub(r"(?<!\n)\n(?!\n)", " ", page_body)  # 단일 줄바꿈 -> 공백
            page_body = re.sub(r"[ \t]{2,}", " ", page_body)        # 다중 공백 축소

            pages.append(page_header + page_body)

        text = head + "\n\n--- page " + "\n\n--- page ".join(pages)
    else:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """
    문단 기반 chunking: 빈줄(\n\n) 기준으로 합치면서 chunk_size 맞추기
    """
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""

    for p in paras:
        if not cur:
            cur = p
            continue

        if len(cur) + 2 + len(p) <= chunk_size:
            cur += "\n\n" + p
        else:
            chunks.append(cur)

            if chunk_overlap > 0 and len(cur) > chunk_overlap:
                tail = cur[-chunk_overlap:]
                cur = tail + "\n\n" + p
            else:
                cur = p

    if cur:
        chunks.append(cur)

    return chunks


def ingest_pdf(
    pdf_filename: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    reset: bool = False,
):
    pdf_path = DATA_DIR / pdf_filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    file_hash = sha256_file(pdf_path)

    # 1) 텍스트 추출
    raw = read_pdf_text(pdf_path)

    # 2) 클리닝
    raw = clean_pdf_text(raw)

    if not raw or len(raw.strip()) < 30:
        raise RuntimeError("PDF 텍스트 추출 결과가 비어있거나 너무 짧음 (OCR 제거 상태)")

    # 3) 청킹
    chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"PDF loaded: {pdf_path.name} | chunks={len(chunks)} | hash={file_hash}")

    # 4) DB insert
    with get_conn() as conn:
        with conn.cursor() as cur:
            if reset:
                cur.execute(
                    "DELETE FROM documents WHERE metadata->>'file_hash' = %s",
                    (file_hash,),
                )

            for idx, chunk in enumerate(chunks):
                emb = model.encode(chunk).tolist()
                meta: Dict[str, Any] = {
                    "source": "pdf",
                    "storage_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "file_hash": file_hash,
                    "chunk_index": idx,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embed_model": MODEL_NAME,
                    "extractor": "pdfplumber",
                }
                cur.execute(
                    "INSERT INTO documents (content, metadata, embedding) VALUES (%s, %s, %s)",
                    (chunk, Jsonb(meta), emb),
                )
        conn.commit()

    print("✅ ingest complete (DB inserted)")


if __name__ == "__main__":
    ingest_pdf("경복궁가이드pdf.pdf", reset=True)
