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

MODEL_NAME = "all-mpnet-base-v2"  # 768 dim (중간 성능, 무료)
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
    site_id: int = 1,  # 테스트용 기본값 (실제로는 백엔드에서 받아야 함)
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
            # tb_document에 먼저 INSERT (문서 메타데이터)
            cur.execute(
                """
                INSERT INTO tb_document 
                (site_id, original_name, storage_url, file_hash, 
                 chunk_size, chunk_overlap, embedding_model, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING doc_id
                """,
                (
                    site_id,
                    pdf_path.name,  # original_name
                    "",  # storage_url (S3 URL은 백엔드에서 받아야 함)
                    file_hash,
                    chunk_size,
                    chunk_overlap,
                    MODEL_NAME,
                    "PROCESSING",  # 시작: PROCESSING
                ),
            )
            doc_id = cur.fetchone()[0]
            print(f"Document created: doc_id={doc_id}")

            # tb_doc_chunk에 청크들 INSERT
            if reset:
                cur.execute(
                    "DELETE FROM tb_doc_chunk WHERE doc_id = %s",
                    (doc_id,),
                )

            for idx, chunk in enumerate(chunks):
                emb = model.encode(chunk).tolist()
                meta = {
                    "chunk_index": idx,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embed_model": MODEL_NAME,
                    "extractor": "pdfplumber",
                }
                cur.execute(
                    """
                    INSERT INTO tb_doc_chunk 
                    (site_id, doc_id, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        site_id,
                        doc_id,
                        idx,
                        chunk,
                        Jsonb(meta),
                        emb,
                    ),
                )

            # 문서 처리 완료: status를 COMPLETED로 변경
            cur.execute(
                """
                UPDATE tb_document 
                SET status = %s, processed_at = NOW()
                WHERE doc_id = %s
                """,
                ("COMPLETED", doc_id),
            )

        conn.commit()

    print("✅ ingest complete (DB inserted)")


if __name__ == "__main__":
    # site_id=1로 테스트 (실제로는 백엔드에서 받아야 함)
    ingest_pdf("경복궁가이드pdf.pdf", site_id=1, reset=True)
