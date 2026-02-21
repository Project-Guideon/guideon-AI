"""
로컬 테스트용 - 단일 PDF 파일을 직접 DB에 삽입
실제 API는 PDF2db.py의 create_doc_record + process_pdf_bytes 사용
"""
import hashlib
from pathlib import Path

import pdfplumber
from psycopg.types.json import Jsonb

from app.core.DB.connect_db import get_conn
from app.core.DB.PDF2db import clean_pdf_text, chunk_text, MODEL_NAME, model


BASE_DIR = Path(__file__).resolve().parents[1]  # app/core
DATA_DIR = BASE_DIR / "data"


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


def ingest_pdf(
    pdf_filename: str,
    site_id: int = 1,  # 테스트용 기본값 (실제로는 백엔드에서 받아야 함)
    chunk_size: int = 600,
    chunk_overlap: int = 100,
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
                    pdf_path.name,
                    "",  # storage_url (S3 URL은 백엔드에서 받아야 함)
                    file_hash,
                    chunk_size,
                    chunk_overlap,
                    MODEL_NAME,
                    "PROCESSING",
                ),
            )
            doc_id = cur.fetchone()[0]
            print(f"Document created: doc_id={doc_id}")

            if reset:
                cur.execute(
                    "DELETE FROM tb_doc_chunk WHERE doc_id = %s",
                    (doc_id,),
                )

            for idx, chunk in enumerate(chunks):
                emb = model.encode(chunk, normalize_embeddings=True).tolist()
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
                    (site_id, doc_id, idx, chunk, Jsonb(meta), emb),
                )

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
