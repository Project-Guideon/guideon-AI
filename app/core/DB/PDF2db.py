import io
import re
from typing import List

import pdfplumber
from sentence_transformers import SentenceTransformer
from psycopg.types.json import Jsonb

from app.core.DB.connect_db import get_conn


MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # 768 dim (다국어, 한국어 최적화)
model = SentenceTransformer(MODEL_NAME)


# ── 공용 헬퍼 (PDF2db_test.py에서도 import해서 사용) ──────────────────────────

def clean_pdf_text(text: str) -> str:
    """PDF 추출 텍스트 정리 (줄바꿈/공백/빈줄)"""
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)   # 줄 끝 공백 제거
    text = re.sub(r"\n{3,}", "\n\n", text)   # 과도한 빈줄 축소

    parts = text.split("\n\n--- page ")
    if len(parts) > 1:
        head = parts[0]
        pages = []
        for p in parts[1:]:
            idx = p.find("---\n")
            if idx == -1:
                pages.append(p)
                continue
            page_header = p[:idx + 4]
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
    """문단 기반 chunking: 빈줄(\n\n) 기준으로 합치면서 chunk_size 맞추기"""
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


# ── API용 함수 (FastAPI BackgroundTasks에서 사용) ─────────────────────────────

def create_doc_record(
    original_name: str,
    file_hash: str,
    site_id: int,
    file_size: int,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """
    API가 즉시 doc_id를 반환하기 위해 먼저 PENDING 상태로 레코드 생성.
    실제 PDF 처리는 process_pdf_bytes()가 백그라운드에서 담당.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tb_document
                (site_id, original_name, storage_url, file_hash, file_size,
                 chunk_size, chunk_overlap, embedding_model, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'PENDING')
                RETURNING doc_id
                """,
                (site_id, original_name, "", file_hash, file_size,
                 chunk_size, chunk_overlap, MODEL_NAME),
            )
            # storage_url은 백엔드에서 받아야 함 (S3 URL) => 우선 ""로 넣어둠
            doc_id = cur.fetchone()[0]
        conn.commit()
    return doc_id


def process_pdf_bytes(
    doc_id: int,
    pdf_bytes: bytes,
    site_id: int,
    chunk_size: int,
    chunk_overlap: int,
    purge_old_chunks: bool = True,
) -> None:
    """
    FastAPI BackgroundTasks에서 호출.
    PENDING -> PROCESSING -> COMPLETED / FAILED 순으로 상태 변경.
    """
    try:
        # 1) bytes -> 텍스트 추출 (io.BytesIO로 파일처럼 다룸)
        texts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                texts.append(f"\n\n--- page {i+1} ---\n{t}")
        raw = "".join(texts).strip()

        # 2) 클리닝 + 유효성 검사
        raw = clean_pdf_text(raw)
        if not raw or len(raw.strip()) < 30:
            raise RuntimeError("PDF 텍스트 추출 결과가 비어있거나 너무 짧음")

        # 3) 청킹
        chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"[PDF] doc_id={doc_id} | chunks={len(chunks)}")

        # 4) DB 저장 (PENDING -> PROCESSING -> COMPLETED)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tb_document SET status = 'PROCESSING' WHERE doc_id = %s",
                    (doc_id,),
                )
                # 재처리 시 기존 청크 삭제 (purge_old_chunks=True)
                if purge_old_chunks:
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
                    """UPDATE tb_document
                       SET status = 'COMPLETED', processed_at = NOW()
                       WHERE doc_id = %s""",
                    (doc_id,),
                )
            conn.commit()
        print(f"[PDF] doc_id={doc_id} 처리 완료")

    except Exception as e:
        print(f"[PDF] doc_id={doc_id} 처리 실패: {e}")
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE tb_document
                           SET status = 'FAILED', failed_reason = %s
                           WHERE doc_id = %s""",
                        (str(e), doc_id),
                    )
                conn.commit()
        except Exception:
            pass
