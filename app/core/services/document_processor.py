"""
Core 연동 전용 문서 처리 서비스.

흐름:
  Core → POST /v1/documents/process
    → storage_url에서 PDF 다운로드
    → 청킹 / 임베딩 → tb_doc_chunk INSERT
    → Core에 PATCH 콜백 (COMPLETED or FAILED)

기존 PDF2db.py의 process_pdf_bytes()는 건드리지 않음.
clean_pdf_text, chunk_text 헬퍼만 재사용.
"""
from __future__ import annotations

import asyncio
import io
import os

import httpx
import pdfplumber
from psycopg.types.json import Jsonb

from app.core.DB.PDF2db import clean_pdf_text, chunk_text
from app.core.DB.connect_db import get_conn

MODEL_NAME = "text-embedding-3-small"
CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8080")


def _extract_text_from_pdf(pdf_bytes: bytes) -> list[str]:
    """pdfplumber는 동기 라이브러리 → to_thread로 호출"""
    texts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            texts.append(f"\n\n--- page {i+1} ---\n{t}")
    return texts


def _process_chunks_sync(
    doc_id: int,
    site_id: int,
    chunks: list[str],
    chunk_size: int,
    chunk_overlap: int,
    openai_client,
    embedding_model: str,
) -> None:
    """DB 작업 + 임베딩 API 호출은 동기 블로킹 → to_thread로 호출"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tb_doc_chunk WHERE doc_id = %s", (doc_id,))
            for idx, chunk in enumerate(chunks):
                emb = openai_client.embeddings.create(
                    model=embedding_model,
                    input=chunk,
                ).data[0].embedding
                meta = {
                    "chunk_index": idx,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "embed_model": embedding_model,
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
        conn.commit()


async def process_pdf_from_url(
    doc_id: int,
    site_id: int,
    storage_url: str,
    chunk_size: int,
    chunk_overlap: int,
    openai_client,
    embedding_model: str = MODEL_NAME,
) -> None:
    """
    Core에서 요청받은 문서를 처리하고 결과를 Core에 콜백.

    - tb_document 상태 업데이트: Core 콜백으로 처리 (FastAPI가 직접 쓰지 않음)
    - tb_doc_chunk: FastAPI가 직접 INSERT (벡터 데이터는 AI 서버 소유)
    """
    try:
        # 1. storage_url에서 PDF 다운로드
        async with httpx.AsyncClient() as http:
            resp = await http.get(storage_url)
            resp.raise_for_status()
            pdf_bytes = resp.content

        # 2. 텍스트 추출 (블로킹 작업을 스레드 풀에서 실행)
        texts = await asyncio.to_thread(_extract_text_from_pdf, pdf_bytes)

        raw = clean_pdf_text("".join(texts).strip())
        if not raw or len(raw.strip()) < 30:
            raise RuntimeError("PDF 텍스트 추출 결과가 비어있거나 너무 짧음")

        # 3. 청킹
        chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"[processor] doc_id={doc_id} | chunks={len(chunks)}")

        # 4. 임베딩 + tb_doc_chunk INSERT (블로킹 작업을 스레드 풀에서 실행)
        await asyncio.to_thread(
            _process_chunks_sync,
            doc_id, site_id, chunks, chunk_size, chunk_overlap, openai_client, embedding_model,
        )

        # 5. Core에 COMPLETED 콜백
        await _callback_core(site_id, doc_id, "COMPLETED", None)

    except Exception as e:
        print(f"[processor] doc_id={doc_id} 처리 실패: {e}")
        await _callback_core(site_id, doc_id, "FAILED", str(e))


async def _callback_core(
    site_id: int, doc_id: int, status: str, failed_reason: str | None
) -> None:
    payload: dict = {"status": status}
    if failed_reason:
        payload["failed_reason"] = failed_reason

    try:
        async with httpx.AsyncClient() as http:
            await http.patch(
                f"{CORE_BASE_URL}/internal/v1/sites/{site_id}/documents/{doc_id}/status",
                json=payload,
                timeout=10.0,
            )
    except Exception as e:
        print(f"[processor] Core 콜백 실패 doc_id={doc_id} status={status}: {e}")
