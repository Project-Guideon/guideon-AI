"""
Core 연동 전용 문서 처리 서비스 (v2 파이프라인).

[전체 흐름]
  1. Spring Core → POST /v1/documents/process 요청
  2. FastAPI가 백그라운드 태스크로 이 모듈 실행
  3. storage_url에서 PDF 바이트 다운로드
  4. Core에 PROCESSING 상태 콜백
  5. v2 파이프라인 실행:
       PDF → pymupdf4llm 마크다운 변환
           → 헤더 기반 섹션 파싱 (청킹)
           → 각 섹션 GPT 요약 + 키워드 추출
           → 요약+키워드 텍스트로 임베딩 생성
           → tb_doc_chunk_v2에 DELETE → INSERT
  6. Core에 COMPLETED / FAILED 상태 콜백

[역할 분리]
  - tb_document 상태 관리: Spring Core 담당 (FastAPI는 콜백만 보냄)
  - tb_doc_chunk_v2 벡터 데이터: FastAPI 담당 (AI 서버가 직접 INSERT)

[청킹 파라미터] (FastAPI 내부 고정값, Spring에서 전달받지 않음)
  - MAX_CHUNK_SIZE  : 섹션 최대 글자 수
  - MIN_CHUNK_SIZE  : 너무 짧은 섹션 병합 기준
  - CHUNK_OVERLAP   : 섹션 간 겹치는 글자 수 (문맥 유지용)
"""
from __future__ import annotations

import asyncio
import os

import httpx
from psycopg.types.json import Jsonb

from app.core.DB.connect_db import get_conn
from app.core.DB.PDF2db_v2 import (
    pdf_to_markdown,
    parse_markdown_sections,
    generate_search_summary,
    MODEL_NAME,
    client as openai_client_v2,
)

# Spring Core 서비스 주소 (콜백 전송 대상)
# Docker 환경에서는 .env의 CORE_BASE_URL=http://host.docker.internal:8080 사용
CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8080")

# 청킹 파라미터 (Spring API에서 받지 않고 여기서 고정)
MAX_CHUNK_SIZE = 600   # 섹션 최대 글자 수 (초과 시 추가 분할)
MIN_CHUNK_SIZE = 80    # 섹션 최소 글자 수 (미달 시 앞 섹션과 병합)
CHUNK_OVERLAP  = 100   # 섹션 간 겹치는 글자 수 (문맥 연속성 확보)


def _process_v2_sync(doc_id: int, site_id: int, pdf_bytes: bytes) -> None:
    """
    v2 파이프라인 동기 실행 함수 (asyncio.to_thread로 스레드 풀에서 호출됨).

    GPT API 호출과 DB 작업이 포함되므로 블로킹 함수로 구현.
    asyncio 이벤트 루프를 블로킹하지 않기 위해 별도 스레드에서 실행.
    """
    # ── STEP 1. PDF → 마크다운 변환 ──────────────────────────────────────────
    # pymupdf4llm을 사용해 PDF를 마크다운 텍스트로 변환.
    # 헤더(#, ##, ###)가 보존되어 이후 섹션 파싱에 활용됨.
    print(f"[processor] doc_id={doc_id} | 마크다운 변환 중...", flush=True)
    md_text = pdf_to_markdown(pdf_bytes)
    if not md_text or len(md_text.strip()) < 30:
        raise RuntimeError("PDF 마크다운 변환 결과가 비어있거나 너무 짧음")
    print(f"[processor] doc_id={doc_id} | 마크다운 변환 완료 ({len(md_text)}자)", flush=True)

    # ── STEP 2. 마크다운 → 섹션 파싱 (청킹) ─────────────────────────────────
    # 헤더 구조를 기준으로 문서를 섹션 단위로 분할.
    # 각 섹션은 {"section_title": str, "content": str, "level": int} 형태.
    sections = parse_markdown_sections(
        md_text,
        max_chunk_size=MAX_CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"[processor] doc_id={doc_id} | 섹션 파싱 완료: {len(sections)}개", flush=True)

    # ── STEP 3. 섹션별 GPT 요약 + 임베딩 생성 ───────────────────────────────
    # - generate_search_summary: GPT-4o로 섹션 요약 + 키워드 추출
    # - 임베딩 대상: "섹션제목 + 요약 + 키워드" 합성 텍스트
    #   (원문 대신 요약 텍스트를 임베딩하여 검색 정확도 향상)
    # - 임베딩 모델: MODEL_NAME (PDF2db_v2.py에서 정의)
    rows_to_insert = []
    for idx, sec in enumerate(sections):
        content       = sec["content"]
        section_title = sec["section_title"]

        print(f"[processor] doc_id={doc_id} | 청크 {idx+1}/{len(sections)} 요약+임베딩 중...", flush=True)
        summary_result = generate_search_summary(content=content, section_title=section_title)
        summary  = summary_result["summary"]
        keywords = summary_result["keywords"]

        # 요약 + 키워드를 합쳐 임베딩 → 검색 시 의미 기반 매칭 성능 향상
        search_text = f"{section_title} {summary} {' '.join(keywords)}"
        emb = openai_client_v2.embeddings.create(
            model=MODEL_NAME,
            input=search_text,
        ).data[0].embedding

        # metadata: 나중에 디버깅/검색 필터링에 사용할 부가 정보
        meta = Jsonb({
            "chunk_index":    idx,
            "section_level":  sec["level"],   # 헤더 깊이 (# = 1, ## = 2, ...)
            "embed_model":    MODEL_NAME,
            "extractor":      "pymupdf4llm",
            "pipeline_version": "v2",
        })

        rows_to_insert.append((site_id, doc_id, idx, section_title, content, summary, keywords, emb, meta))

    # ── STEP 4. DB 저장 (DELETE → INSERT 원자적 처리) ───────────────────────
    # 재처리(reprocess) 시 기존 청크를 완전히 교체하기 위해
    # INSERT 전에 같은 doc_id의 기존 데이터를 먼저 삭제.
    # 두 작업을 하나의 트랜잭션으로 묶어 부분 저장 방지.
    print(f"[processor] doc_id={doc_id} | DB 저장 중... ({len(rows_to_insert)}개 청크)", flush=True)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM tb_doc_chunk_v2 WHERE doc_id = %s AND site_id = %s",
                (doc_id, site_id),
            )
            for row in rows_to_insert:
                cur.execute(
                    """
                    INSERT INTO tb_doc_chunk_v2
                    (site_id, doc_id, chunk_index, section_title,
                     content, summary, keywords, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                    """,
                    row,
                )
        conn.commit()
    print(f"[processor] doc_id={doc_id} | DB 저장 완료", flush=True)


async def process_pdf_from_url(
    doc_id: int,
    site_id: int,
    storage_url: str,
) -> None:
    """
    Core에서 요청받은 문서를 v2 파이프라인으로 처리하고 결과를 Core에 콜백.

    Spring Core의 FastApiDocumentService.processDocument()가 이 함수를
    BackgroundTask로 등록하여 비동기 실행함.

    상태 전이: PENDING → PROCESSING → COMPLETED | FAILED
      - PROCESSING: PDF 다운로드 완료 후, 실제 처리 시작 전
      - COMPLETED : 모든 청크 DB 저장 완료 후
      - FAILED    : 어느 단계에서든 예외 발생 시
    """
    print(f"[processor] doc_id={doc_id} 처리 시작: {storage_url}", flush=True)
    try:
        # 1. BFF가 저장한 파일 서버에서 PDF 바이트 다운로드
        #    storage_url 예시: http://admin-bff:8081/internal/files/1/abc123.pdf
        async with httpx.AsyncClient() as http:
            resp = await http.get(storage_url)
            resp.raise_for_status()
            pdf_bytes = resp.content
        print(f"[processor] doc_id={doc_id} | PDF 다운로드 완료 ({len(pdf_bytes)} bytes)", flush=True)

        # 2. Core에 PROCESSING 상태 콜백
        #    프론트엔드가 "처리 중" 상태를 즉시 확인할 수 있도록 먼저 알림
        await _callback_core(site_id, doc_id, "PROCESSING", None)
        print(f"[processor] doc_id={doc_id} | Core 콜백 PROCESSING", flush=True)

        # 3. v2 파이프라인 실행 (GPT API + DB 작업 = 블로킹 → 스레드 풀 위임)
        #    asyncio.to_thread: 블로킹 함수를 별도 스레드에서 실행해 이벤트 루프 보호
        await asyncio.to_thread(_process_v2_sync, doc_id, site_id, pdf_bytes)

        # 4. 처리 완료 → Core에 COMPLETED 콜백
        await _callback_core(site_id, doc_id, "COMPLETED", None)
        print(f"[processor] doc_id={doc_id} | 처리 완료 → Core 콜백 COMPLETED", flush=True)

    except Exception as e:
        import traceback
        print(f"[processor] doc_id={doc_id} 처리 실패: {e}", flush=True)
        traceback.print_exc()
        # 실패 사유를 Core에 전달 → tb_document.failed_reason 컬럼에 저장됨
        await _callback_core(site_id, doc_id, "FAILED", str(e))


async def _callback_core(
    site_id: int, doc_id: int, status: str, failed_reason: str | None
) -> None:
    """
    Spring Core의 문서 상태 업데이트 엔드포인트에 PATCH 요청을 보내는 함수.

    대상 엔드포인트: PATCH /internal/v1/sites/{siteId}/documents/{docId}/status
    요청 바디: {"status": "PROCESSING"|"COMPLETED"|"FAILED", "failed_reason": "..."}

    콜백 실패 시 예외를 상위로 전파하지 않고 로그만 남김.
    (콜백 실패가 전체 처리 결과에 영향을 주지 않도록)
    """
    payload: dict = {"status": status}
    if failed_reason:
        payload["failed_reason"] = failed_reason

    try:
        async with httpx.AsyncClient() as http:
            resp = await http.patch(
                f"{CORE_BASE_URL}/internal/v1/sites/{site_id}/documents/{doc_id}/status",
                json=payload,
                timeout=10.0,
            )
            resp.raise_for_status()
    except Exception as e:
        print(f"[processor] Core 콜백 실패 doc_id={doc_id} status={status}: {e}", flush=True)
