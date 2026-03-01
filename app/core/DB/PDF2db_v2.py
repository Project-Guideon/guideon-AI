"""
구조화 RAG v2 — PDF 처리 파이프라인

흐름:
  PDF bytes → pymupdf4llm(마크다운 변환) → # 헤더 기반 섹션 파싱
  → GPT-4o-mini 검색용 요약 생성 → 요약 임베딩 → DB 저장

기존 PDF2db.py(v1)와 독립적으로 동작하며, tb_doc_chunk_v2 테이블에 저장.
"""
from __future__ import annotations

import io
import json
import re
import tempfile
from typing import List, Dict, Any

import pymupdf4llm
from psycopg.types.json import Jsonb

from app.core.DB.connect_db import get_conn

import os
from openai import OpenAI

MODEL_NAME = "text-embedding-3-small"
SUMMARY_MODEL_NAME = "gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 1) PDF → 마크다운 변환 ──────────────────────────────────────────────────

def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """PDF bytes를 pymupdf4llm으로 마크다운 텍스트로 변환."""
    # pymupdf4llm은 파일 경로를 요구하므로 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    return md_text


# ── 2) 마크다운 → 구조화 섹션 파싱 ──────────────────────────────────────────

def parse_markdown_sections(
    md_text: str,
    max_chunk_size: int = 600,
    min_chunk_size: int = 80,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """마크다운 헤더(#, ##, ###)를 기준으로 섹션 분리.

    Returns:
        [{"section_title": "...", "content": "...", "level": 1}, ...]

    - 헤더가 없는 긴 텍스트는 max_chunk_size 기준으로 단락 분리
    - 너무 짧은 섹션(< min_chunk_size)은 다음 섹션과 병합
    """
    # 마크다운 헤더 패턴: # Title, ## Title, ### Title
    header_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    sections: List[Dict[str, Any]] = []
    last_pos = 0
    last_title = ""
    last_level = 0

    for match in header_pattern.finditer(md_text):
        # 헤더 이전 텍스트를 이전 섹션의 content로
        before_text = md_text[last_pos:match.start()].strip()
        if before_text:
            sections.append({
                "section_title": last_title,
                "content": before_text,
                "level": last_level,
            })

        last_title = match.group(2).strip()
        last_level = len(match.group(1))  # # = 1, ## = 2, ### = 3
        last_pos = match.end()

    # 마지막 섹션
    remaining = md_text[last_pos:].strip()
    if remaining:
        sections.append({
            "section_title": last_title,
            "content": remaining,
            "level": last_level,
        })

    # 헤더가 하나도 없으면 전체를 하나의 섹션으로
    if not sections:
        sections.append({
            "section_title": "",
            "content": md_text.strip(),
            "level": 0,
        })

    # 짧은 섹션 병합
    merged: List[Dict[str, Any]] = []
    for sec in sections:
        if merged and len(merged[-1]["content"]) < min_chunk_size:
            # 이전 섹션이 너무 짧으면 현재와 병합
            prev = merged[-1]
            prev["content"] += "\n\n" + sec["content"]
            if sec["section_title"] and not prev["section_title"]:
                prev["section_title"] = sec["section_title"]
        else:
            merged.append(sec)

    # 마지막 섹션이 너무 짧으면 이전과 병합
    if len(merged) > 1 and len(merged[-1]["content"]) < min_chunk_size:
        merged[-2]["content"] += "\n\n" + merged[-1]["content"]
        merged.pop()

    # 너무 긴 섹션은 단락 기준으로 분할 (overlap 포함)
    final: List[Dict[str, Any]] = []
    for sec in merged:
        if len(sec["content"]) <= max_chunk_size:
            final.append(sec)
        else:
            # 단락(\n\n) 기준으로 분할
            paragraphs = sec["content"].split("\n\n")
            current = ""
            overlap_buf = ""   # 이전 chunk의 마지막 단락(들)을 보관
            part_num = 0
            for para in paragraphs:
                if current and len(current) + len(para) + 2 > max_chunk_size:
                    part_num += 1
                    title = f"{sec['section_title']} (part {part_num})" if sec["section_title"] else ""
                    final.append({
                        "section_title": title,
                        "content": current.strip(),
                        "level": sec["level"],
                    })
                    # overlap: current의 끝부분을 다음 chunk 앞에 붙임
                    overlap_buf = current[-chunk_overlap:].strip() if chunk_overlap else ""
                    current = (overlap_buf + "\n\n" + para).strip() if overlap_buf else para
                else:
                    current = current + "\n\n" + para if current else para

            if current.strip():
                part_num += 1
                title = sec["section_title"]
                if part_num > 1:
                    title = f"{sec['section_title']} (part {part_num})" if sec["section_title"] else ""
                final.append({
                    "section_title": title,
                    "content": current.strip(),
                    "level": sec["level"],
                })

    return final


# ── 3) GPT 검색용 요약 생성 ──────────────────────────────────────────────────

def generate_search_summary(
    content: str,
    section_title: str,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """GPT로 검색용 요약 + 키워드를 생성.

    Returns:
        {"summary": "...", "keywords": ["...", ...]}
    """
    system_prompt = (
        "당신은 관광 안내 문서의 검색 최적화 전문가입니다.\n"
        "주어진 텍스트를 분석하여 검색에 최적화된 요약과 키워드를 생성하세요.\n"
        "반드시 JSON 형식으로만 응답하세요."
    )

    user_prompt = f"""다음 텍스트에 대해 검색용 요약을 생성해주세요.

섹션 제목: {section_title or '(없음)'}

텍스트:
{content[:3000]}

아래 JSON 형식으로만 응답하세요:
{{
  "summary": "3-4문장으로 이 텍스트의 핵심 내용을 요약. 텍스트에 등장하는 모든 인물, 사건, 장소를 빠짐없이 포함하세요.",
  "keywords": [
    "텍스트에 등장하는 인물명(왕, 왕비, 신하 등) 전부",
    "장소명, 건물명 전부",
    "사건명, 연도 전부",
    "입장료·수량 등 숫자 정보",
    "핵심 개념어",
    "총 10-15개"
  ]
}}

중요: keywords에는 텍스트에 나오는 고유명사(인물, 장소, 사건)를 모두 포함해야 합니다. 누락하지 마세요."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"summary": raw, "keywords": []}

    # 안전한 기본값
    if "summary" not in result:
        result["summary"] = content[:200]
    if "keywords" not in result or not isinstance(result["keywords"], list):
        result["keywords"] = []

    return result


# ── 4) 전체 파이프라인 ──────────────────────────────────────────────────────

def create_doc_record(
    original_name: str,
    file_hash: str,
    site_id: int,
    file_size: int,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> int:
    """tb_document에 PENDING 레코드 생성 (v1과 동일한 테이블 공유)."""
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
            doc_id = cur.fetchone()[0]
        conn.commit()
    return doc_id


def process_pdf_bytes_v2(
    doc_id: int,
    pdf_bytes: bytes,
    site_id: int,
    max_chunk_size: int = 600,
    min_chunk_size: int = 80,
    purge_old_chunks: bool = True,
) -> None:
    """구조화 RAG v2 PDF 처리 파이프라인.

    PDF → 마크다운 → 섹션 파싱 → GPT 요약 → 임베딩 → DB 저장
    """
    try:
        # 1) PDF → 마크다운
        print(f"[PDF-v2] doc_id={doc_id} | 마크다운 변환 시작...")
        md_text = pdf_to_markdown(pdf_bytes)

        if not md_text or len(md_text.strip()) < 30:
            raise RuntimeError("PDF 마크다운 변환 결과가 비어있거나 너무 짧음")

        # 2) 마크다운 → 구조화 섹션 파싱
        sections = parse_markdown_sections(
            md_text,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
        )
        print(f"[PDF-v2] doc_id={doc_id} | 섹션 수={len(sections)}")

        # 3) DB 저장
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tb_document SET status = 'PROCESSING' WHERE doc_id = %s",
                    (doc_id,),
                )

                if purge_old_chunks:
                    cur.execute(
                        "DELETE FROM tb_doc_chunk_v2 WHERE doc_id = %s",
                        (doc_id,),
                    )

                for idx, sec in enumerate(sections):
                    content = sec["content"]
                    section_title = sec["section_title"]

                    # GPT 요약 생성
                    print(f"[PDF-v2] doc_id={doc_id} | 청크 {idx+1}/{len(sections)} 요약 생성 중...")
                    summary_result = generate_search_summary(
                        content=content,
                        section_title=section_title,
                    )
                    summary = summary_result["summary"]
                    keywords = summary_result["keywords"]

                    # 검색용 텍스트 조합 → 임베딩
                    search_text = f"{section_title} {summary} {' '.join(keywords)}"
                    emb = client.embeddings.create(
                        model=MODEL_NAME,
                        input=search_text,
                    ).data[0].embedding

                    meta = {
                        "chunk_index": idx,
                        "section_level": sec["level"],
                        "embed_model": MODEL_NAME,
                        "summary_model": SUMMARY_MODEL_NAME,
                        "extractor": "pymupdf4llm",
                        "pipeline_version": "v2",
                    }

                    cur.execute(
                        """
                        INSERT INTO tb_doc_chunk_v2
                        (site_id, doc_id, chunk_index, section_title,
                         content, summary, keywords, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            site_id, doc_id, idx, section_title,
                            content, summary, keywords, emb, Jsonb(meta),
                        ),
                    )

                cur.execute(
                    """UPDATE tb_document
                       SET status = 'COMPLETED', processed_at = NOW()
                       WHERE doc_id = %s""",
                    (doc_id,),
                )
            conn.commit()

        print(f"[PDF-v2] doc_id={doc_id} | 처리 완료 ({len(sections)}개 청크)")

    except Exception as e:
        print(f"[PDF-v2] doc_id={doc_id} | 처리 실패: {e}")
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
