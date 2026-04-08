"""
구조화 RAG v2 — PDF 처리 파이프라인

흐름:
  PDF bytes → pymupdf4llm(마크다운 변환) → # 헤더 기반 섹션 파싱
  → GPT-4o 검색용 요약 생성 → 요약 임베딩 → DB 저장

기존 PDF2db.py(v1)와 독립적으로 동작하며, tb_doc_chunk_v2 테이블에 저장.

개선점:
1. 기존의 단일 title -> 계층적 title
2. overlap이 글자 단위에서 문장 단위로 개선
3. 임베딩 시 section_title + summary + keywords 조합
"""
from __future__ import annotations

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
SUMMARY_MODEL_NAME = "gpt-5-mini"
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

    # 소제목 패턴(1), 2), …)을 ### 헤더로 변환 → 의미 단위 청크 분할 지원
    md_text = re.sub(r'^(\d+)\)\s*', r'### \1) ', md_text, flags=re.MULTILINE)

    return md_text


# ── 2) 마크다운 → 구조화 섹션 파싱 ──────────────────────────────────────────

def parse_markdown_sections(
    md_text: str,
    max_chunk_size: int = 600,
    min_chunk_size: int = 80,
    sentence_overlap: int = 2,
) -> List[Dict[str, Any]]:
    """마크다운 헤더(#, ##, ###, ####)를 기준으로 계층형 섹션 분리.

    Returns:
        [{"section_title": "...", "content": "...", "level": 1}, ...]

    개선점:
    - 헤더 경로를 계층형으로 유지 (예: 관람안내 > 운영시간 > 야간개장)
    - 긴 섹션은 문단 기준 분할
    - overlap은 문자 단위가 아니라 문장 단위로 붙임
    - 너무 짧은 청크는 앞 청크와 병합
    """
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    numbered_header_pattern = re.compile(r"^(\d+(?:\.\d+)*[.)]?)\s+(.+?)\s*$", re.MULTILINE)

    def split_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        # 한국어/영어 혼합 대응용
        parts = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=니다\.)\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def get_sentence_overlap(text: str, keep_last_n: int = 2) -> str:
        sents = split_sentences(text)
        if not sents:
            return text[-100:].strip()  # fallback
        return " ".join(sents[-keep_last_n:]).strip()

    def split_long_paragraph(paragraph: str, max_size: int) -> List[str]:
        """문단 하나가 너무 길면 문장 기준으로 재분할. 최후에는 글자수 기준."""
        paragraph = paragraph.strip()
        if len(paragraph) <= max_size:
            return [paragraph]

        sents = split_sentences(paragraph)
        if not sents:
            # 문장 분리가 안 되면 글자수 기준 fallback
            return [paragraph[i:i + max_size] for i in range(0, len(paragraph), max_size)]

        out = []
        buf = ""

        for s in sents:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_size:
                buf = f"{buf} {s}"
            else:
                out.append(buf.strip())
                if len(s) <= max_size:
                    buf = s
                else:
                    # 문장 하나가 너무 길면 글자수 기준 fallback
                    for i in range(0, len(s), max_size):
                        piece = s[i:i + max_size].strip()
                        if piece:
                            out.append(piece)
                    buf = ""

        if buf.strip():
            out.append(buf.strip())

        return out

    sections: List[Dict[str, Any]] = []

    matches = []
    for m in header_pattern.finditer(md_text):
        matches.append(("md", m.start(), m))

    for m in numbered_header_pattern.finditer(md_text):
        matches.append(("num", m.start(), m))

    matches.sort(key=lambda x: x[1])

    if not matches:
        raw = md_text.strip()
        if not raw:
            return []
        
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        
        if len(paragraphs) == 1:
            return [{
                "section_title": "본문",
                "content": raw,
                "level": 0,
            }]
        
        return [
            {
                "section_title": f"본문 {i+1}",
                "content": paragraph,
                "level": 0,
            }
            for i, paragraph in enumerate(paragraphs)
        ]
    title_stack: List[str] = []

    first_start = matches[0][1]
    preamble = md_text[:first_start].strip()
    if preamble:
        sections.append({
            "section_title": "서론",
            "content": preamble,
            "level": 0,
        })

    for i, (kind, _, match) in enumerate(matches):
        if kind == "md":
            level = len(match.group(1))
            title = match.group(2).strip()
        else:
            num_token = match.group(1).strip()
            title = match.group(2).strip()
            level = num_token.count(".") + 1
        
        if len(title) > 80 or len(title.split()) > 12:
            title = f"섹션 {i+1}"
        if not title:
            title = f"섹션 {i+1}"

        content_start = match.end()
        content_end = matches[i + 1][1] if i + 1 < len(matches) else len(md_text)
        content = md_text[content_start:content_end].strip()

        if not content:
            continue

        title_stack = title_stack[:level - 1]
        title_stack.append(title)
        section_title = " > ".join(title_stack)

        sections.append({
            "section_title": section_title,
            "content": content,
            "level": level,
        })

    # 너무 긴 섹션은 문단 기준으로 분할
    split_sections: List[Dict[str, Any]] = []
    for sec in sections:
        content = sec["content"].strip()
        if len(content) <= max_chunk_size:
            split_sections.append(sec)
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

        # 리스트 블록이 쪼개지지 않도록 약간 보존
        normalized_blocks: List[str] = []
        list_buf: List[str] = []

        def flush_list_buf():
            nonlocal list_buf
            if list_buf:
                normalized_blocks.append("\n".join(list_buf).strip())
                list_buf = []

        for p in paragraphs:
            lines = [ln.rstrip() for ln in p.splitlines() if ln.strip()]
            is_list_block = bool(lines) and bool(re.match(r"^([-*•]|\d+[.)])\s+", lines[0].strip()))

            if is_list_block:
                list_buf.append("\n".join(lines))
            else:
                flush_list_buf()
                normalized_blocks.append(p)

        flush_list_buf()

        blocks: List[str] = []
        for block in normalized_blocks:
            if len(block) <= max_chunk_size:
                blocks.append(block)
            else:
                blocks.extend(split_long_paragraph(block, max_chunk_size))

        current = ""
        part_num = 0

        for block in blocks:
            candidate = f"{current}\n\n{block}".strip() if current else block
            if current and len(candidate) > max_chunk_size:
                part_num += 1
                chunk_title = sec["section_title"] if part_num == 1 else f"{sec['section_title']} (part {part_num})"
                split_sections.append({
                    "section_title": chunk_title,
                    "content": current.strip(),
                    "level": sec["level"],
                })

                overlap_buf = get_sentence_overlap(current, keep_last_n=sentence_overlap)
                current = f"{overlap_buf}\n\n{block}".strip() if overlap_buf else block
            else:
                current = candidate

        if current.strip():
            part_num += 1
            chunk_title = sec["section_title"] if part_num == 1 else f"{sec['section_title']} (part {part_num})"
            split_sections.append({
                "section_title": chunk_title,
                "content": current.strip(),
                "level": sec["level"],
            })

    # 너무 짧은 청크는 앞 청크와 병합
    merged: List[Dict[str, Any]] = []
    for sec in split_sections:
        if merged and len(sec["content"]) < min_chunk_size:
            merged[-1]["content"] = f"{merged[-1]['content']}\n\n{sec['content']}".strip()
        else:
            merged.append(sec)

    # 마지막도 너무 짧으면 앞과 병합
    if len(merged) > 1 and len(merged[-1]["content"]) < min_chunk_size:
        merged[-2]["content"] = f"{merged[-2]['content']}\n\n{merged[-1]['content']}".strip()
        merged.pop()

    return merged

# ── 3) GPT 검색용 요약 생성 ──────────────────────────────────────────────────

def generate_search_summary(
    content: str,
    section_title: str,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:
    """GPT로 검색용 요약 + 키워드를 생성.

    Returns:
        {"summary": "...", "keywords": ["...", ...]}
    """
    import re
    
    safe_content = content if content is not None else ""
    safe_title = section_title if section_title is not None else "(없음)"

    def _sanitize(text: str) -> str:
        # 1) 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # 2) 서로게이트 문자 제거 (JSON 직렬화 깨뜨리는 원인)
        text = text.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        # 3) JSON에서 문제 되는 특수 유니코드 제거 (BOM, 제로폭 문자 등)
        text = re.sub(r'[\ufffe\uffff\ufeff\ufdd0-\ufdef]', '', text)
        # 4) JSON 직렬화 가능한지 최종 확인
        try:
            json.dumps(text)
        except (ValueError, UnicodeEncodeError):
            text = text.encode('ascii', errors='ignore').decode('ascii')
        return text

    content = _sanitize(safe_content)
    section_title = _sanitize(safe_title)
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
  "summary": "4-6문장으로 이 텍스트의 핵심 내용을 요약. 텍스트에 등장하는 모든 인물, 사건, 장소를 빠짐없이 포함하세요.",
  "keywords": [
    "텍스트에 등장하는 인물명(왕, 왕비, 신하 등) 전부",
    "장소명, 건물명 전부",
    "사건명, 연도 전부",
    "입장료·수량 등 숫자 정보",
    "핵심 개념어",
    "총 20-25개"
  ]
}}

중요: keywords에는 텍스트에 나오는 고유명사(인물, 장소, 사건)를 모두 포함해야 합니다. 누락하지 마세요."""

    # 요청 전 JSON 직렬화 검증
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        json.dumps(messages, ensure_ascii=False)
    except (ValueError, UnicodeEncodeError) as e:
        print(f"[summary] WARNING: 메시지 JSON 직렬화 실패, ASCII로 폴백: {e}", flush=True)
        user_prompt = user_prompt.encode('ascii', errors='ignore').decode('ascii')
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        #temperature=0.1, temperature 지원안함
        #max_tokens=500, gpt-5-mini는 max_tokens 대신 max_completion_tokens 사용
        # 제한을 안두는게 맞을듯max_completion_tokens=2000, #질문도 포함해서 4096 토큰까지 지원하므로 알아서 설정하게
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    print(f"[summary] section='{section_title}' | finish_reason={finish_reason} | raw_len={len(raw_content) if raw_content else 0}", flush=True)
    if finish_reason == "length":
        print(f"[summary] WARNING: 토큰 제한으로 응답이 잘림! section='{section_title}'", flush=True)
    if raw_content is None:
        print(f"[summary] ERROR: raw_content is None for section='{section_title}'", flush=True)
        return {"summary": content[:200], "keywords": []}
    raw = raw_content.strip()
    print(f"[summary] raw response (first 300): {raw[:300]}", flush=True)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[summary] ERROR: JSON 파싱 실패 for section='{section_title}'", flush=True)
        result = {"summary": raw, "keywords": []}

    # 안전한 기본값
    if "summary" not in result:
        result["summary"] = content[:200]
    if "keywords" not in result or not isinstance(result["keywords"], list):
        result["keywords"] = []
    else:
        result["keywords"] = [
            str(keyword).strip()
            for keyword in result["keywords"]
            if str(keyword).strip()
        ]

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

def build_search_text(section_title: str, summary: str, keywords: List[str]) -> str:
    return f"""
[section]
{section_title}

[summary]
{summary}

[keywords]
{' '.join(keywords)}
""".strip()

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

        # 3-a) PROCESSING 상태 즉시 커밋 (폴링 시 반영되도록)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE tb_document SET status = 'PROCESSING' WHERE doc_id = %s",
                    (doc_id,),
                )
            conn.commit()

        # 3-b) DB 밖에서 모든 청크의 요약/임베딩을 미리 준비
        rows_to_insert = []
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
            search_text = build_search_text(
                section_title=section_title, 
                summary=summary, 
                keywords=keywords
            )
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

            rows_to_insert.append((
                site_id, doc_id, idx, section_title,
                content, summary, keywords, emb, Jsonb(meta),
            ))

        # 3-c) DELETE → INSERT → COMPLETED 를 단일 트랜잭션으로 원자적 실행
        with get_conn() as conn:
            with conn.cursor() as cur:
                if purge_old_chunks:
                    cur.execute(
                        "DELETE FROM tb_doc_chunk_v2 WHERE doc_id = %s",
                        (doc_id,),
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
        except Exception as update_err:
            print(f"[PDF-v2] doc_id={doc_id} | 상태 갱신 실패: {update_err}")
        raise
