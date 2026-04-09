"""
구조화 RAG v2 — PDF 처리 파이프라인

흐름:
  PDF bytes → pymupdf4llm(마크다운 변환)
           → fitz 보조 분석(인라인 소제목 감지)
           → # 헤더 기반 섹션 파싱
           → GPT 검색용 요약 생성 → 요약 임베딩 → DB 저장

기존 PDF2db.py(v1)와 독립적으로 동작하며, tb_doc_chunk_v2 테이블에 저장.

개선점:
1. 하이브리드 방식: pymupdf4llm(텍스트 추출/정렬) + fitz(인라인 소제목 감지 + fallback)
2. 같은 크기에서 소수 폰트 감지 → 인라인 소제목을 ### 헤더로 분리
3. 한자 주석(인라인)과 진짜 소제목(줄 시작)을 x좌표로 구분
4. overlap이 글자 단위에서 문장 단위로 개선
5. 임베딩 시 section_title + summary + keywords 조합
6. 페이지별 pymupdf4llm↔fitz 비교 → 텍스트 누락 시 fitz fallback
"""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter, defaultdict
from typing import List, Dict, Any, Set, Tuple

import fitz  # PyMuPDF
import pymupdf4llm
from psycopg.types.json import Jsonb

from app.core.DB.connect_db import get_conn

import os
from openai import OpenAI

MODEL_NAME = "text-embedding-3-small"
SUMMARY_MODEL_NAME = "gpt-5-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── 1) PDF → 마크다운 변환 (하이브리드: pymupdf4llm + fitz 보조) ────────────

def _detect_inline_headings(pdf_bytes: bytes) -> Tuple[Set[str], Set[str]]:
    """fitz로 PDF를 분석하여 (인라인 소제목, 소수폰트 소제목) 텍스트 목록을 반환.

    pymupdf4llm이 감지 못하는 인라인 소제목을 찾아냄.
    예: "경복궁의 명칭   경복궁은 조선 왕조가..." 에서 "경복궁의 명칭" 부분.

    Returns:
        (inline_heading_texts, heading_font_texts)
        - inline_heading_texts: 본문 줄에 끼어있는 소제목 (### 헤더로 분리할 대상)
        - heading_font_texts: 소수 폰트의 모든 소제목 텍스트 (##### 헤더 라인 분리용)

    판별 기준:
    1. 같은 크기 그룹에서 소수 폰트(minority font) 감지
       - 글자 수가 다수 폰트의 20% 미만
       - 평균 span 길이 4~40자, span 50개 이하
    2. 인라인(한자 주석) vs 소제목 구분
       - 줄의 시작 위치에 있으면 → 소제목
       - 줄 중간에 끼어있으면 → 인라인 주석 (무시)
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with fitz.open(tmp_path) as doc:
            all_spans = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                for block in page.get_text("dict")["blocks"]:
                    if block["type"] != 0:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text:
                                continue
                            all_spans.append({
                                "text": text,
                                "size": round(span["size"], 1),
                                "font": span.get("font", ""),
                                "bbox": span["bbox"],
                                "page": page_idx,
                            })
    finally:
        os.unlink(tmp_path)

    if not all_spans:
        return set(), set()

    # ── 본문 크기 판별 (서술형 점수 기반) ──
    size_groups: dict = defaultdict(lambda: {
        "total_chars": 0, "span_count": 0, "period_count": 0,
    })
    for s in all_spans:
        g = size_groups[s["size"]]
        g["total_chars"] += len(s["text"])
        g["span_count"] += 1
        g["period_count"] += len(re.findall(r"[.!?。]", s["text"]))

    best_size, best_score = 12.0, -1.0
    for size, g in size_groups.items():
        if g["total_chars"] < 50:
            continue
        avg_len = g["total_chars"] / max(g["span_count"], 1)
        period_d = g["period_count"] / max(g["total_chars"], 1) * 1000
        score = g["total_chars"] * (1 + avg_len / 50) * (1 + period_d)
        if score > best_score:
            best_score = score
            best_size = size

    # ── 소수 폰트 감지 ──
    size_font_chars: Dict[float, Counter] = defaultdict(Counter)
    size_font_count: Dict[Tuple[float, str], int] = defaultdict(int)
    for s in all_spans:
        size_font_chars[s["size"]][s["font"]] += len(s["text"])
        size_font_count[(s["size"], s["font"])] += 1

    heading_fonts_by_size: Dict[float, set] = defaultdict(set)
    for sz, font_counter in size_font_chars.items():
        if len(font_counter) < 2:
            continue
        fonts_sorted = font_counter.most_common()
        _, majority_chars = fonts_sorted[0]
        for font, chars in fonts_sorted[1:]:
            sc = size_font_count[(sz, font)]
            avg_len = chars / max(sc, 1)
            if (chars < majority_chars * 0.2
                    and 4 <= avg_len <= 40
                    and sc <= 50):
                heading_fonts_by_size[sz].add(font)

    if not heading_fonts_by_size:
        return set(), set()

    # ── 소수 폰트의 모든 소제목 텍스트 수집 (##### 헤더 라인 분리용) ──
    heading_font_texts: Set[str] = set()
    for s in all_spans:
        sz = s["size"]
        if s["font"] in heading_fonts_by_size.get(sz, set()):
            text = s["text"].strip()
            if 4 <= len(text) <= 60:
                heading_font_texts.add(text)

    # ── 인라인 vs 소제목 구분 (x좌표 기반) ──
    Y_TOL = 3.0
    line_fonts: Dict[Tuple[int, float], Counter] = defaultdict(Counter)
    line_min_x: Dict[Tuple[int, float], float] = {}
    for s in all_spans:
        y_bucket = round(s["bbox"][1] / Y_TOL) * Y_TOL
        key = (s["page"], y_bucket)
        line_fonts[key][s["font"]] += len(s["text"])
        x0 = s["bbox"][0]
        if key not in line_min_x or x0 < line_min_x[key]:
            line_min_x[key] = x0

    inline_heading_texts: Set[str] = set()
    for s in all_spans:
        sz = s["size"]
        if s["font"] not in heading_fonts_by_size.get(sz, set()):
            continue

        y_bucket = round(s["bbox"][1] / Y_TOL) * Y_TOL
        key = (s["page"], y_bucket)
        fonts_on_line = line_fonts[key]
        other_chars = sum(c for f, c in fonts_on_line.items() if f != s["font"])

        # 같은 줄에 다른 폰트의 본문이 없으면 → pymupdf4llm이 이미 처리
        if other_chars <= len(s["text"]):
            continue

        # 줄 시작 위치에 있는 소수 폰트 → 인라인 소제목
        min_x = line_min_x.get(key, 0)
        if abs(s["bbox"][0] - min_x) < 5.0:
            inline_heading_texts.add(s["text"])

    return inline_heading_texts, heading_font_texts


def _inject_inline_headings(md_text: str, heading_texts: Set[str]) -> str:
    """pymupdf4llm 마크다운에서 인라인 소제목을 ### 헤더로 분리.

    "경복궁의 명칭 경복궁은 조선 왕조가..." 형태를
    "### 경복궁의 명칭\n\n경복궁은 조선 왕조가..." 로 변환.
    """
    if not heading_texts:
        return md_text

    # 긴 텍스트부터 매칭 (짧은 것이 긴 것의 부분문자열일 수 있으므로)
    sorted_headings = sorted(heading_texts, key=len, reverse=True)

    for heading in sorted_headings:
        escaped = re.escape(heading)
        # 공백 1개 이상이면 매칭 (pymupdf4llm이 공백을 1개로 합치는 경우 대응)
        pattern = re.compile(
            rf"^({escaped})\s+(\S)",
            re.MULTILINE,
        )
        md_text = pattern.sub(rf"### \1\n\n\2", md_text)

    return md_text


def _split_header_body_lines(md_text: str, heading_font_texts: Set[str]) -> str:
    """##### 제목본문... 형태를 ##### 제목\\n\\n본문... 으로 분리.

    pymupdf4llm이 h5 헤더의 제목과 본문을 줄바꿈 없이 한 줄에 출력하는 문제 대응.
    fitz에서 추출한 소수 폰트 소제목 텍스트를 기준으로 정확한 분리 지점을 찾는다.
    """
    if not heading_font_texts:
        return md_text

    # 긴 텍스트부터 매칭 (짧은 것이 긴 것의 부분문자열일 수 있으므로)
    sorted_headings = sorted(heading_font_texts, key=len, reverse=True)

    header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def _try_split(m: re.Match) -> str:
        hashes = m.group(1)
        text = m.group(2)

        # 짧으면 정상적인 제목 → 그대로
        if len(text) <= 80:
            return m.group(0)

        # fitz 소제목 텍스트와 매칭하여 분리 지점 결정
        for heading in sorted_headings:
            if text.startswith(heading) and len(text) > len(heading) + 1:
                body = text[len(heading):].strip()
                if body:
                    return f"{hashes} {heading}\n\n{body}"

        return m.group(0)

    return header_re.sub(_try_split, md_text)


def _extract_fitz_page_text(doc: fitz.Document, page_idx: int) -> str:
    """fitz로 단일 페이지의 plain text를 추출 (비교용)."""
    page = doc[page_idx]
    return page.get_text("text").strip()


def _fitz_page_to_markdown(page) -> str:
    """fitz로 단일 페이지를 마크다운(헤더 포함)으로 변환.

    page.get_text("text") 대신 사용하여 폰트 크기/종류 분석으로
    헤더 구조를 보존한 마크다운을 생성한다.

    헤더 판별 기준:
    - 본문보다 1.8배 이상 큰 폰트 → ### (주요 섹션)
    - 본문보다 1.1배 이상 큰 폰트 → #### (부제목)
    - 본문과 같은 크기이지만 소수 폰트 → ##### (소제목)
    """
    blocks = page.get_text("dict")["blocks"]

    # ── 1) 전체 span 수집 (폰트 분석용) ──
    all_spans: List[Dict] = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    all_spans.append({
                        "text": text,
                        "size": round(span["size"], 1),
                        "font": span.get("font", ""),
                    })

    if not all_spans:
        return ""

    # ── 2) 본문 크기 판별 (가장 많은 글자 수의 크기) ──
    size_chars: Counter = Counter()
    for s in all_spans:
        size_chars[s["size"]] += len(s["text"])
    body_size = size_chars.most_common(1)[0][0]

    # ── 3) 본문 크기에서 소수 폰트 감지 (소제목 폰트) ──
    font_chars_at_body: Counter = Counter()
    for s in all_spans:
        if s["size"] == body_size:
            font_chars_at_body[s["font"]] += len(s["text"])

    heading_fonts: set = set()
    if len(font_chars_at_body) >= 2:
        _, majority_chars = font_chars_at_body.most_common(1)[0]
        for font, chars in font_chars_at_body.items():
            if chars < majority_chars * 0.3:
                heading_fonts.add(font)

    # ── 4) 라인별 처리 → 마크다운 생성 ──
    output: List[str] = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line_data in block["lines"]:
            texts: List[str] = []
            dom_size = 0.0
            dom_font = ""
            dom_len = 0

            for span in line_data["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                texts.append(text)
                if len(text) > dom_len:
                    dom_len = len(text)
                    dom_size = round(span["size"], 1)
                    dom_font = span.get("font", "")

            line_text = " ".join(texts).strip()
            if not line_text:
                continue

            # 페이지 번호 스킵 (숫자만 1~3자리)
            if re.match(r"^\d{1,3}$", line_text):
                continue

            # 헤더 판별
            header_level = 0
            tlen = len(line_text)

            if dom_size > body_size * 1.8 and tlen <= 40:
                header_level = 3   # ### 주요 섹션
            elif dom_size > body_size * 1.1 and tlen <= 40 and size_chars[dom_size] < 80:
                header_level = 4   # #### 부제목 (한자 제목 등)
            elif dom_font in heading_fonts and tlen <= 60:
                header_level = 5   # ##### 소제목

            if header_level:
                output.append(f"{'#' * header_level} {line_text}")
            else:
                output.append(line_text)

    # ── 5) 후처리: 블록 순서 문제로 뒤에 온 섹션 헤더(###, ####)를 앞으로 이동 ──
    # PDF 레이아웃에서 페이지 제목이 별도 블록에 있어 fitz가 본문 뒤에 반환하는 경우 대응
    if len(output) > 2:
        trailing_headers: List[str] = []
        while output and re.match(r"^#{3,4}\s+", output[-1]):
            trailing_headers.insert(0, output.pop())
        if trailing_headers:
            output = trailing_headers + output

    return "\n\n".join(output)


def _is_page_content_sufficient(page_md: str, fitz_text: str, min_ratio: float = 0.3) -> bool:
    """pymupdf4llm 결과가 fitz 대비 충분한 텍스트를 포함하는지 판정.

    - fitz 텍스트가 50자 미만이면 해당 페이지는 지도/이미지 페이지로 간주 → 충분
    - pymupdf4llm 결과가 fitz 텍스트 길이의 min_ratio 미만이면 → 불충분
    """
    # 마크다운에서 헤더 기호(#)와 빈 줄 제거 후 순수 텍스트 길이 측정
    clean_md = re.sub(r"^#{1,6}\s+.*$", "", page_md, flags=re.MULTILINE).strip()
    fitz_len = len(fitz_text)

    if fitz_len < 50:
        # fitz도 텍스트가 거의 없으면 이미지/지도 페이지 → 스킵
        return True

    return len(clean_md) >= fitz_len * min_ratio


def pdf_to_markdown(pdf_bytes: bytes) -> str:
    """PDF bytes를 하이브리드 방식으로 마크다운 텍스트로 변환.

    1단계: pymupdf4llm으로 페이지별 마크다운 생성
    2단계: fitz로 페이지별 텍스트 추출 + 인라인 소제목 감지
    3단계: 페이지별로 pymupdf4llm 결과가 부족하면 fitz 텍스트로 대체
    4단계: 감지된 인라인 소제목을 ### 헤더로 분리
    """
    # Step 1: pymupdf4llm 페이지별 마크다운
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        page_chunks = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
    finally:
        os.unlink(tmp_path)

    # Step 2: fitz로 페이지별 텍스트 추출 (plain: 비교용, markdown: fallback용)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        with fitz.open(tmp_path) as doc:
            fitz_plain = [_extract_fitz_page_text(doc, i) for i in range(len(doc))]
            fitz_md = [_fitz_page_to_markdown(doc[i]) for i in range(len(doc))]
    finally:
        os.unlink(tmp_path)

    # Step 3: 페이지별 비교 → 부족하면 fitz fallback (헤더 구조 보존)
    final_pages: List[str] = []
    for i, chunk in enumerate(page_chunks):
        page_md = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        fitz_text = fitz_plain[i] if i < len(fitz_plain) else ""

        if _is_page_content_sufficient(page_md, fitz_text):
            final_pages.append(page_md)
        else:
            fallback = fitz_md[i] if i < len(fitz_md) else fitz_text
            print(f"[pdf_to_md] PAGE {i+1}: pymupdf4llm 부족 → fitz fallback "
                  f"(md={len(page_md)}자, fitz={len(fitz_text)}자)", flush=True)
            final_pages.append(fallback)

    md_text = "\n\n".join(final_pages)

    # Step 4: fitz 보조 분석 — 인라인 소제목 감지 + 주입
    inline_headings, heading_font_texts = _detect_inline_headings(pdf_bytes)
    if inline_headings:
        print(f"[pdf_to_md] 인라인 소제목 {len(inline_headings)}개 감지: "
              f"{list(inline_headings)[:5]}", flush=True)

    md_text = _inject_inline_headings(md_text, inline_headings)

    # Step 5: ##### 제목+본문 한 줄 합쳐진 헤더 라인 분리
    if heading_font_texts:
        print(f"[pdf_to_md] 소제목 폰트 텍스트 {len(heading_font_texts)}개 감지: "
              f"{list(heading_font_texts)[:5]}", flush=True)
    md_text = _split_header_body_lines(md_text, heading_font_texts)

    # Step 6: 번호형 소제목(1~2자리 숫자 + 괄호)을 ### 헤더로 변환
    md_text = re.sub(
        r'^(\d{1,2}\))\s+(.{2,30})\s*$',
        r'### \1 \2',
        md_text,
        flags=re.MULTILINE,
    )

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
    # level → title 매핑 (계층 구조 유지용)
    level_titles: Dict[int, str] = {}

    first_start = matches[0][1]
    preamble = md_text[:first_start].strip()
    if preamble:
        sections.append({
            "section_title": "서론",
            "content": preamble,
            "level": 0,
        })

    for i, (kind, _, match) in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # 제목이 너무 길면 본문으로 취급 (pymupdf4llm이 body를 h5로 잘못 변환하는 경우)
        if len(title) > 80:
            content_start = match.start()
            content_end = matches[i + 1][1] if i + 1 < len(matches) else len(md_text)
            body = md_text[content_start:content_end].strip()
            # 이전 섹션에 본문 병합
            if sections and body:
                sections[-1]["content"] = f"{sections[-1]['content']}\n\n{body}".strip()
            continue

        if not title:
            title = f"섹션 {i+1}"

        # 숫자만으로 된 제목(페이지 번호 등)은 계층 구조에서 제외
        if re.match(r"^\d{1,3}$", title):
            content_start = match.end()
            content_end = matches[i + 1][1] if i + 1 < len(matches) else len(md_text)
            body = md_text[content_start:content_end].strip()
            if sections and body:
                sections[-1]["content"] = f"{sections[-1]['content']}\n\n{body}".strip()
            continue

        content_start = match.end()
        content_end = matches[i + 1][1] if i + 1 < len(matches) else len(md_text)
        content = md_text[content_start:content_end].strip()

        # 현재 level 이상의 기존 항목 제거 → 같은 레벨은 형제로 처리
        for lv in [lv for lv in level_titles if lv >= level]:
            del level_titles[lv]
        level_titles[level] = title

        if not content:
            # 하위 헤더만 따르는 상위 헤더: 계층 구조는 유지하되 빈 섹션은 건너뜀
            continue
        section_title = " > ".join(
            level_titles[lv] for lv in sorted(level_titles.keys())
        )

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
                "extractor": "pymupdf4llm+fitz",
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
