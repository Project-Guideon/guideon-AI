#!/usr/bin/env python3
"""
RAG 성능 평가 스크립트 (RAGAS)

비교 대상:
  - System 0: No RAG  (Plain GPT — 할루시네이션 기준선)
  - System 1: V1 RAG  (기본 벡터 검색,  tb_doc_chunk,   site_id=4)
  - System 2: V2 RAG  (LlamaParser + Hybrid Search, tb_doc_chunk_v2, site_id=3)

사전 설치:
  pip install ragas datasets pandas openpyxl langchain-openai

실행 (저장소 루트에서):
  python app/core/data/evaluate_rag.py

출력 (RESULTS_DIR):
  results7/rag_eval_raw.csv      — 문항별 상세 결과
  results7/rag_eval_summary.xlsx — 시스템별 평균 요약 (논문용)
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Dict

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # guideon_AI/
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
TESTSET_PATH  = Path(__file__).parent / "경복궁_RAG_테스트셋.txt"
RESULTS_DIR   = Path(__file__).parent / "results7"
SITE_ID_V1    = 4     # V1 RAG (pdfplumber 청킹+오버랩, tb_doc_chunk)
SITE_ID_V2    = 3     # V2 RAG (LlamaParser+요약, tb_doc_chunk_v2)
TOP_K         = 5     # 검색 청크 수
LLM_MODEL     = "gpt-4o-mini"   # 답변 생성 모델 (실서비스 동일)
SUMMARY_MODEL = "gpt-5-mini"    # V2 요약 생성 모델 (PDF2db_v2.py 동일, 참고용)
EVAL_DELAY    = 1.0   # API rate-limit 방지 (초)

RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 테스트셋 파싱
# ─────────────────────────────────────────────────────────────────────────────

def parse_testset(path: Path) -> List[Dict[str, str]]:
    """경복궁_RAG_테스트셋.txt → 질문/정답 리스트"""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\d+\)\s*\n질문:\s*(.+?)\n정답:\s*(.+?)(?:\n출처|\n\n)",
        re.DOTALL,
    )
    items = []
    for m in pattern.finditer(text):
        question = m.group(1).strip()
        answer   = re.sub(r"\s+", " ", m.group(2).strip())
        items.append({"question": question, "reference": answer})
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 2. 답변 생성
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_RAG = (
    "당신은 경복궁 관광 안내 도우미입니다.\n"
    "제공된 참고 정보(context)만 근거로 답변하세요.\n"
    "규칙:\n"
    "  - 반드시 한국어로만 답변할 것\n"
    "  - 첫 문장에서 질문의 핵심에 직접 답변할 것 (배경 설명 먼저 하지 말 것)\n"
    "  - 전체 2~5문장으로 간결하게 작성\n"
    "  - context에 없는 내용은 절대 추측하지 말 것\n"
    "  - 정보가 없으면 '관련 정보를 찾을 수 없습니다'라고만 답할 것\n"
    "  - 출처 문서명은 언급하지 않아도 됨"
)

SYSTEM_PROMPT_NO_RAG = (
    "당신은 경복궁 관광 안내 도우미입니다.\n"
    "한국어로 2~5문장으로 자연스럽게 답변하세요."
)


def generate_answer_no_rag(client: OpenAI, question: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
            {"role": "user",   "content": f"질문: {question}"},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_answer_with_context(
    client: OpenAI, question: str, contexts: List[str]
) -> str:
    context_str = "\n\n".join(
        f"[문서 {i+1}]\n{c}" for i, c in enumerate(contexts)
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {
                "role": "user",
                "content": (
                    f"질문: {question}\n\n"
                    f"참고 정보:\n{context_str}\n\n"
                    "위 정보를 바탕으로 답변해 주세요."
                ),
            },
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return (resp.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM 정답 판별
# ─────────────────────────────────────────────────────────────────────────────

def judge_correctness(client: OpenAI, question: str, reference: str, answer: str) -> bool:
    """LLM이 (질문, 정답, 시스템 답변)을 보고 정답 여부를 판단."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 답변 평가자입니다.\n"
                    "질문, 정답, 시스템 답변을 보고 시스템 답변이 정답의 핵심 사실을 포함하는지 판단하세요.\n"
                    "완벽히 일치할 필요는 없으나, 핵심 사실(수치, 명칭, 조건 등)이 맞아야 합니다.\n"
                    "반드시 'correct' 또는 'incorrect' 중 하나만 출력하세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"질문: {question}\n\n"
                    f"정답: {reference}\n\n"
                    f"시스템 답변: {answer}\n\n"
                    "판단 (correct/incorrect):"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=10,
    )
    result = (resp.choices[0].message.content or "").strip().lower()
    return result == "correct"


def run_correctness_eval(df: pd.DataFrame) -> pd.DataFrame:
    """각 행의 no_rag/v1/v2 답변에 대해 LLM 정답 판별 후 컬럼 추가."""
    raw_path = RESULTS_DIR / "rag_eval_raw.csv"

    # 이미 판별된 행 확인 (correct_no_rag 값이 있는 행은 스킵)
    df = df.copy()
    for col in ["correct_no_rag", "correct_v1", "correct_v2"]:
        if col not in df.columns:
            df[col] = None

    # bool 타입 정규화 (이전 실행 결과 재사용 시)
    for col in ["correct_no_rag", "correct_v1", "correct_v2"]:
        df[col] = df[col].map(
            lambda x: x if isinstance(x, bool)
            else (str(x).strip().lower() == "true" if pd.notna(x) and str(x).strip() != "" else None)
        )

    pending = df[df["correct_no_rag"].isna()].index.tolist()
    if not pending:
        print("정답 판별 결과 이미 존재, 재사용합니다.")
        return df

    print(f"[정답판별] 미완료 {len(pending)}건 처리 시작")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    total = len(pending)

    for i, idx in enumerate(pending, 1):
        row = df.loc[idx]
        q, ref = row["question"], row["reference"]
        print(f"[정답판별 {i}/{total}] {q[:40]}...")

        df.at[idx, "correct_no_rag"] = judge_correctness(client, q, ref, row["ans_no_rag"])
        time.sleep(EVAL_DELAY)
        df.at[idx, "correct_v1"] = judge_correctness(client, q, ref, row["ans_v1"])
        time.sleep(EVAL_DELAY)
        df.at[idx, "correct_v2"] = judge_correctness(client, q, ref, row["ans_v2"])
        time.sleep(EVAL_DELAY)

        # 행마다 즉시 저장
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"  → 저장 완료 ({i}/{total})")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. 3가지 시스템 실행 → 원시 결과 수집
# ─────────────────────────────────────────────────────────────────────────────

def run_all_systems(testset: List[Dict]) -> pd.DataFrame:
    from app.core.services.rag_pgvector    import PgVectorRAG, OpenAIEmbedder
    from app.core.services.rag_pgvector_v2 import PgVectorRAG_V2

    raw_path = RESULTS_DIR / "rag_eval_raw.csv"

    # 이미 완료된 행을 dict로 로드 (q_id → row)
    rows: Dict[int, dict] = {}
    if raw_path.exists():
        try:
            existing = pd.read_csv(raw_path, encoding="utf-8-sig")
            # q_id 컬럼이 있는 행만 유효하게 취급 (중간에 껴든 헤더행 제거)
            existing = existing[pd.to_numeric(existing["q_id"], errors="coerce").notna()]
            existing["q_id"] = existing["q_id"].astype(int)
            rows = {int(r["q_id"]): r.to_dict() for _, r in existing.iterrows()}
            if rows:
                print(f"[재개] 기존 결과 {len(rows)}건 발견 → 나머지만 진행합니다.")
        except Exception as e:
            print(f"[경고] 기존 결과 로드 실패 ({e}) → 처음부터 시작합니다.")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedder      = OpenAIEmbedder(openai_client)
    rag_v1        = PgVectorRAG(embedder)
    rag_v2        = PgVectorRAG_V2(embedder)

    total = len(testset)

    for idx, item in enumerate(testset, 1):
        if idx in rows:
            print(f"[{idx:02d}/{total}] 이미 완료 — 스킵")
            continue

        q   = item["question"]
        ref = item["reference"]
        print(f"[{idx:02d}/{total}] {q[:40]}...")

        # System 0: No RAG
        try:
            ans0 = generate_answer_no_rag(openai_client, q)
        except Exception as e:
            ans0 = f"ERROR: {e}"
        time.sleep(EVAL_DELAY)

        # System 1: V1 RAG (site_id=4)
        try:
            chunks_v1 = rag_v1.retrieve(query=q, site_id=SITE_ID_V1, k=TOP_K)
            ctxs_v1   = [c.content for c in chunks_v1]
            ans1      = generate_answer_with_context(openai_client, q, ctxs_v1)
        except Exception as e:
            ctxs_v1, ans1 = [], f"ERROR: {e}"
        time.sleep(EVAL_DELAY)

        # System 2: V2 RAG LlamaParser+Hybrid (site_id=3)
        try:
            chunks_v2 = rag_v2.hybrid_search(query=q, site_id=SITE_ID_V2, k=TOP_K)
            ctxs_v2   = [
                f"[{c.section_title}]\n{c.content}"
                if c.section_title else c.content
                for c in chunks_v2
            ]
            ans2      = generate_answer_with_context(openai_client, q, ctxs_v2)
        except Exception as e:
            ctxs_v2, ans2 = [], f"ERROR: {e}"
        time.sleep(EVAL_DELAY)

        rows[idx] = {
            "q_id":       idx,
            "question":   q,
            "reference":  ref,
            "ans_no_rag": ans0,
            "ans_v1":     ans1,
            "ans_v2":     ans2,
            "ctxs_v1":    json.dumps(ctxs_v1,  ensure_ascii=False),
            "ctxs_v2":    json.dumps(ctxs_v2,  ensure_ascii=False),
        }
        # 전체 DataFrame을 덮어쓰기 → 중복·헤더 오염 원천 차단
        df_partial = pd.DataFrame(
            [rows[k] for k in sorted(rows)]
        )
        df_partial.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"  → 저장 완료 ({idx}/{total})")

    df = pd.DataFrame([rows[k] for k in sorted(rows)])
    print(f"\n✓ 원시 결과 저장: {raw_path} (총 {len(df)}건)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. RAGAS 평가
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_PHRASES = ("관련 정보를 찾을 수 없습니다", "검색 결과 없음", "참고 정보 없음")

def _patch_korean_prompt(ar_metric) -> None:
    """
    RAGAS AnswerRelevancy 내부 역질문 생성 프롬프트에 한국어 지시를 추가한다.
    RAGAS 버전별로 속성명이 다를 수 있어 여러 경로를 시도한다.
    """
    korean_prefix = (
        "You MUST generate all questions in Korean (한국어). "
        "절대 영어로 질문을 작성하지 마세요. "
        "반드시 한국어로만 질문을 작성하세요. "
    )
    patched = False
    # RAGAS 버전별 속성명을 모두 시도 (0.4.x → question_generation, 구버전 → 기타)
    for attr in ("question_generation", "question_generation_prompt",
                 "_question_generation_prompt", "answer_relevancy_prompt",
                 "generate_questions_prompt"):
        prompt_obj = getattr(ar_metric, attr, None)
        if prompt_obj is None:
            continue
        # StringPrompt / PromptTemplate / PydanticPrompt 계열 (instruction 속성)
        if hasattr(prompt_obj, "instruction"):
            try:
                prompt_obj.instruction = korean_prefix + (prompt_obj.instruction or "")
                patched = True
                break
            except Exception:
                pass
        if hasattr(prompt_obj, "text"):
            try:
                prompt_obj.text = korean_prefix + (prompt_obj.text or "")
                patched = True
                break
            except Exception:
                pass
    if not patched:
        print("[WARN] AnswerRelevancy 프롬프트 패치 실패 — 영어 역질문으로 평가될 수 있습니다.")
        print("[WARN] RAGAS 버전 확인: pip show ragas")
    else:
        print("[INFO] AnswerRelevancy 한국어 프롬프트 패치 완료")


def run_ragas(df: pd.DataFrame, system: str) -> Dict[str, float]:
    """
    system: "no_rag" | "v1" | "v2"

    RAGAS 지표:
      - Faithfulness      : 답변이 검색된 청크에 근거하는가  (No RAG 제외)
      - Answer Relevancy  : 답변이 질문과 관련 있는가        (전 시스템)
      - Context Recall    : 정답 내용을 청크가 포함하는가    (No RAG 제외)
      - Context Precision : 검색된 청크가 얼마나 관련 있는가 (No RAG 제외)

    AnswerRelevancy 개선 포인트:
      1. 내부 역질문 생성 프롬프트를 한국어로 패치 (영어 역질문 → 낮은 유사도 방지)
      2. text-embedding-3-large 로 업그레이드 (한국어 의미 포착 향상)
      3. strictness=3로 샘플 수 증가 (스코어 안정성 향상)
      4. 폴백 답변("관련 정보 없음")은 AnswerRelevancy 계산에서 제외
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings

    api_key = os.getenv("OPENAI_API_KEY")
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from ragas.run_config import RunConfig

    # RunConfig: 동시 처리 수 제한 + 충분한 타임아웃 + 재시도로 TimeoutError 근본 차단
    run_config = RunConfig(
        timeout=180,        # 샘플 1개당 최대 대기 180초
        max_retries=10,     # 실패 시 최대 10회 재시도
        max_wait=120,       # 재시도 간 최대 대기 120초 (지수 백오프)
        max_workers=6,      # 동시 API 호출 수 6로 제한 (핵심 설정)
        seed=42,
    )

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        timeout=180,     # run_config.timeout과 일치
        max_retries=5,
    ))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key,
            timeout=120,
            max_retries=5,
        )
    )

    ans_col  = {"no_rag": "ans_no_rag", "v1": "ans_v1", "v2": "ans_v2"}[system]
    ctx_col  = {"no_rag": None,         "v1": "ctxs_v1", "v2": "ctxs_v2"}[system]

    samples = []
    skipped_fallback = 0
    for _, row in df.iterrows():
        response = row[ans_col]
        if ctx_col:
            contexts = json.loads(row[ctx_col]) or ["검색 결과 없음"]
        else:
            contexts = ["참고 정보 없음"]

        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=response,
                retrieved_contexts=contexts,
                reference=row["reference"],
            )
        )

    # AnswerRelevancy 전용: 폴백 답변 제외 샘플 별도 구성
    ar_samples = []
    for s in samples:
        if any(phrase in (s.response or "") for phrase in _FALLBACK_PHRASES):
            skipped_fallback += 1
        else:
            ar_samples.append(s)
    if skipped_fallback:
        print(f"[INFO] AnswerRelevancy: 폴백 답변 {skipped_fallback}건 제외 "
              f"({len(ar_samples)}/{len(samples)}건 평가)")

    dataset    = EvaluationDataset(samples=samples)
    ar_dataset = EvaluationDataset(samples=ar_samples) if ar_samples else dataset

    # strictness=3: 역질문 3개 생성 (1=기본값, 5=안정적이나 API 부하 큼)
    ar_metric = AnswerRelevancy(llm=llm, embeddings=emb, strictness=3)
    _patch_korean_prompt(ar_metric)

    scores: Dict[str, float] = {}

    # AnswerRelevancy 단독 평가 (폴백 제외 샘플)
    print(f"[RAGAS] {system} — AnswerRelevancy 평가 중...")
    ar_result = evaluate(
        dataset=ar_dataset,
        metrics=[ar_metric],
        run_config=run_config,
        raise_exceptions=False,
    )
    ar_scores = ar_result.to_pandas().select_dtypes(include="number").mean().to_dict()
    scores.update(ar_scores)

    if system != "no_rag":
        # 나머지 지표는 전체 샘플로 평가
        print(f"[RAGAS] {system} — Faithfulness / ContextRecall / ContextPrecision 평가 중...")
        other_metrics = [
            Faithfulness(llm=llm),
            ContextRecall(llm=llm),
            ContextPrecision(llm=llm),
        ]
        other_result = evaluate(
            dataset=dataset,
            metrics=other_metrics,
            run_config=run_config,
            raise_exceptions=False,
        )
        other_scores = other_result.to_pandas().select_dtypes(include="number").mean().to_dict()
        scores.update(other_scores)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. 요약 및 저장
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    systems = {
        "No RAG (Plain GPT)":           "no_rag",
        "V1 RAG (Chunking+Overlap)":    "v1",
        "V2 RAG (LlamaParser+Summary)": "v2",
    }

    summary_rows = []
    for display_name, sys_key in systems.items():
        print(f"\n{'='*50}")
        print(f"  {display_name}")
        print(f"{'='*50}")

        scores = run_ragas(df, sys_key)

        col_map = {"no_rag": "correct_no_rag", "v1": "correct_v1", "v2": "correct_v2"}
        correct_col = col_map[sys_key]
        if correct_col in df.columns:
            correct_ratio = round(df[correct_col].mean(), 4)
        else:
            correct_ratio = "N/A"

        row = {"System": display_name, "correctness": correct_ratio}
        for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            val = scores.get(metric)
            row[metric] = round(float(val), 4) if val is not None else "N/A"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def save_results(df_raw: pd.DataFrame, df_summary: pd.DataFrame):
    df_raw.to_csv(RESULTS_DIR / "rag_eval_raw.csv", index=False, encoding="utf-8-sig")

    xlsx_path = RESULTS_DIR / "rag_eval_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_raw.to_excel(writer, sheet_name="Raw",     index=False)
    print(f"\n✓ 요약 결과 저장: {xlsx_path}")

    print("\n" + "="*70)
    print("  논문용 성능 비교 요약")
    print("="*70)
    col_order = ["System", "correctness", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    print(df_summary[col_order].to_string(index=False))
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("경복궁 RAG 평가 시작")
    print(f"테스트셋: {TESTSET_PATH}")
    print(f"결과 저장: {RESULTS_DIR}\n")

    testset = parse_testset(TESTSET_PATH)
    print(f"파싱된 문항 수: {len(testset)}")

    # run_all_systems 내부에서 이어받기 처리 (끊겨도 완료된 행은 보존)
    # 처음부터 다시 하려면 results/rag_eval_raw.csv 삭제 후 재시작
    df_raw = run_all_systems(testset)

    df_raw     = run_correctness_eval(df_raw)
    df_summary = build_summary(df_raw)
    save_results(df_raw, df_summary)


if __name__ == "__main__":
    main()
