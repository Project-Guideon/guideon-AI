from __future__ import annotations

from typing import List

import numpy as np

from app.core.services.rag_pgvector import PgVectorRAG
from app.graph.state import GraphState


# ── MMR 구현 ──────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def _mmr_select(
    query_emb: np.ndarray,
    doc_embs: List[np.ndarray],
    chunks: list,
    top_k: int,
    lambda_param: float = 0.5,
) -> list:
    """MMR 알고리즘으로 관련성 + 다양성을 고려해 top_k 청크 선택.

    lambda_param:
        1.0 → 순수 관련성 (일반 검색과 동일)
        0.0 → 순수 다양성
        0.5 → 균형 (기본값)
    """
    selected_idx: list = []
    remaining_idx: list = list(range(len(chunks)))

    while len(selected_idx) < top_k and remaining_idx:
        if not selected_idx:
            # 첫 번째 선택: 쿼리와 가장 유사한 청크
            scores = [_cosine_sim(query_emb, doc_embs[i]) for i in remaining_idx]
            best_pos = int(np.argmax(scores))
        else:
            mmr_scores = []
            for i in remaining_idx:
                relevance = _cosine_sim(query_emb, doc_embs[i])
                # 이미 선택된 청크들과의 최대 유사도 (중복도)
                redundancy = max(
                    _cosine_sim(doc_embs[i], doc_embs[j]) for j in selected_idx
                )
                mmr_scores.append(
                    lambda_param * relevance - (1 - lambda_param) * redundancy
                )
            best_pos = int(np.argmax(mmr_scores))

        best_idx = remaining_idx[best_pos]
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [chunks[i] for i in selected_idx]


# ── 팩토리 ────────────────────────────────────────────────────────────────────

def make_retrieve_node(rag: PgVectorRAG):
    """RAG 검색 노드 팩토리.

    - retry_count == 0 : 일반 similarity 검색 (top_k)
    - retry_count  > 0 : top_k * 2 후보 + DB 저장 임베딩 → MMR 선택 (재인코딩 없음)
    """

    def retrieve_node(state: GraphState) -> dict:
        query: str = state.get("retrieval_query_ko") or state.get("normalized_text", "")
        site_id: int = state.get("site_id", 1)
        top_k: int = state.get("top_k", 5)
        retry_count: int = state.get("retry_count", 0)

        use_mmr: bool = retry_count > 0

        if not use_mmr:
            # ── 첫 번째 시도: 일반 검색 ──────────────────────────────────
            raw_chunks = rag.retrieve(query=query, site_id=site_id, k=top_k)
            chunks_dict = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.content,
                    "metadata": c.metadata,
                    "similarity": c.similarity,
                }
                for c in raw_chunks
            ]

        else:
            # ── 재시도: DB 저장 임베딩으로 MMR (청크 재인코딩 없음) ────────
            fetch_k = top_k * 2
            q_emb, raw_chunks, chunk_embs = rag.retrieve_with_embeddings(
                query=query, site_id=site_id, k=fetch_k
            )

            candidates = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.content,
                    "metadata": c.metadata,
                    "similarity": c.similarity,
                }
                for c in raw_chunks
            ]

            if len(candidates) > top_k:
                chunks_dict = _mmr_select(q_emb, chunk_embs, candidates, top_k)
            else:
                chunks_dict = candidates

        max_sim = max((c["similarity"] for c in chunks_dict), default=0.0)

        trace = dict(state.get("trace") or {})
        trace["retrieve"] = {
            "query": query,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "chunks_count": len(chunks_dict),
            "max_similarity": round(max_sim, 4),
        }

        return {"retrieved_chunks": chunks_dict, "trace": trace}

    return retrieve_node
