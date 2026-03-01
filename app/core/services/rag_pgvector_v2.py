"""
구조화 RAG v2 — Hybrid Search 검색 엔진

벡터 검색(요약 임베딩) + BM25(tsvector) → RRF 점수 병합.
기존 PgVectorRAG와 동일한 인터페이스(retrieve 메서드)를 제공하여 교체 가능.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.DB.connect_db import get_conn
from app.core.services.rag_pgvector import OpenAIEmbedder, RetrievedChunk


@dataclass
class RetrievedChunkV2(RetrievedChunk):
    """v2 청크: 요약/키워드/섹션 제목 추가."""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    section_title: str = ""
    rrf_score: float = 0.0


class PgVectorRAG_V2:
    """Hybrid Search (벡터 + BM25) 기반 RAG 검색 엔진."""

    def __init__(self, embedder: OpenAIEmbedder):
        self.embedder = embedder

    # ── 벡터 검색 (요약 임베딩 기반) ────────────────────────────────────────
    # 질문을 임베딩으로 변환 → DB의 요약 임베딩과 cosine 유사도 비교
    # 의미적으로 유사한 청크를 찾음 (예: "입장료" ≈ "관람 요금")

    def _vector_search(
        self, query_emb: List[float], site_id: int, k: int = 10
    ) -> List[Dict[str, Any]]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, summary, keywords,
                           section_title, metadata, embedding,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM tb_doc_chunk_v2
                    WHERE site_id = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_emb, site_id, query_emb, k),
                )
                rows = cur.fetchall()

        results = []
        for row in rows:
            chunk_id, doc_id, content, summary, keywords, section_title, meta, embedding, sim = row
            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": content,
                "summary": summary,
                "keywords": keywords or [],
                "section_title": section_title or "",
                "metadata": meta if isinstance(meta, dict) else {},
                "embedding": embedding,
                "similarity": float(sim),
            })
        return results

    # ── BM25 검색 (tsvector 기반) ──────────────────────────────────────────
    # 질문의 키워드가 원문/요약/제목에 직접 매칭되는지 검색
    # 고유명사, 숫자 등 정확한 키워드 매칭에 강함 (예: "3,000원", "경복궁")

    def _bm25_search(
        self, query: str, site_id: int, k: int = 10
    ) -> List[Dict[str, Any]]:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, summary, keywords,
                           section_title, metadata,
                           ts_rank_cd(search_tsv, plainto_tsquery('simple', %s)) AS bm25_score
                    FROM tb_doc_chunk_v2
                    WHERE site_id = %s
                      AND search_tsv @@ plainto_tsquery('simple', %s)
                    ORDER BY bm25_score DESC
                    LIMIT %s
                    """,
                    (query, site_id, query, k),
                )
                rows = cur.fetchall()

        results = []
        for row in rows:
            chunk_id, doc_id, content, summary, keywords, section_title, meta, score = row
            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": content,
                "summary": summary,
                "keywords": keywords or [],
                "section_title": section_title or "",
                "metadata": meta if isinstance(meta, dict) else {},
                "bm25_score": float(score),
            })
        return results

    # ── RRF(Reciprocal Rank Fusion) 병합 ──────────────────────────────────
    # 벡터 검색 순위와 BM25 검색 순위를 합쳐서 최종 점수 산출
    # 두 검색이 서로 보완 → 한쪽이 놓치는 결과를 다른 쪽이 보완

    def hybrid_search(
        self,
        query: str,
        site_id: int,
        k: int = 5,
        fetch_k: int = 10,
        vector_weight: float = 0.6,
        rrf_constant: int = 60,
    ) -> List[RetrievedChunkV2]:
        """RRF score = vector_weight/(rank_vec+C) + (1-vector_weight)/(rank_bm25+C)"""
        # 쿼리 임베딩
        query_emb = self.embedder.embed(query)

        # 두 검색 실행
        vec_results = self._vector_search(query_emb, site_id, k=fetch_k)
        bm25_results = self._bm25_search(query, site_id, k=fetch_k)

        # RRF 점수 계산
        rrf_scores: Dict[int, float] = defaultdict(float)
        chunk_data: Dict[int, Dict[str, Any]] = {}

        for rank, item in enumerate(vec_results):
            cid = item["chunk_id"]
            rrf_scores[cid] += vector_weight / (rank + 1 + rrf_constant)
            chunk_data[cid] = item

        bm25_weight = 1.0 - vector_weight
        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rrf_scores[cid] += bm25_weight / (rank + 1 + rrf_constant)
            if cid not in chunk_data:
                chunk_data[cid] = item

        # RRF 점수 기준 정렬 → Top-k
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]

        results: List[RetrievedChunkV2] = []
        for cid in sorted_ids:
            data = chunk_data[cid]
            results.append(
                RetrievedChunkV2(
                    chunk_id=cid,
                    doc_id=data["doc_id"],
                    content=data["content"],          # LLM에 넘길 원문
                    metadata=data["metadata"],
                    similarity=data.get("similarity", 0.0),
                    summary=data["summary"],
                    keywords=data["keywords"],
                    section_title=data["section_title"],
                    rrf_score=round(rrf_scores[cid], 6),
                )
            )

        return results

    def hybrid_search_with_embeddings(
        self,
        query: str,
        site_id: int,
        k: int = 10,
        vector_weight: float = 0.6,
        rrf_constant: int = 60,
    ):
        """MMR용 — hybrid search(벡터+BM25) 후 임베딩을 함께 반환.

        Returns:
            (q_emb, chunks, chunk_embs)
            - q_emb      : np.ndarray  쿼리 임베딩
            - chunks     : List[RetrievedChunkV2]
            - chunk_embs : List[np.ndarray]  각 청크의 임베딩 (MMR 다양성 계산용)
        """
        import json
        import numpy as np

        q_emb_list = self.embedder.embed(query)
        q_emb = np.array(q_emb_list, dtype=np.float32)

        # 두 검색 실행 (벡터 검색은 embedding 포함)
        vec_results = self._vector_search(q_emb_list, site_id, k=k)
        bm25_results = self._bm25_search(query, site_id, k=k)

        # RRF 점수 계산
        rrf_scores: Dict[int, float] = defaultdict(float)
        chunk_data: Dict[int, Dict[str, Any]] = {}

        for rank, item in enumerate(vec_results):
            cid = item["chunk_id"]
            rrf_scores[cid] += vector_weight / (rank + 1 + rrf_constant)
            chunk_data[cid] = item

        bm25_weight = 1.0 - vector_weight
        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rrf_scores[cid] += bm25_weight / (rank + 1 + rrf_constant)
            if cid not in chunk_data:
                chunk_data[cid] = item

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        chunks: List[RetrievedChunkV2] = []
        chunk_embs: List[np.ndarray] = []

        for cid in sorted_ids:
            data = chunk_data[cid]
            chunks.append(
                RetrievedChunkV2(
                    chunk_id=cid,
                    doc_id=data["doc_id"],
                    content=data["content"],
                    metadata=data["metadata"],
                    similarity=data.get("similarity", 0.0),
                    summary=data.get("summary") or "",
                    keywords=data.get("keywords") or [],
                    section_title=data.get("section_title") or "",
                    rrf_score=round(rrf_scores[cid], 6),
                )
            )
            emb = data.get("embedding")
            if emb is not None:
                if isinstance(emb, str):
                    emb = json.loads(emb)
                chunk_embs.append(np.array(emb, dtype=np.float32))
            else:
                # BM25에서만 나온 청크는 embedding 없음 → zero vector
                chunk_embs.append(np.zeros_like(q_emb))

        return q_emb, chunks, chunk_embs

    # ── 기존 PgVectorRAG 인터페이스 호환 ──────────────────────────────────

    def retrieve(
        self, query: str, site_id: int = 1, k: int = 5
    ) -> List[RetrievedChunk]:
        """PgVectorRAG.retrieve()와 동일한 시그니처. 내부에서 hybrid_search 사용."""
        return self.hybrid_search(query=query, site_id=site_id, k=k)

    def retrieve_with_embeddings(
        self, query: str, site_id: int = 1, k: int = 15
    ):
        """MMR 호환용 — 벡터 검색만 수행하고 임베딩을 함께 반환."""
        import json
        import numpy as np

        q_emb_list = self.embedder.embed(query)
        q_emb = np.array(q_emb_list, dtype=np.float32)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, metadata,
                           embedding,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM tb_doc_chunk_v2
                    WHERE site_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_emb.tolist(), site_id, q_emb.tolist(), k),
                )
                rows = cur.fetchall()

        chunks: List[RetrievedChunk] = []
        chunk_embs = []

        for chunk_id, doc_id, content, meta, embedding, similarity in rows:
            if meta is None:
                meta = {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=content,
                    metadata=meta if isinstance(meta, dict) else {"raw": meta},
                    similarity=round(float(similarity), 4),
                )
            )
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            chunk_embs.append(np.array(embedding, dtype=np.float32))

        return q_emb, chunks, chunk_embs
