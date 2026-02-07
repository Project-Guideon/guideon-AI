# services/rag_pgvector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from app.core.DB.connect_db import get_conn


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    content: str
    metadata: Dict[str, Any]


class PgVectorRAG:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def retrieve(self, query: str, site_id: int = 1, k: int = 5) -> List[RetrievedChunk]:
        q_emb = self.model.encode(query).tolist()

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, metadata
                    FROM tb_doc_chunk
                    WHERE site_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (site_id, q_emb, k),
                )
                rows: List[Tuple[int, int, str, Any]] = cur.fetchall()

        chunks: List[RetrievedChunk] = []
        for chunk_id, doc_id, content, meta in rows:
            # meta가 JSONB면 파이썬 dict로 오거나 문자열로 올 수 있어서 안전 처리
            if meta is None:
                meta = {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=content,
                    metadata=meta if isinstance(meta, dict) else {"raw": meta},
                )
            )

        return chunks
