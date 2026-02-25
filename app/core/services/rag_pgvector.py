# services/rag_pgvector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

# from sentence_transformers import SentenceTransformer
from app.core.DB.connect_db import get_conn


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    content: str
    metadata: Dict[str, Any]
    similarity: float = 0.0          # 1 - cosine distance (높을수록 유사)

class OpenAIEmbedder:
    def __init__(self, client, model="text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding  # length = 1536
    
class PgVectorRAG:
    # PDF2db2.py와 동일한 모델 필수 (저장 벡터 공간 일치)
    def __init__(self, embedder: OpenAIEmbedder):
        self.embedder = embedder

    def retrieve(self, query: str, site_id: int = 1, k: int = 5) -> List[RetrievedChunk]:
        q_emb = self.embedder.embed(query)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM tb_doc_chunk
                    WHERE site_id = %s
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                    """,
                    #order by 오름차순으로 하면 맨 앞에 제일 작은것이 나옴 =>작을수록 가장 유사한것
                    (q_emb, site_id, q_emb, k),
                )
                #output: chunk_id, doc_id, content, metadata, similarity
                rows = cur.fetchall()

        chunks: List[RetrievedChunk] = []
        for chunk_id, doc_id, content, meta, similarity in rows:
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

        return chunks

    def retrieve_with_embeddings(
        self, query: str, site_id: int = 1, k: int = 15
    ):
        """MMR 용 — 청크와 함께 DB 저장 임베딩을 반환 (청크 재인코딩 없음).

        Returns:
            query_emb  : np.ndarray        — 쿼리 임베딩 (MMR 계산용)
            chunks     : List[RetrievedChunk]
            chunk_embs : List[np.ndarray]  — 각 청크의 저장 임베딩
        """
        import json
        import numpy as np

        # np.array()로 명시적 변환 → pyright 타입 오류 방지
        ''' q_emb: np.ndarray = np.array(
            self.model.encode(query, normalize_embeddings=True), dtype=np.float32
        )'''
        q_emb_list = self.embedder.embed(query)     # List[float] length 1536
        q_emb = np.array(q_emb_list, dtype=np.float32)

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, metadata,
                           embedding,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM tb_doc_chunk
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
            # pgvector 드라이버에 따라 string 으로 반환될 수 있으므로 파싱
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            chunk_embs.append(np.array(embedding, dtype=np.float32))

        return q_emb, chunks, chunk_embs
