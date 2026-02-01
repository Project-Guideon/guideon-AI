from app.core.DB.connect_db import get_conn

SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 검색 속도(있으면 좋음)
CREATE INDEX IF NOT EXISTS idx_documents_embedding
ON documents USING hnsw (embedding vector_cosine_ops);
"""

with get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute(SQL)
    conn.commit()

print("✅ documents table ready")
