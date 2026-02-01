from sentence_transformers import SentenceTransformer
from app.core.DB.connect_db import get_conn

MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dim
model = SentenceTransformer(MODEL_NAME)

def search(query: str, k: int = 5):
    q_emb = model.encode(query).tolist()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q_emb, k),
            )
            return cur.fetchall()

if __name__ == "__main__":
    q = "경복궁 바깥을 두른 담장의 길이는 몇 m야?"
    rows = search(q, k=5)

    print("=== QUERY ===")
    print(q)
    print("\n=== TOP CHUNKS ===")
    for rid, content, meta in rows:
        print(f"\n--- id={rid} meta={meta} ---")
        print(content[:400])
