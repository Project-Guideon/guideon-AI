from sentence_transformers import SentenceTransformer
from app.core.DB.connect_db import get_conn

MODEL_NAME = "all-mpnet-base-v2"  # 768 dim (PDF2db2와 동일한 모델)
model = SentenceTransformer(MODEL_NAME)

def search(query: str, site_id: int = 1, k: int = 5):
    """
    쿼리를 입력받아 유사한 청크를 검색
    
    Args:
        query: 검색 쿼리 텍스트
        site_id: 관광지 ID (기본값 1)
        k: 반환할 상위 결과 개수
    
    Returns:
        [(chunk_id, content, metadata), ...] 튜플 리스트
    """
    q_emb = model.encode(query).tolist()

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
            return cur.fetchall()

if __name__ == "__main__":
    q = "경복궁 바깥을 두른 담장의 길이는 몇 m야?"
    rows = search(q, site_id=1, k=5)

    print("=== QUERY ===")
    print(q)
    print("\n=== TOP CHUNKS ===")
    if rows:
        for chunk_id, doc_id, content, meta in rows:
            print(f"\n--- chunk_id={chunk_id} doc_id={doc_id} ---")
            print(f"metadata: {meta}")
            print(f"content: {content[:400]}")
    else:
        print("검색 결과가 없습니다.")
