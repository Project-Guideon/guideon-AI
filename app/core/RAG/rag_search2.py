import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from app.core.DB.connect_db import get_conn
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"  # 768 dim (PDF2db2와 동일한 모델)
#"all-mpnet-base-v2"   768 dim (PDF2db2와 동일한 모델)=>실패 =>한국어 지원이 잘안되었던듯
#paraphrase-multilingual-mpnet-base-v2  # 768 dim 다국어 => 성공
#text-embedding-3-small/large
model = SentenceTransformer(MODEL_NAME)

# OpenAI 클라이언트
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = "gpt-4o-mini"  # 빠르고 저렴한 모델

def search_with_scores(query: str, site_id: int = 1, k: int = 5):
    """ 
    쿼리와 유사도 점수를 함께 반환 (디버깅용)
    
    Args:
        query: 검색 쿼리
        site_id: 관광지 ID
        k: 반환할 상위 결과 개수
    
    Returns:
        [(chunk_id, doc_id, content, similarity_score), ...] 
    """
    q_emb = model.encode(query).tolist()

    with get_conn() as conn:
        with conn.cursor() as cur:
            # pgvector의 <=> 연산자는 거리 반환, 1-거리 = 유사도
            cur.execute(
                """
                SELECT chunk_id, doc_id, content, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM tb_doc_chunk
                WHERE site_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q_emb, site_id, q_emb, k),
            )
            return cur.fetchall()


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


def answer(query: str, site_id: int = 1, k: int = 5):
    """
    쿼리에 대한 답변을 생성합니다 (Retrieval + Generation)
    
    Args:
        query: 사용자 질문
        site_id: 관광지 ID
        k: 검색할 상위 결과 개수
    
    Returns:
        {
            'query': 사용자 질문,
            'answer': GPT가 생성한 정답,
            'sources': 참고한 청크 목록
        }
    """
    # 1단계: 관련 청크 검색
    chunks = search(query, site_id=site_id, k=k)
    
    if not chunks:
        return {
            'query': query,
            'answer': '관련 정보를 찾을 수 없습니다.',
            'sources': []
        }
    
    # 2단계: 검색 결과를 컨텍스트로 변환
    context = "\n\n".join([
        f"[청크 {i+1}]\n{content}"
        for i, (_, _, content, _) in enumerate(chunks)
    ])
    
    # 3단계: GPT에 요청
    system_prompt = """당신은  관광 안내 전문가입니다.
사용자의 질문에 대해 제공된 문서 정보(context)를 바탕으로만 정확하고 친절하게 답변하세요.
정보가 없으면 "관련 정보를 찾을 수 없습니다"라고 답하세요."""
    
    user_message = f"""질문: {query}

제공 정보(Context):
{context}

위 정보를 기반으로 질문에 답변해주세요."""
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    answer_text = response.choices[0].message.content
    
    return {
        'query': query,
        'answer': answer_text,
        'sources': [
            {'chunk_id': chunk_id, 'content': content[:200]}
            for chunk_id, _, content, _ in chunks
        ]
    }

if __name__ == "__main__":
    q = "경회루에 대해 설명 해줘"
    
    print("=" * 80)
    print("🔍 RAG 검색 + GPT 답변")
    print("=" * 80)
    
    # 유사도 점수와 함께 상위 10개 결과 보기
    print(f"\n📝 쿼리: {q}")
    print("\n🎯 상위 5개 검색 결과 (유사도 점수 포함):")
    print("─" * 80)
    
    results = search_with_scores(q, site_id=1, k=5)
    
    for i, (chunk_id, doc_id, content, score) in enumerate(results, 1):
        similarity_pct = score * 100
        print(f"\n{i}. chunk_id={chunk_id} | 유사도={similarity_pct:.2f}%")
        print(f"   내용: {content[:100]}...")
    
    
    
    # GPT 답변 생성
    print("\n" + "=" * 80)
    result = answer(q, site_id=1, k=5)
    print(f"\n📌 질문: {result['query']}")
    print(f"\n💬 답변:\n{result['answer']}")
    print(f"\n📚 참고 청크:")
    for source in result['sources']:
        print(f"   - chunk_id={source['chunk_id']}: {source['content'][:80]}...")
    print("=" * 80)
