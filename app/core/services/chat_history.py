import os
import json
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CHAT_HISTORY_MAX_TURNS = 10  # 최근 10턴 (user+assistant 각 1개씩 = 20개 항목)

_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _client


def load_chat_history(session_id: str) -> list[dict]:
    """
    Redis에서 최근 대화 내역 조회
    반환: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    실패 시 빈 리스트 반환 (LLM 호출은 계속 진행)
    """
    try:
        r = _get_redis()
        key = f"chat:{session_id}"
        items = r.lrange(key, -(CHAT_HISTORY_MAX_TURNS * 2), -1)
        history = [json.loads(item) for item in items]

        print(f"[ChatHistory] 조회: session_id={session_id}, key={key}, count={len(history)}")
        for i, msg in enumerate(history):
            role = msg.get("role", "?")
            content = msg.get("content", "")[:50]  # 50자까지만 출력
            print(f"  [{i}] {role}: {content}{'...' if len(msg.get('content', '')) > 50 else ''}")

        return history
    except Exception as e:
        print(f"[ChatHistory] 조회 실패: session_id={session_id}, {type(e).__name__}: {e}")
        return []
