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
        _client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=1,   # 연결 타임아웃 1초
            socket_timeout=2,           # 읽기 타임아웃 2초
            health_check_interval=30,   # 30초마다 연결 상태 확인
        )
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

        # role/content 유효성 검증 — 오염 데이터가 GPT 메시지에 섞이는 것 방지
        history = []
        for item in items:
            try:
                msg = json.loads(item)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                history.append({"role": role, "content": content})

        print(f"[ChatHistory] 조회: session_id={session_id}, key={key}, count={len(history)}")
        for i, msg in enumerate(history):
            role = msg.get("role", "?")
            content = msg.get("content", "")[:50]  # 50자까지만 출력
            print(f"  [{i}] {role}: {content}{'...' if len(msg.get('content', '')) > 50 else ''}")

        return history
    except Exception as e:
        print(f"[ChatHistory] 조회 실패: session_id={session_id}, {type(e).__name__}: {e}")
        return []
