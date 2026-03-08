# db 연결 코드
# .env 파일 생성 후 사용
# 항상 실행
# GUIDEON\guideon-AI> python -m app.core.DB.connect_db
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()  # .env 자동 로드

def get_conn():
    required = [
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
        "POSTGRES_USER", "POSTGRES_PASSWORD",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(
            f"get_conn: missing required env vars: {', '.join(missing)}"
        )

    port_str = os.getenv("POSTGRES_PORT")
    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(
            f"get_conn: POSTGRES_PORT must be an integer, got {port_str!r}"
        )

    connect_timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))

    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=port,
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=connect_timeout,
    )