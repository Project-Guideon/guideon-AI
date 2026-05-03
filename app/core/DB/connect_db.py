import os
import psycopg
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("get_conn: missing required env var: DATABASE_URL")

    parsed = urlparse(database_url)
    connect_timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))

    return psycopg.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        dbname=parsed.path.lstrip("/"),
        user=parsed.username,
        password=parsed.password,
        connect_timeout=connect_timeout,
    )