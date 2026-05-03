import os
import psycopg
from urllib.parse import urlparse, parse_qsl
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("get_conn: missing required env var: DATABASE_URL")

    parsed = urlparse(database_url)

    if not parsed.scheme.startswith(("postgresql", "postgres")):
        raise ValueError("DATABASE_URL must start with postgresql:// or postgres://")

    dbname = parsed.path.lstrip("/")
    if not dbname:
        raise ValueError("DATABASE_URL must include database name")

    connect_timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))

    params = dict(parse_qsl(parsed.query))
    params.setdefault("connect_timeout", connect_timeout)

    return psycopg.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        dbname=dbname,
        user=parsed.username,
        password=parsed.password,
        **params,
    )