import os
import psycopg
from urllib.parse import urlparse, parse_qsl
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    database_url = os.getenv("DATABASE_URL")
    connect_timeout = int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "5"))

    if not database_url:
        required = ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise ValueError(f"get_conn: missing required env vars: {', '.join(missing)}")

        return psycopg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            connect_timeout=connect_timeout,
        )

    parsed = urlparse(database_url)

    if not parsed.scheme.startswith(("postgresql", "postgres")):
        raise ValueError("DATABASE_URL must start with postgresql:// or postgres://")

    dbname = parsed.path.lstrip("/")
    if not dbname:
        raise ValueError("DATABASE_URL must include database name")

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