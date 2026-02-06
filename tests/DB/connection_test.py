import os
from dotenv import load_dotenv

load_dotenv()

def test_db():
    from sqlalchemy import create_engine, text

    db_url = os.getenv("DATABASE_URL")
    assert db_url, "DATABASE_URL is missing in .env"

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        val = conn.execute(text("SELECT 1")).scalar()
    print("[DB] OK:", val)

    return engine

def test_redis():
    import redis

    redis_url = os.getenv("REDIS_URL")
    assert redis_url, "REDIS_URL is missing in .env"

    r = redis.from_url(redis_url, decode_responses=True)
    pong = r.ping()
    print("[Redis] OK:", pong)


# --- 추가: DB 내부 조회 함수들 ---
from sqlalchemy import text

def show_tables(engine):
    """DB의 public 스키마 내 테이블 목록 출력"""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        tables = [row[0] for row in result]
    print("[DB] Tables:", tables)

def show_table_data(engine, table_name, limit=5):
    """특정 테이블의 데이터 일부 출력"""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
        print(f"[DB] Data from {table_name} (최대 {limit}개):")
        for row in result:
            print(row)

def show_table_columns(engine, table_name):
    """특정 테이블의 컬럼명과 타입 출력"""
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
        """))
        print(f"[DB] Columns in {table_name}:")
        for row in result:
            print(row)

if __name__ == "__main__":
    print("DATABASE_URL =", os.getenv("DATABASE_URL"))
    print("REDIS_URL    =", os.getenv("REDIS_URL"))
    engine = test_db()
    test_redis()
    print("✅ All connections OK")

    # --- 추가: DB 내부 조회 예시 ---
    show_tables(engine)
    # 아래 두 줄은 실제 테이블명이 있을 때만 사용하세요 (예시: 'users')
    # show_table_data(engine, 'users')
    # show_table_columns(engine, 'users')