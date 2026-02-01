# db 연결 코드
# .env 파일 생성 후 사용
# 항상 실행
# GUIDEON\guideon-AI> python -m app.core.DB.connect_db
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()  # .env 자동 로드

def get_conn():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )