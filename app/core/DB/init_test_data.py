#!/usr/bin/env python3
"""
초기화 스크립트: 테스트용 기본 데이터 삽입
테스트 단계에서 PDF를 로드하기 위해 필요한 tb_site 데이터를 생성합니다.
"""

import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

def init_test_data():
    """테스트용 기본 데이터 초기화"""
    
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", 5432))
    dbname = os.getenv("POSTGRES_DB", "guideon_db")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD")
    
    print(f"🔌 연결 중: {host}:{port}/{dbname} (user={user})")
    
    conn = psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )
    
    print("✅ DB 연결 성공")
    
    cur = conn.cursor()
    
    try:
        # 1. 테스트용 사이트 삽입 (site_id=1)
        print("📝 tb_site에 데이터 삽입 중...")
        cur.execute("""
            INSERT INTO tb_site (name, is_active, created_at, updated_at)
            VALUES ('경복궁', TRUE, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """)
        
        print(f"   삽입된 행: {cur.rowcount}")
        
        conn.commit()
        print("✅ 커밋 완료")
        
        # 검증: 실제로 저장되었는지 확인
        cur.execute("SELECT COUNT(*) FROM tb_site WHERE name='경복궁';")
        count = cur.fetchone()[0]
        print(f"✅ 검증: tb_site 데이터 {count}개 존재")
        
        if count > 0:
            print("✅ 테스트 데이터 초기화 완료!")
            print("   - tb_site: site_id=1 (경복궁)")
        else:
            print("⚠️  데이터가 저장되지 않았습니다. ON CONFLICT 때문일 수 있습니다.")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ 초기화 오류: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    init_test_data()
