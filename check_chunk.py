import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="guideon",
    user="postgres",
    password="postgres",
)

cur = conn.cursor()

doc_id = 2

cur.execute("""
    SELECT chunk_index, section_title, summary, keywords, content
    FROM tb_doc_chunk_v2
    WHERE doc_id = %s
    ORDER BY chunk_index
""", (doc_id,))

rows = cur.fetchall()

with open(f"doc_{doc_id}_chunks.txt", "w", encoding="utf-8") as f:
    for chunk_index, section_title, summary, keywords, content in rows:
        f.write("=" * 120 + "\n")
        f.write(f"CHUNK_INDEX   : {chunk_index}\n")
        f.write(f"SECTION_TITLE : {section_title}\n")
        f.write(f"SUMMARY       : {summary}\n")
        f.write(f"KEYWORDS      : {keywords}\n")
        f.write("-" * 120 + "\n")
        f.write((content or "") + "\n")
        f.write("=" * 120 + "\n\n")

cur.close()
conn.close()

print(f"doc_{doc_id}_chunks.txt 저장 완료")