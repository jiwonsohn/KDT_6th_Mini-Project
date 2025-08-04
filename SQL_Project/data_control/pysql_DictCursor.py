
import pymysql
import pandas as pd
import csv

conn = pymysql.connect(host = '172.20.97.217', user='member1', password='1234', db='product', charset='utf8')

# DF의 칼럼들을 같이 리턴
cur = conn.cursor(pymysql.cursors.DictCursor)

# 쿼리 실행
cur.execute('select * from language')

# 모든 데이터 가져오기
rows = cur.fetchall()


language_df = pd.DataFrame(rows)
print(language_df)

print()

print(language_df['name'])

cur.close()
conn.close()