
import pymysql
import pandas as pd
import csv

conn = pymysql.connect(host = '172.20.97.217', user='member1', password='1234', db='product', charset='utf8')

# DF의 칼럼들을 같이 리턴
cur = conn.cursor()

query = """
select c.email
from customer as c
    inner join rental as r
    on c.customer_id = r.customer_id
where date(r.rental_date) = (%s)""" 

# 쿼리 실행
cur.execute(query,('2005-06-14'))

# 모든 데이터 가져오기
rows = cur.fetchall()

# 튜플 형태 출력
for row in rows:
    print(row)


cur.close()
conn.close()