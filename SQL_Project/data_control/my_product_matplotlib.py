
'''
생필품 도시별 카테고리별 품목별 판매가격 DF

연도 2019 - 2023
'''

import pymysql
import pandas as pd
import csv

conn = pymysql.connect(host = '172.20.97.217', user='member1', password='1234', db='product', charset='utf8')

# DF의 칼럼들을 같이 리턴
cur = conn.cursor(pymysql.cursors.DictCursor)

query = """ 
select ppp.type_id as '상품_코드', ppp.product_type as '상품_타입', ppp.category_name as '소분류', 
ppp.product_name as '제품명' , price.product_price as '가격', price.year as '년도'
from product_price_tb price
	inner join
	(select ttt.type_id, ttt.product_type, ttt.category_name, name.product_name , name.product_id
	from product_name_tb as name
	inner join 
		(select ptype.type_id, ptype.product_type, categ.category_name, categ.category_id 
			from product_type_tb as ptype
			inner join product_category_tb as categ
			on ptype.type_id = categ.type_id
			where ptype.type_id = 'C') as ttt
			on name.category_id = ttt.category_id) as ppp
	on price.product_id = ppp.product_id
order by price.year asc;
"""

# 쿼리 실행
cur.execute(query)

# 모든 데이터 가져오기
rows = cur.fetchall()

rawDF = pd.DataFrame(rows)






