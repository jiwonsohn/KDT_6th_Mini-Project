
import pymysql

conn = pymysql.connect(host = 'localhost', user='root', password='1234', 
                          db='sqlclass_db', charset='utf8')


curs = conn.cursor()


# curs.excute() - 1번에 1개 행 데이터만 insert---------------
# sql = """ insert into customer(name,category, region)
#     values (%s, %s, %s)"""


# curs.execute(sql, ('홍길동',1,'서울'))
# curs.execute(sql, ('이연수',2,'서울'))

# print('INSERT 완료')

# curs.execute('select * from customer')

# # 모든 데이터 가져오기
# rows = curs.fetchall()

# print(rows)

# curs.close()
# conn.close()

# curs.excutemany() - 1번에 여러 개 행 데이터 insert---------------
sql = """ insert into customer(name,category, region)
    values (%s, %s, %s)"""

# 튜플 형태로!!
data = (
    ('홍진우', 1, '서울'),
    ('강지수', 2, '부산'),
    ('김청진', 1, '대구'),
)

# 딕셔너리???
# data = {
#     'name': ('홍진우', '강지수', '김청진'),
#    'category': (1, 2, 1),
#     'region': ('서울', '부산', '대구'),
# }

curs.executemany(sql, data)

conn.commit()

print('exeutemany() 완료')

curs.close()
conn.close()







