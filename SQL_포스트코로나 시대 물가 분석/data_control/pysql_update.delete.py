

'''
update, delete
'''

import pymysql

conn = pymysql.connect(host = 'localhost', user='root', password='1234', 
                          db='sqlclass_db', charset='utf8')

curs = conn.cursor()

sql = """ 
    update customer
    set region = '서울특별시'
    where region='서울'
    """

curs.execute(sql)
print('update 완료')

# 이름이 홍길동인 데이터 삭제
sql = "delete from customer where name=%s"
curs.execute(sql, '홍길동')
print('delete 홍길동')

# 실제 DB & 테이블 반영
conn.commit()

curs.close()
conn.close()







