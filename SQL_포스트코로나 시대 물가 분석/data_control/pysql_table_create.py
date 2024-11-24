
import pymysql


def create_table(conn,cur):
    try:
        # 기존 테이블 삭제
        query1 = "drop table if exists customer"

        # 테이블 새로 생성
        query2 = """ 
            create table customer
            (name varchar(10),
            category smallint,
            region varchar(10))
            """
        
        cur.execute(query1)
        cur.execute(query2)

        # 실행결과 확정
        conn.commit()
        print('Table 생성 완료')

    # 예외 처리
    # Exception --> 최상위권 클래스 처리문 할당
    except Exception as e:
        print(e)

    
def main():
    conn = pymysql.connect(host = 'localhost', user='root', password='1234', 
                          db='sqlclass_db', charset='utf8')
    
    cur = conn.cursor()

    # 테이블 생성 함수 호출
    create_table(conn,cur)

    # 연결 종료
    cur.close()
    conn.close()

    print('Database 연결 종료')

main()