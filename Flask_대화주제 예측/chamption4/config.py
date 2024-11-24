

import os

# SQLite3 RDBMS 파일 경로 관련
BASE_DIR = os.path.dirname(__file__)
DB_NAME = 'web_project'                  # SQLite -> .db 확장자 이름!!

DB_MYSQL_URI = f'mysql+pymysql://jiwon:1234@172.20.60.151:3306/{DB_NAME}'

# DB 관련 기능 구현 시 사용할 전역변수
SQLALCHEMY_DATABASE_URI = DB_MYSQL_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False


