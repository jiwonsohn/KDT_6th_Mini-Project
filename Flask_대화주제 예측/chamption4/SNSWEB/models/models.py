
# --------------------------------------------------------
# 데이터베이스의 테이블 정의 클래스
# --------------------------------------------------------

# 모듈로딩
from SNSWEB import DB

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer

# orm과 매핑 선언
Base = declarative_base()


# --------------------------------------------------------
# Pet 테이블 정의 클래스
#   -texts: 대화내용
#   -subjects: 대화주제 (1:반려동물, 0:군대)
# --------------------------------------------------------

class Pet(Base):

    __tablename__ = 'Pet'

    id = Column(Integer, primary_key=True)
    subjects = Column(String(10))
    texts = Column(String(10000))




