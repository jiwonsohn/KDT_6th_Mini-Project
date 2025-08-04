# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import sys, codecs # 인코딩 관련
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import torch
# from custom_utils import *

# 모듈 로딩
# - Model 실행 관련
import torch              
import torch.nn as nn              
import torch.nn.functional as F 
# from torchinfo import summary 

# - 그 외
from RNN_utils import *
import os
import pickle


# 동작관련 전역 변수----------------------------------
SCRIPT_MODE = True    # Jupyter Mode : False, WEB Mode : True
cgitb.enable()         # Web상에서 진행상태 메시지를 콘솔에서 확인할수 있도록 하는 기능


# 사용자 정의 함수-----------------------------------------------------------
# WEB에서 사용자에게 보여주고 입력받는 함수 ---------------------------------
# 함수명 : showHTML
# 재 료 : 사용자 입력 데이터, 판별 결과
# 결 과 : 사용자에게 보여질 HTML 코드

def showHTML(data, msg):
    print("Content-Type: text/html; charset=utf-8")
    print(f"""
    
        <!DOCTYPE html>
        <html lang="en">
         <head>
          <meta charset="UTF-8">
          <title>---영화리뷰 감정 예측---</title>
         </head>
         <body>
          <form>
            <input type='text' name='review' placeholder='리뷰 내용 입력', value={data}><br/>
            
            <p><input type="submit" value="가보자고~~"><Br/>[ {msg} ]</p>
          </form>
         </body>
        </html>""")

    
# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : movieReviewPosNeg
# 재 료 : 사용자 입력 데이터
# 결 과 : 긍정/부정, 확률


# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# # (2) 모델 로딩

MODEL_FILE = r'C:\Users\KDP-43\Desktop\NLTK_project\models\model_num_loss(0.7953)_score(0.6651).pth'
mymodel = load_model(MODEL_FILE)

# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()
# (3-2) Form안에 textarea 태크 속 데이터 가져오기
data = ""
data = form.getvalue("review", default=""))

#text ="Happy New Year" # 테스트용 (쥬피터 내부)

# (3-3) 판별하기

STOP_FILE = './kor_stopwordsVer2.txt'
VOCAB_FILE = './vocab.pkl'
max_length = 35

result = movieReviewPosNeg(mymodel, data, STOP_FILE, VOCAB_FILE, max_length)
result_input = f'{result}'

# result_input = detectLang(data)

# data 초기화
# data = [""]*3
data = ""

# (4) 사용자에게 WEB 화면 제공
showHTML(data,result_input)
