# 위에 라인 : 셀 내용을 파일로 생성/ 한번 생성후에는 마스킹

# 모듈 로딩--------------------------------------------
import os.path     # 파일 및 폴더 관련
import cgi, cgitb  # cgi 프로그래밍 관련
import joblib      # AI 모델 관련
import sys, codecs # 인코딩 관련
from pydoc import html # html 코드 관련 : html을 객체로 처리?
import torch
from custom_utils import *


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
          <title>---패혈증 생존 예측---</title>
         </head>
         <body>
          <form>
            <input type='text' name='age' placeholder='나이 입력', value={data[0]}><br/>
            <input type='text' name='gender' placeholder='성별 입력', value={data[1]}><br/>
            <input type='text' name='count' placeholder='발병횟수 입력', value={data[2]}><br/>

            <p><input type="submit" value="생존 예측"><Br/>[ {msg} ]</p>
          </form>
         </body>
        </html>""")

    
# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : detectLang
# 재 료 : 사용자 입력 데이터
# 결 과 : 판별 언어명(영어, 프랑스~)


def detectLang(data_input, model):
	
    # data_input = ['25','남자','2'] [나이, 성별, 발병 횟수]

    result=""
    gender_dict = {"남자":0,"남":0,"남성":0, "여자":1,"여":1,"여성":1}

    if data_input == [""*3]: result="입력 대기" 

    elif len(data_input) !=3: 
		
        result ="나이, 성별, 발병 횟수 순으로 다시 입력하세요."

    # input data 성별 전처리
    elif data_input[1] not in list(gender_dict.keys()): 
        result="입력대기"
    
    elif data_input != [""*3]:

        data_input[1] = gender_dict[data_input[1]]
        # # print(data_input)

        data_input = list(map(int, data_input))

        # 텐서화
        dataTS = torch.FloatTensor(data_input).reshape(1,-1)
        # print(dataTS, dataTS.shape)

        # test data 예측-----------------------------------------
        # 모델 로드
        # pklfile = r'C:\Users\KDP-43\Desktop\DL_project\cgi-bin\model\alive_1\model_all.pth'

        # SepsisModel = joblib.load(pklfile)

        model.eval()
        with torch.no_grad():
            pre_test = model(dataTS)

        # result ="생존" if (pre_test > 0.5).item() else "사망"
        result ="사망" if (pre_test > 0.8).item() else "생존"

        data_input

    return result

# 기능 구현 ------------------------------------------------
# (1) WEB 인코딩 설정
if SCRIPT_MODE:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach()) #웹에서만 필요 : 표준출력을 utf-8로

# # (2) 모델 로딩

# pklfile = r'C:\Users\KDP-43\Desktop\DL_project\cgi-bin\model\alive_1\model_all.pth'
    
# SepsisModel = joblib.load(pklfile)

## 저장경로
SAVE_PATH= r'C:\Users\KDP-43\Desktop\DL_web\models\Adam\alive_1'

# 저장 파일명
SAVE_MODEL = SAVE_PATH+'\model_all.pth'

if os.path.exists(SAVE_MODEL):
	SepsisLODModel= torch.load(SAVE_MODEL, weights_only=False)
	print("경로상 파일이 존재합니다.")
else:
	print(f'{SAVE_MODEL} 파일이 존재하지 않습니다. 다시 확인하세요.')



# (3) WEB 사용자 입력 데이터 처리
# (3-1) HTML 코드에서 사용자 입력 받는 form 태크 영역 객체 가져오기
form = cgi.FieldStorage()
# (3-2) Form안에 textarea 태크 속 데이터 가져오기
data = []
data.append(form.getvalue("age", default=""))
data.append(form.getvalue("gender", default=""))
data.append(form.getvalue("count", default=""))
#text ="Happy New Year" # 테스트용 (쥬피터 내부)

# (3-3) 판별하기
# msg =""
# if not '' in data:
result_input = detectLang(data, SepsisLODModel)
result_input = f'{result_input}'

# result_input = detectLang(data)

# data 초기화
data = [""]*3
# (4) 사용자에게 WEB 화면 제공
showHTML(data,result_input)
