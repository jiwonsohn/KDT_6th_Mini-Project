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
from flask_utils import *
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
    return(f"""
    
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>---SNS 대화 주제 예측---</title>
		   
    <!-- 외부 css 파일 적용 -->
    <link type="text/css" rel="stylesheet" href="{{url_for('static', filename='css/main.css')}}">

</head>
<body>
    <header>무슨 얘기해?ㅁ?</header>

    <form>
        
        <label>입력 예시</label><br/>
        <pre>
            난 이미 짐승인 거 자각 하고 있고 날 키워줄 동물이 필요해
        </pre>
        <textarea name="msg" placeholder="대화를 입력하세요" rows="10" cols="50"></textarea><br/>
        <p><input type="submit" value="무슨 얘기중이야?"><br/>{msg}</p>
    </form>


</body>
</html>""")

    
# 사용자 입력 텍스트 판별하는 함수---------------------------------------------------------------------------
# 함수명 : topicPredict
# 재 료 : 사용자 입력 데이터
# 결 과 : 긍정/부정, 확률

def topicPredict(mymodel, texts, STOP_FILE, VOCAB_FILE, max_length):

	# 불용어 로드
	kor_stopwords = clean_korStopwords(STOP_FILE)

	kor_stopwords.append('키키')
	kor_stopwords.append('는')
	kor_stopwords.append('은')

	# 텍스트 데이터 정규화 처리
	cleaned = cleantext(texts)

	# 단어사전 로드
	with open(VOCAB_FILE, 'rb') as f:
		vocab = pickle.load(f)

	print("단어 사전이 불러와졌습니다.")
	# print(vocab)

	# 토크나이저 인스턴스 생성
	tokenizer = Okt()

	# 토큰화
	tokenList = [ token for token in tokenizer.morphs(cleaned, norm=False, stem=False) if token not in kor_stopwords]
	# print(tokenList)

	for idx, token in enumerate(tokenList):
		if token not in vocab.keys():
			tokenList[idx] = '<unk>'
        
	# print(tokenList)

	# 사전을 사용해 토큰을 정수 인덱스로 변환
	indices = [vocab[token] for token in tokenList]

	# 최대 길이를 맞추기 위해 패딩 처리
	if len(indices) < max_length:
		indices += [vocab['<pad>']] * (max_length - len(indices))
	else:
		indices = indices[:max_length]

	# 파이토치 텐서로 변환
	input_tensor = torch.tensor(indices).unsqueeze(0)  

	# 예측
	mymodel.eval()

	with torch.no_grad():
		
		# 예측
		pre_y = mymodel(input_tensor)
		probability = torch.sigmoid(pre_y).item()
		
		# 결과 출력 (임계값을 0.5로 설정해 긍정/부정 분류)
		if probability > 0.5:
			result =  "긍정", probability
		else:
			result =  "부정", probability
			
	print(result)


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
data = form.getvalue("review", default="")
print(data)

#text ="Happy New Year" # 테스트용 (쥬피터 내부)

# (3-3) 판별하기

STOP_FILE = './kor_stopwordsVer2.txt'
VOCAB_FILE = './pet_vocab.pkl'
max_length = 20

result = movieReviewPosNeg(mymodel, data, STOP_FILE, VOCAB_FILE, max_length)
result_input = f'{result}'


# data 초기화
# data = [""]*3
data = ""

# (4) 사용자에게 WEB 화면 제공
showHTML(data,result_input)