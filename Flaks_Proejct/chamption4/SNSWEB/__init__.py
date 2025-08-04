
# -------------------------------------------------------------
# Flask Framework에서 WebServer 구동 파일
#  - 파일명: __init__.py
# -------------------------------------------------------------

# 모듈 로딩
from flask import Flask, render_template, request

# 모델 로드 관련 모듈 로딩
import os
import torch


# 토큰화 & 형태소 분석 모듈 ----------------------------------
from konlpy.tag import Okt
from collections import Counter

# 파이토치 모듈 ----------------------------------------------
import torch
import torch.nn as nn
from torch import optim


# 데이터 전처리 & 로드 모듈 -----------------------------------
import re
import os
import pandas as pd
import numpy as np

import pickle

# 기타--------------------------------------------------------
# from flask_utils import *

#---------------------------------------------------------------
# WEB
#---------------------------------------------------------------

# def load_model(MODEL_FILE):
# 	if os.path.exists(MODEL_FILE):
# 		model = torch.load(MODEL_FILE, weights_only=False)
# 		return model

# 	else:
# 		result = '파일이 없엉. 다시 봐봐'
# 		return result


# def movieReviewPosNeg(mymodel, texts, STOP_FILE, VOCAB_FILE, max_length):
      
# 	if texts == "": return "입력값이 없습니다."

# 	# 불용어 로드
# 	kor_stopwords = clean_korStopwords(STOP_FILE)
      
# 	kor_stopwords.append('키키')
# 	kor_stopwords.append('은')
# 	kor_stopwords.append('는')

# 	# 텍스트 데이터 정규화 처리
# 	cleaned = cleantext(texts)

# 	# 단어사전 로드
# 	with open(VOCAB_FILE, 'rb') as f:
# 		vocab = pickle.load(f)

# 	print("단어 사전이 불러와졌습니다.")
# 	# print(vocab)

# 	# 토크나이저 인스턴스 생성
# 	tokenizer = Okt()

# 	# 토큰화
# 	tokenList = [ token for token in tokenizer.morphs(cleaned, norm=False, stem=False) if token not in kor_stopwords]
# 	# print(tokenList)

# 	for idx, token in enumerate(tokenList):
# 		if token not in vocab.keys():
# 			tokenList[idx] = '<unk>'
        
# 	# print(tokenList)

# 	# 사전을 사용해 토큰을 정수 인덱스로 변환
# 	indices = [vocab[token] for token in tokenList]

# 	# 최대 길이를 맞추기 위해 패딩 처리
# 	if len(indices) < max_length:
# 		indices += [vocab['<pad>']] * (max_length - len(indices))
# 	else:
# 		indices = indices[:max_length]

# 	# 파이토치 텐서로 변환
# 	input_tensor = torch.tensor(indices).unsqueeze(0)  

# 	# 예측
# 	mymodel.eval()

# 	with torch.no_grad():
		
# 		# 예측
# 		pre_y = mymodel(input_tensor)
# 		probability = torch.sigmoid(pre_y).item()
		
# 		# 결과 출력 (임계값을 0.5로 설정해 긍정/부정 분류)
# 		if probability > 0.5:
# 			result =  "반려동물", probability
# 		else:
# 			result =  "아니다이눔아", probability
			
# 	return result


#---------------------------------------------------------------
# MODEL Train 
#---------------------------------------------------------------

# 텍스트 전처리 함수
def cleantext(text):

	# # \n\n을 공백으로 대체
	# text = re.sub(r"\n\n", " ", text) 
    
	# 한글 공백 제외 삭제
	text = re.sub(r"[^가-힣\s]", "", str(text))

	return text.strip()


# 단어 사전 생성 함수
def build_vocab(corpus, n_vocab, special_tokens): 
    counter = Counter()
    
    for tokens in corpus:
        counter.update(tokens)
        
    vocab = special_tokens			# <pad>, <unk>
    for token, count in counter.most_common(n_vocab): 
        vocab.append(token)
        
    return vocab


def clean_korStopwords(STOP_FILE):
    with open(STOP_FILE, 'rt', encoding='utf-8') as f:
        stopwords = f.read().split('\n')
        
	# 은혁님 한국어 불용어 모음 추가함으로써 생긴 중복 제거
    return list(set(stopwords))


## 정수 인코딩 & 패딩 ----------------------------------------------

# 패딩 함수
def pad_sequences(sequences, max_length, pad_value):
    
	result = list()

	for seq in sequences:
		seq = seq[:max_length]
		pad_length = max_length - len(seq)

		padded_sequnce = seq + [pad_value]*pad_length

		result.append(padded_sequnce)

	return np.asarray(result)


# 모델 로드
model_PATH = r'C:\Users\KDP-43\Desktop\Flask_Project\models\model_num_loss(0.3344)_score(0.8260).pth'
# model = load_model(model_PATH)
# print(model)

STOP_FILE = './kor_stopwordsVer2.txt'
VOCAB_FILE = './pet_vocab.pkl'
max_length = 20

# -------------------------------------------------------------
# Application 생성 함수
#  - 함수명: create_app
# -------------------------------------------------------------


def create_app():

    # flask web server 인스턴스 생성
    APP = Flask(__name__)


    # URL 처리 모듈
    from .views import main_view
    APP.register_blueprint(main_view.mainBP)


    return APP

    

# 조건부 실행                                                              
if __name__ == '__main__':

    app = create_app()
    app.run()





