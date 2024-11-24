
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


def load_model(MODEL_FILE):
	if os.path.exists(MODEL_FILE):
		model = torch.load(MODEL_FILE, weights_only=False)
		return model

	else:
		result = '파일이 없엉. 다시 봐봐'
		return result


def movieReviewPosNeg(model, texts, STOP_FILE, VOCAB_FILE, max_length):

	if texts == "": return "입력값이 없습니다."

	# 불용어 로드
	kor_stopwords = clean_korStopwords(STOP_FILE)

	kor_stopwords.append('.')
	kor_stopwords.append('\n')
	kor_stopwords.append('\n ')
	kor_stopwords.append('영화')
	kor_stopwords.append('은')
	kor_stopwords.append('는')
	kor_stopwords.append('유아인')
	kor_stopwords.append('류승완')
	kor_stopwords.append('황정민')
	kor_stopwords.append('이순신')

	# 텍스트 데이터 정규화 처리
	cleaned = cleantext(texts)
	print(cleaned)

	# 단어사전 로드
	with open(VOCAB_FILE, 'rb') as f:
		vocab = pickle.load(f)

	# print("단어 사전이 불러와졌습니다.")
	# print(vocab)

	# 토크나이저 인스턴스 생성
	tokenizer = Okt()

	# 토큰화
	tokenList = [ token for token in tokenizer.morphs(cleaned, norm=False, stem=False) if token not in kor_stopwords]
	print(tokenList)

	for idx, token in enumerate(tokenList):
		if token not in vocab.keys():
			tokenList[idx] = '<unk>'
        
	print(tokenList)

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

	return result
			
	

# text = "열등의식과 무력감이 빚어낸 기묘한 리듬감.  봉준호 감독 스타일의 정점!"
text = "믿고보는 범죄도시... 깔깔 웃고 즐거운 2시간이였다. 2번째 보고난후 또 봐도 재밌네 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 인생은 혼자살수없나봄..."

MODEL_FILE = r'C:\Users\KDP-43\Desktop\NLTK_project\models\model_num_loss(0.7953)_score(0.6651).pth'
STOP_FILE = './kor_stopwordsVer2.txt'
VOCAB_FILE = './vocab.pkl'
max_length = 35

mymodel = load_model(MODEL_FILE)

result = movieReviewPosNeg(mymodel, text, STOP_FILE, VOCAB_FILE, max_length)

print(result)