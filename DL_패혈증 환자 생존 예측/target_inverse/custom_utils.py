
# ------------------------------------
'''
sepsis 생사 예측 모델에 필요한 함수 & 클래스 정리
'''
# ------------------------------------

# ----------------------------------------
# 모듈 로딩
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinarySpecificity
from torchmetrics.classification import BinaryConfusionMatrix
from torchinfo import summary
import torch.optim.lr_scheduler as lr_scheduler


# Data 로딩 & 시각화 모듈 로딩
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split



# 모듈 버전 확인
def ver_check():
    print(f'torch v.{torch.__version__}')
    print(f'pandas v.{pd.__version__}')

# ----------------------------------------
# 함수: 피쳐 & 타겟 DF 크기, 차원 확인
# ----------------------------------------

def print_shape_dim(tmp1DF, tmp2DF):

    print(f'tmp1DF=> {tmp1DF.shape} {tmp1DF.ndim}D')
    print(f'tmp2DF=> {tmp2DF.shape} {tmp2DF.ndim}D')


# -------------------------------------------------------
#모델 클래스 설계 & 정의
'''
* 클래스이름: SepsisLODModel
* 클래스목적: sepsis 피쳐 바탕으로 환자의 9일 후 생사여부 예측
* 부모클래스: nn.Module
* 매개_변수: in_in, out_out, h_in=[], h_out=[]
* 기능_역할:
	- init(): 모델 구조 설정

* 클래스 구조:<br>
	- 은닉층 개수=> 동적
	- 입력층: 입력: 3		출력: 동적	AF: ReLU<br>
	- 은닉층: 입력: 동적	출력: 동적	AF: ReLU<br>
	- 출력층: 입력: 동적	출력: 1		AF: Sigmoid<br>
'''
# ------------------------------------------------------


class SepsisLODModel(nn.Module):
    
	'''
	def __init__(self, in_out, h_in=[], h_out=[]):

		super().__init__()
		self.in_layer = nn.Linear(3, h_in[0,0] if len(h_in) else in_out)
		self.hd_layers = nn.ModuleList()
		
		for idx in range(len(h_in)):
			self.hd_layers.append(nn.Linear(h_in[idx], h_out[idx]))

		self.ot_layer = nn.Linear(h_out[-1] if len(h_in) else in_out, 1)
	'''
	
	def __init__(self,in_out,perceptrons = []) :
		super().__init__()
		self.in_layer = nn.Linear(3,perceptrons[0] if len(perceptrons) else in_out)

		self.hd_layers = nn.ModuleList()
		for idx in range(len(perceptrons)-1) :
			self.hd_layers.append(nn.Linear(perceptrons[idx], perceptrons[idx+1]))

		self.ot_layer = nn.Linear(perceptrons[-1] if len(perceptrons) else in_out,1)

	def forward(self, input_data):
		x = F.relu(self.in_layer(input_data))

		for linear in self.hd_layers:
			x = F.relu(linear(x))

		return F.sigmoid(self.ot_layer(x))
	

class SepsisLODModel2(nn.Module):
    
	'''
	def __init__(self, in_out, h_in=[], h_out=[]):

		super().__init__()
		self.in_layer = nn.Linear(3, h_in[0,0] if len(h_in) else in_out)
		self.hd_layers = nn.ModuleList()
		
		for idx in range(len(h_in)):
			self.hd_layers.append(nn.Linear(h_in[idx], h_out[idx]))

		self.ot_layer = nn.Linear(h_out[-1] if len(h_in) else in_out, 1)
	'''
	
	def __init__(self,in_out,perceptrons = []) :
		super().__init__()
		self.in_layer = nn.Linear(3,perceptrons[0] if len(perceptrons) else in_out)

		self.hd_layers = nn.ModuleList()
		for idx in range(len(perceptrons)-1) :
			self.hd_layers.append(nn.Linear(perceptrons[idx], perceptrons[idx+1]))

		self.ot_layer = nn.Linear(perceptrons[-1] if len(perceptrons) else in_out,1)

	def forward(self, input_data):
		x = F.relu(self.in_layer(input_data))

		for linear in self.hd_layers:
			x = F.relu(linear(x))

		# return F.sigmoid(self.ot_layer(x))
		return self.ot_layer(x)



# -------------------------------------------------------
# 데이터셋 클래스 설계 & 정의
# -------------------------------------------------------

'''
* 클래스이름: SepsisDataset
* 클래스목적: featureDF, targetDF 텐서화
* 부모클래스: nn.Module
* 매개_변수: featureDF, targetDF
* 기능_역할:
	- init(): 데이터 구조, 속성 선언
    - len(): 데이터 길이
	- getitem(): 데이터 텐서화
	
'''
# ------------------------------------------------------


class SepsisDataset(Dataset):
    
	def __init__(self, featureDF, targetDF):
		self.featureDF = featureDF
		self.targetDF=targetDF
		self.n_rows=featureDF.shape[0]
		self.n_features=featureDF.shape[1]
		
	def __len__(self):
		return self.n_rows
	
	def __getitem__(self, idx):
		# 텐서화
		featureTS = torch.FloatTensor(self.featureDF.iloc[idx].values)
		targetTS = torch.FloatTensor(self.targetDF.iloc[idx].values)

		return featureTS, targetTS


# -------------------------------------------------------
# train, test, vaild 데이터셋 shape, 차원 & 타겟 클래스 비율 출력
# -------------------------------------------------------

def dataset_print(X_train,y_train,X_test,y_test,X_val,y_val):

    print(f'X_train: {X_train.shape} {X_train.ndim}D')
    print(f'y_train: {y_train.shape} {y_train.ndim}D')
    print(f'y_train :\n {y_train.value_counts()}')
    print(f'y_train :\n {y_train.value_counts()*(100/len(y_train))}')
    print()
    print(f'X_val: {X_val.shape} {X_val.ndim}D')
    print(f'y_val: {y_val.shape} {y_val.ndim}D')
    print(f'y_val :\n {y_val.value_counts()}')
    print(f'y_val :\n {y_val.value_counts()*(100/len(y_val))}')
    print()
    print(f'X_test: {X_test.shape} {X_test.ndim}D')
    print(f'y_test: {y_test.shape} {y_test.ndim}D')
    print(f'y_test :\n {y_test.value_counts()}')
    print(f'y_test :\n {y_test.value_counts()*(100/len(y_test))}')
    

# -------------------------------------------------------
# train,  vaild DS 모델 LOSS, Recall, F1-score plot
# -------------------------------------------------------


def showLossScore(LOSS_history,Recall_history, F1score_history):
    fig, axes = plt.subplots(1,3, figsize=(15,6))
    TH = len(F1score_history['Train'])
    # print(TH)
	
    axes[0].plot(range(1, TH+1), LOSS_history['Train'][:TH],label="Train")
    axes[0].plot(range(1, TH+1), LOSS_history['Val'][:TH],label="Valid")
    axes[0].grid()
    # axes[0].legend()
    axes[0].set_xticks(range(1, TH+1)[::2], range(1, TH+1)[::2])
    axes[0].set_xlabel("TH")
    axes[0].set_ylabel("LOSS")
    axes[0].set_title("LOSS")

    axes[1].plot(range(1, TH+1), F1score_history['Train'][:TH],label="Train")
    axes[1].plot(range(1, TH+1), F1score_history['Val'][:TH],label="Valid")
    axes[1].grid()
    # axes[1].legend()
    axes[1].set_xticks(range(1, TH+1)[::2], range(1, TH+1)[::2])
    axes[1].set_xlabel("TH")
    axes[1].set_ylabel("f1-score")
    axes[1].set_title("f1-score")

    axes[2].plot(range(1, TH+1), Recall_history['Train'][:TH],label="Train")
    axes[2].plot(range(1, TH+1), Recall_history['Val'][:TH],label="Valid")
    axes[2].grid()
    axes[2].legend()
    axes[2].set_xticks(range(1, TH+1)[::2], range(1, TH+1)[::2])
    axes[2].set_xlabel("TH")
    axes[2].set_ylabel("Recall")
    axes[2].set_title("Recall")
	
def showLossScore2(LOSS_history,Recall_history,F1score_history, specfi_history):
    fig, axes = plt.subplots(2,2, figsize=(14,10))

# fig.suptitle(f"overSamppling ratio:{resamp_ratio} n_neigh: {n_neighbor}")

    TH = len(F1score_history['Train'])

    axes[0,0].plot(range(1, TH+1), LOSS_history['Train'][:TH],label="Train")
    axes[0,0].plot(range(1, TH+1), LOSS_history['Val'][:TH],label="Valid")
    axes[0,0].grid()
    axes[0,0].legend()
    axes[0,0].set_xlabel("TH")
    axes[0,0].set_ylabel("LOSS")
    axes[0,0].set_title("LOSS")

    axes[0,1].plot(range(1, TH+1), F1score_history['Train'][:TH],label="Train")
    axes[0,1].plot(range(1, TH+1), F1score_history['Val'][:TH],label="Valid")
    axes[0,1].grid()
    axes[0,1].legend()
    axes[0,1].set_xlabel("TH")
    axes[0,1].set_ylabel("f1-score")
    axes[0,1].set_title("f1-score")

    axes[1,0].plot(range(1, TH+1), Recall_history['Train'][:TH],label="Train")
    axes[1,0].plot(range(1, TH+1), Recall_history['Val'][:TH],label="Valid")
    axes[1,0].grid()
    axes[1,0].legend()
    axes[1,0].set_xlabel("TH")
    axes[1,0].set_ylabel("Recall")
    axes[1,0].set_title("Recall")

    axes[1,1].plot(range(1, TH+1), specfi_history['Train'][:TH],label="Train")
    axes[1,1].plot(range(1, TH+1), specfi_history['Val'][:TH],label="Valid")
    axes[1,1].grid()
    axes[1,1].legend()
    axes[1,1].set_xlabel("TH")
    axes[1,1].set_ylabel("Specificity")
    axes[1,1].set_title("Specificity")

# def showLossScore(LOSS_history,Recall_history,F1score_history, specfi_history, resamp_ratio,n_neighbor):
#     fig, axes = plt.subplots(2,2, figsize=(14,5))

#     fig.suptitle(f"overSamppling ratio:{resamp_ratio} n_neigh: {n_neighbor}")

#     TH = len(F1score_history['Train'])

#     axes[0,0].plot(range(1, TH+1), LOSS_history['Train'][:TH],label="Train")
#     axes[0,0].plot(range(1, TH+1), LOSS_history['Val'][:TH],label="Valid")
#     axes[0,0].grid()
#     # axes[0,0].legend()
#     axes[0,0].set_xlabel("TH")
#     axes[0,0].set_ylabel("LOSS")
#     axes[0,0].set_title("LOSS")

#     axes[0,1].plot(range(1, TH+1), F1score_history['Train'][:TH],label="Train")
#     axes[0,1].plot(range(1, TH+1), F1score_history['Val'][:TH],label="Valid")
#     axes[0,1].grid()
#     # axes[0,1].legend()
#     axes[0,1].set_xlabel("TH")
#     axes[0,1].set_ylabel("f1-score")
#     axes[0,1].set_title("f1-score")

#     axes[1,0].plot(range(1, TH+1), Recall_history['Train'][:TH],label="Train")
#     axes[1,0].plot(range(1, TH+1), Recall_history['Val'][:TH],label="Valid")
#     axes[1,0].grid()
#     # axes[1,0].legend()
#     axes[1,0].set_xlabel("TH")
#     axes[1,0].set_ylabel("Recall")
#     axes[1,0].set_title("Recall")
	
#     axes[1,1].plot(range(1, TH+1), specfi_history['Train'][:TH],label="Train")
#     axes[1,1].plot(range(1, TH+1), specfi_history['Val'][:TH],label="Valid")
#     axes[1,1].grid()
#     axes[1,1].legend()
#     axes[1,1].set_xlabel("TH")
#     axes[1,1].set_ylabel("Specificity")
#     axes[1,1].set_title("Specificity")