
# --------------------------------------------------
# Flask Framework에서 모듈단위 URL 처리 파일
#   - 파일명: main_view.py
# --------------------------------------------------

# 모듈 로딩
from flask import Blueprint, render_template, request, url_for
from werkzeug.utils import redirect

import torch
# from flask_utils import *
# from DBWEB.models.models import Question

# Blueprint 인스턴스 생성
mainBP = Blueprint('MAIN',
                   import_name=__name__,
                   url_prefix='/',
                   template_folder='templates')

# model = torch.load('SNSWEB/models/model_num_loss(0.3344)_score(0.8260).pth', weights_only=False)

# http://localhost:5000/URL 처리 라우팅 함수 정의
@mainBP.route("/", methods=["POST","GET"])
def index():
    if request.method == "POST":
        text = request.form['msg']
        # predict = movieReviewPosNeg(text)
        prediction = "보여줄래 제발.."
        return redirect(url_for('MAIN.Result', prediction = prediction))
        # return render_template('MAIN.Result', predict = predict)

    return render_template('tmp2.html')



# http://localhost:5000/URL 처리 라우팅 함수 정의
@mainBP.route("/predict", methods=["POST","GET"])
def Result():

    # if request.method == "POST":

    prediction= request.args.get('prediction', '보여줘...') 
    return render_template('tmp2_copy.html', prediction=prediction)



