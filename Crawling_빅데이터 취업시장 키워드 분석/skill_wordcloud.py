## 스킬 워드클라우드

# -----------------------------------------------------------
# 영문 wordCloud 생성

# 한글 내용 삭제 - skill_final_text_2.txt
# -----------------------------------------------------------

from wordcloud import STOPWORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

# stopwords = set(STOPWORDS)
# stopwords.add("Python Pytorch")



text = open('skill_final_text_2.txt').read()


# 워드클라우드 출력 형태 이미지 마스크 설정
img_mask = np.array(Image.open('cloud.png'))


# 워드클라우드 분석 ----------------------------------------------------------
wc = WordCloud(width=400, height=400,
               background_color="darkgrey", max_font_size=200,
               stopwords=STOPWORDS,
               repeat=False,
               colormap='seismic', mask=img_mask).generate(text)

# # 딕셔너리 형태로 할당해야만!!
cloud = wc.generate_from_frequencies(dict(tags))

# words_: 객체의 비율 정보가 담긴 딕셔너리 반환
print(wc.words_)

# 워드클라우드 결과 이미지 생성
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(wc)
plt.show()