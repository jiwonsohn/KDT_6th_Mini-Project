import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

import pandas as pd

# 크롬 드라이버 경로 설정 (자신의 경로로 변경해야 함)
chrome_driver_path = "path_to_chromedriver"  # 크롬 드라이버 설치 경로 입력
service = Service(chrome_driver_path)

# 웹 드라이버 실행
driver = webdriver.Chrome()

# 영화 리뷰 url 리스트 로드
movieDF = pd.read_csv('./movie_list.csv')


print()
url_list = movieDF['url']

for idx, url in enumerate(url_list):

    # Watcha 영화 리뷰 페이지로 이동
    driver.get(url)
    # url = 'https://pedia.watcha.com/ko-KR/contents/tEZAvmQ/comments'

    # 페이지 로딩 대기
    time.sleep(3)

    # 스크롤을 내려서 추가 리뷰를 로딩
    for i in range(300):  # 원하는 만큼 스크롤 (숫자를 늘리면 더 많은 리뷰를 가져옴)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # 리뷰 평점과 내용 크롤링
    reviews = driver.find_elements(By.CLASS_NAME, 'NrA8QHzP')  # 리뷰가 들어 있는 클래스 확인
    ratings = driver.find_elements(By.CLASS_NAME, 'aytsxOVO')  # 평점이 들어 있는 클래스 확인

    review_tot = []
    rating_tot = []

    # 리뷰와 평점을 출력
    for i in range(min(len(reviews), len(ratings))):
        review_text = reviews[i].text
        rating_score = ratings[i].text

        # 데이터 추가
        review_tot.append(review_text)
        rating_tot.append(rating_score)

        # 확인
        # print("=" * 50)
        # print("=" * 50)
        # print(f"평점: {rating_score}")
        # print("-"*50)
        # print(f"리뷰: {review_text}")
        # print("="*50)

        # # 로그 추가: 각 크롤링 데이터 확인
        # print(f"평점 {i+1}: {rating_score}")
        # print(f"리뷰 {i+1}: {review_text}")
        # print("=" * 50)

    # 저장
    DF_NAME = movieDF['title'][idx]
    tmpDF = pd.DataFrame({"review":review_tot,'rating':rating_tot})
    tmpDF.to_csv(f'./{DF_NAME}_review.csv', index=False, encoding='utf-8-sig')
    del tmpDF

# 드라이버 종료
driver.quit()