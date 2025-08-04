
# 공고링크별 데이터 반환

# 반환할 데이터
# 7개
# ['경력','학력','스킬','핵심역량','우대','기본우대','자격증']

from urllib.request import urlopen
import requests
# from urllib.request import Request
from bs4 import BeautifulSoup
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By

import random
import re
from tabulate import tabulate
import time

import collections

# Callable 에러 해결
# AttributeError: module 'collections' has no attribute 'Callable'
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

# 채용 공고 csv 로드
rawDF = pd.read_csv('MLengineer_list.csv', encoding='utf-8')
# print(tabulate(rawDF.head(5), tablefmt='pretty'))

# print(rawDF.columns)

print(rawDF['공고링크'][0])
# print(type(rawDF['공고링크'][0]))

                                                                                                                                                                    
# driver 객체 생성
chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome()

tmp = []

# 기업별 채용공고 정보 저장 in list/dict
for idx, link in enumerate(rawDF['공고링크']):

    ## 동적 크롤링-------------------------------------------------
    driver.get(link)
    # 보안코드 방지
    time.sleep(random.randint(3,5))
    company = driver.find_element(By.CLASS_NAME, 'tbList').text
    print('-' * 50)
    print(company)

    tmp.append([idx, rawDF['공고링크'][idx], company])



    # ## 정적 크롤링
    # # user 정보 선언
    # urlrequest = requests.get(link, headers={'User-Agent': 'Mozilla/5.0'}).content
    
    # # html 코드 로드
    # html = urlopen(urlrequest)
                                                
    # # BeautifulSoup 객체 선언
    # soup = BeautifulSoup(html.read().decode('utf-8'), 'html.parser')

    # tot = soup.find_all('div', {'class':'tbCol'})

    # print(tot)    

tttDF = pd.DataFrame(tmp, columns=('인덱스','                링크','정보'))
tttDF.to_csv('info_result.csv', encoding='utf-8', mode='w', index=False)
# tmp.to_csv(file_name, encoding='utf-8', mode='w', index=False)                                                                                                                                         

print(len(company))                     

# 리스트 txt 파일 저장




                                                                                                                                                                                                                                                                
                                                              














    
















