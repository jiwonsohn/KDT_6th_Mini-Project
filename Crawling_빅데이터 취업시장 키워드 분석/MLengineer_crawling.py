

# 잡코리아 - 채용 - 직무별 - 개발/데이터 - 머신러닝엔지니어
# 채용공고 페이지 url 크롤링

from urllib.request import urlopen
from urllib.request import Request
from bs4 import BeautifulSoup

import collections
import pandas as pd
from tabulate import tabulate

import re
import random
import time

# Callable 에러 해결
# AttributeError: module 'collections' has no attribute 'Callable'
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

def clean_string(s):
    # 줄바꿈(\n, \r) 제거
    s = s.replace('\n', ' ').replace('\r', ' ')
    # 두 칸 이상의 공백을 하나의 공백으로 변경
    s = re.sub(r'\s{2,}', ' ', s)
    # 문자열 양쪽의 공백 제거
    return s.strip()


# 기업별 채용 공고 페이지 url---------------------------------

# url 저장 리스트
company_list = []
page_url_list = []

# 총 33페이지
base_url = 'https://www.jobkorea.co.kr/Search/?stext=%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4&duty=1000242&tabType=recruit&Page_No='

for page_num in range(1, 33+1):

    url = f'https://www.jobkorea.co.kr/Search/?stext=%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4&duty=1000242&tabType=recruit&Page_No={page_num}'

    urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(urlrequest)

    soup = BeautifulSoup(html.read().decode('utf-8'), 'html.parser')

    articles_list = soup.find('article', {'class':'list'}).find_all('article',{'class':'list-item'})
    # print(articles_list)
    


    # 1페이지 당 20개
    for idx in range(len(articles_list)):

        name = articles_list[idx].find('a').get_text().strip()
        page_url = 'https://www.jobkorea.co.kr' + articles_list[idx].find('a')['href']

        page_url_list.append(page_url)
        company_list.append( [name, page_url] )

print(len(page_url_list))
        

# 공고 페이지별 데이터 뽑기 in data_dict

data_dict = {}
columns=['경력','학력','스킬','핵심역량','우대','기본우대','자격증']

total = []

for page in page_url_list:

    urlrequest2 = Request(page, headers={'User-Agent': 'Mozilla/5.0'})
    
    time.sleep(random.randint(3,5))
    html2 = urlopen(urlrequest)

    soup2 = BeautifulSoup(html.read().decode('utf-8'), 'html.parser')

    tb_list = soup.find('div',{'class':'tbCol'}).find('dl',{'class':'tbList'})
    company_name = soup.find('div',{'class':'colInfo'}).find('h4').text()

    total.append( [company_name, tb_list])


    # dt_list = tb_list.find_all('dt')
    # dd_list = tb_list.find_all('dd')

    # data_list = ['']*7

    # for i in range(len(dd_list)):
    #     idx = columns.index(dt_list[i].text.strip())

    #     data_list[idx] = clean_string(dd_list[i].text)

    # dd_list = list(map(lambda x:clean_string(x.text), dd_list))
    # data_dict[company_name] = data_list
    
comp_total_info = pd.DataFrame(total, columns=('기업명','info'))

print(tabulate(comp_total_info.head(5), tablefmt='pretty'))


        
# [공고 기업이름, 공고채용 페이지 url] --> DF 저장

# file_name = 'MLengineer_list.csv'

# tmp = pd.DataFrame(company_list, columns=('기업명','공고링크'))
                                                                                   
# tmp.to_csv(file_name, index=False, encoding='utf-8-sig')






