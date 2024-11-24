

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random

import collections


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

def crawl_list_in_info(data_dict,links):
    columns=['경력','학력','스킬','핵심역량','우대','기본우대','자격증']
    for link in links:
        url = f'https://www.jobkorea.co.kr/{link}'
        # URL에서 HTML 컨텐츠 가져오기
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200:
            print(f"Failed to retrieve contents from {url}")
            return
        time.sleep(random.randint(3,5))
        # BeautifulSoup를 사용하여 HTML 파싱
        soup = BeautifulSoup(response.content, 'html.parser')
        tbList_1 = soup.find_all('div',{'class':'tbCol'})[0].find('dl',{'class':'tbList'}).text.strip()
        tbList_2 = soup.find_all('div',{'class':'tbCol'})[1].find('dl',{'class':'tbList'}).text.strip()
        tbList_3 = soup.find('div', {'class':'tbCol tbCoInfo'}).find('dl', {'class':'tbList'}).text

        
        # company_name = soup.find('div',{'class':'coInfo'}).find('h4').text
        company_name = soup.find('h3', {'class':'hd_3'}).find('div',{'class':'header'}).find('span',{'class':'coName'}).get_text().strip()

        

        # print("="*60)
        # print(company_name)
        # # print(tbList_1)
        # # print(tbList_2)
        # print(tbList_3)
        # print("="*60)

        
        
        # dt_list = tbList.find_all('dt')
        # dd_list = tbList.find_all('dd')
        # data_list=['']*7
        # for i in range(len(dd_list)):
        #     idx = columns.index(dt_list[i].text.strip())
        #     data_list[idx] = clean_string(dd_list[i].text)
        # dd_list = list(map(lambda x:clean_string(x.text), dd_list))
        # data_dict[company_name] = data_list

def crawl_list_link(data_dict, csv_filename, key=1):

    print(key)
    url = f'https://www.jobkorea.co.kr/Search/?duty=1000242&tabType=recruit&Page_No={key}'

    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

    # URL에서 HTML 컨텐츠 가져오기
    if response.status_code != 200:
        print(f"Failed to retrieve contents from {url}")
        return
    
    # BeautifulSoup를 사용하여 HTML 파싱
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 필요한 데이터를 추출하여 리스트에 추가
    article = soup.find('article',{'class':'list'}) 
    if article:
        links = [header.attrs['href'] for header in article.find_all('a',{'class':'information-title-link'})]
        crawl_list_in_info(data_dict, links)
        crawl_list_link(data_dict, csv_filename, key+1)

def main():
    csv_filename = 'output.csv'
    # 데이터를 저장할 리스트 초기화
    data_dict = {}
    crawl_list_link(data_dict, csv_filename)
    df = pd.DataFrame.from_dict(data_dict,orient='index')
    df.columns = ['경력','학력','스킬','핵심역량','우대','기본우대','자격증']
    df.index.name = '회사명'
    df.to_csv(csv_filename, encoding='utf-8-sig')
main()