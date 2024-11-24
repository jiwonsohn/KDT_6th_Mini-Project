
from base import base

'''
    ### base 현황 출력 함수

* 함수이름: showBase
* 함수기능: 1번 공을 던진 후 승부 결과에 따른 base 갱신 상황 update
* 매개변수: str state(홈런,안타,아웃) name(공격팀 이름) / list doru(1루2루3루 상태) / int 4개 Bat, Ball outCount, home 
            bool full_base isStrike / 

* 함수결과: 베이스 현황 출력
'''
def show_base(state, doru, Bat, Ball,strike, outCount,home,full_base,name,isStrike):

    print(f'{"결과":*^40}')
    print(f'타자:    {Bat}',end="\t\t")
    print(f'투수:   {Ball}')


    # 기본값
    oneru,tworu,thrru=0,0,0             

    count=len(doru)

    # 베이스가 비어있을 시,
    if not count : 
        oneru=0
        tworu=0
        thrru=0

    # 베이스에 주자가 있는 경우
    elif count==1: 
        oneru=1                 # 1루 주자 O

    elif count==2: 
        oneru=1                 # 1루 주자 O
        tworu=1                 # 2루 주자 O

    else:
        oneru=1                 # 1루 주자 O
        tworu=1                 # 2루 주자 O
        thrru=1                 # 3루 주자 O

    base(state, oneru, tworu, thrru, strike, outCount,home,full_base,name,isStrike)


if __name__ =='__main__':       # 실행파일 이름=> __main__
    print("--TEST--")

    # print(show_base("만루",1,1,1,))
    # print(f'결과: {play_ball("삼성")}')