
import random as rad
import playsound

from show_base import show_base
from check_Bat import check_Bat 

def play_ball(name):
        
    # 기준
    homeRun = 10                    # 홈런 기준
    outNumber = 2                   # 아웃 주자 한계 기준
    strikeNumber = 2                # 스트라이크 한계 기준

    attack_break = 5                # 공격 기회 당 최고 득점 수

    
    # 경기 현황 변수
    outCount=0                  # 아웃된 선수 누적합계
    home = 0                    # 득점 수 저장
    doru    = []                # 1루 - 2루 -3루  / 0-> 주자 없음 1-> 주자 있음 / 안타 1번 마다 1이 뒤 인덱스 원소로 이동 /

    full_base=False             # 만루 & 스트라이크일 시, 득점 상황 출력 방지 목적 
        
    while outCount != outNumber:
        
        strike = 0              # 스트라이크 횟수

        # 1명 타자 공격
        while strike!=strikeNumber:

            isOut=False                 # Out 여부 check

            # Attack_Break
            if home >= attack_break: 
                print("*"*52)
                b=f"득점이 최대점인 {attack_break}점 이상이므로 공격을 종료합니다"
                print(f'{b:^30}')
                print("*"*52)
                print()
                print()
                
                return attack_break         # 최대 득점 수 반환

            print("-"*59)
            print()
            print("투수가 공을 던졌습니다.")
            print("타자는 공격하세요!\n")

            bat_throw =  input("1에서 100 사이 숫자를 입력하세요:     ").strip()
            print()
            
            Bat  = check_Bat(bat_throw)
            Ball = rad.randint(1,100)

            if Bat==None: continue          # 유효한 입력값을 입력할 때까지 반복

            # 홈런
            elif abs(Bat - Ball) < homeRun:

                # 
                playsound.playsound(r"C:\Users\KDP-43\Desktop\presentation.ver\cong.mp3")
                home = home + 1 + sum(doru)  
                # 갱신 득점 = 기존 득점 + 타자 + (1루2루3루 주자 수)

                strike=0                # 스트라이크 리셋          
                doru=[]                 # 베이스 리셋

                full_base=False             
                state="홈런!!!!!!!!!!!!!!!!!!"

                show_base(state,doru,Bat,Ball,strike,outCount,home,full_base,name,isOut)
                print()

            # 1루 진루
            elif homeRun<= abs(Bat - Ball) <40 :

                doru.append(1)
                state="안타"
                
                strike=0                # 타자 진루 / strike reset
                
                # 만루일 때,
                if len(doru)>3: 

                    full_base=True

                    state="만루인 상황에서 안타"
                    home+=1
                    del doru[-1]

                show_base(state,doru,Bat,Ball,strike,outCount,home,full_base,name,isOut)
    

            # 스트라이크
            else:

                full_base=False             
                strike+=1
                state="스트라이크"
                
                # strikeNumber_진 아웃
                if strike == strikeNumber: 

                    outCount+=1                 
                    isOut=True

                show_base(state,doru,Bat,Ball,strike,outCount,home,full_base,name,isOut)

    print("*"*52)
    b=f"아웃된 선수가 {outNumber}명이므로 공격을 종료합니다"
    print(f'{b:^30}')
    print("*"*52,"\n")

    if home> attack_break: return attack_break
    else:                  return home



# print(f'__name__: {__name__}')


if __name__ =='__main__':       # 실행파일 이름=> __main__
    print("--TEST--")

    print(f'결과: {play_ball("삼성")}')