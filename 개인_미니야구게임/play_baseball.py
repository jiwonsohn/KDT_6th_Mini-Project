# ------------------------------------------------------------------
# 랜덤 추출을 활용한 간단 야구 게임

# 총 2회 (1회 초, 말)
# 타자와 투수 1-100 사이 자연수 랜덤 복원 추출
# 타자(사용자)가 뽑은 수가 투수(컴퓨터)가 뽑은 수와 차이가 작을수록(w.m, 근접) 승점이 높은 득점 시스템

#      abs(타자 - 투수) < 10     => 홈런, 베이스에 있는 주자 수 + 1  득점 & 함성 소리
# 10<= abs(타자 - 투수) < 40     => 안타, 1루 진루 (만루면서 안타일 시, 1점 득점 & 베이스 만루)
# 40<= abs(타자 - 투수) <100     => 스트라이크 +1

# 스트라이크 (strikeNumber: 스트라이크 한계 기준)회 시, 아웃 주자 +1
# 아웃 주자  (outNumber: 아웃 주자 한계 기준)회 시, 공격팀 공격 종료

# 홈팀, 원정팀 각 1번씩 공격 끝나면 1회 종료

# ------------------------------------------------------------------ 

from play_ball import play_ball

# ===============================================================================

# 홈팀 원정팀 이름 & 회차별 득점수 저장 딕셔너리 선언

keys    = ['name', '1회', '2회']

homeTeam = dict.fromkeys(keys)
awayTeam = dict.fromkeys(keys)


# 홈팀 원정팀 이름 선언
print("="* 68)
print()
print(f'{"홈팀과 원정팀을 선정하세요":=^56}\n')
homeTeam['name'] = input("홈팀    이름:    ").strip()
awayTeam['name'] = input("원정팀  이름:    ").strip()
print("="* 68)


# 총 2회 
print()
print(f'{"경기 시작":~^63}')
print()

for j in range(2):

    print()
    print(f'{"="*29} {j+1}회 시작 {"="*29}\n')
    print()

    for i in range(2):                          # n회 초, 말

        if not i:                               # n회 초, 원정팀 공격

            print(f'{"="*30} {j+1}회 초 {"="*30}\n')
            print(f'{"*"*28} {awayTeam["name"]} 공격 {"*"*28}')
            print()

            attackTeam = awayTeam["name"]
            awayTeam[list(awayTeam.keys())[j+1]] = play_ball(attackTeam) # 원정팀 득점 수


        else:                                   # n회 말, 홈팀 공격
            print(f'{"="*30}{j+1}회 말{"="*30}\n')
            print(f'{"="*28} {homeTeam["name"]} 공격 {"="*28}')
            print()

            attackTeam = homeTeam["name"]   
            homeTeam[list(homeTeam.keys())[j+1]] = play_ball(attackTeam) # 홈팀 득점 수

    
    print(f'\n{"="*29} {j+1}회 종료 {"="*29}')
    print()

    print("-"*56)
    print(f'\n홈팀\t{j+1}회\t{homeTeam[list(homeTeam.keys())[j+1]]} 득점')
    print(f'원정팀\t{j+1}회\t{awayTeam[list(awayTeam.keys())[j+1]]} 득점\n')
    print("-"*56)

print()
print(f'{"경기 종료":~^63}')
print()

# 경기 결과 출력

print()
print("="*58)
print(f'{"SCORE":^56}')
print("="*58)


print(f'{"팀":^10} {"1회":^22} {"2회":^27}' )
print("-"*58)
print(f"{homeTeam['name']:^8} {homeTeam['1회']:^25} {homeTeam['2회']:^25}")
print("-"*58)
print(f"{awayTeam['name']:^8} {awayTeam['1회']:^25} {awayTeam['2회']:^25}")
print("="*58)
print()



# 승부 비교
away_score = awayTeam["1회"] + awayTeam["2회"]
home_score = homeTeam["1회"] + homeTeam["2회"]

if away_score > home_score: 
    print("="*58)
    print(f'{"경기 결과":^56}')
    print("="*58)
    print(f'{awayTeam["name"]}이(가) {abs(away_score - home_score)}점 차로 이겼습니다.')
    print("="*58)

elif away_score < home_score:
    print("="*58)
    print(f'{"경기 결과":^56}')
    print("="*58)
    print(f'{homeTeam["name"]}이(가) {abs(away_score - home_score)}점 차로 이겼습니다.')
    print("="*58)

else:
    print("="*58)
    print(f'{"경기 결과":^56}')
    print("="*58)
    print(f'{homeTeam["name"]}과(와) {awayTeam["name"]}는 무승부입니다.')
    print("="*58)

print()
print()
print()
print("="*68)
print(f'{"경기 끄으으으으으읕!":^56}')
print("="*68)
print()




