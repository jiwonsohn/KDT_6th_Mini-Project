
def check_Bat(bat_throw):
    

    if not len(bat_throw):              # 아무것도 입력하지 않을 시
        print("\n", "*"*56)
        print("시작한 경기는 종료될 수 없습니다. ^ㅁ^")
        print("*"*56,"\n")
        return None

    else:
        if bat_throw.isdecimal():       # 입력값 숫자 조합 여부 확인
            if 1<= int(bat_throw) <= 100:
                return int(bat_throw)
            
            else:
                print("\n","*"*56)
    
                print("유효한 범위의 숫자가 아닙니다.\n다시 입력하세요!")
                print("*"*56,"\n")
                return None

        else:                           # 빈칸, 숫자 외에 입력값일 경우
            print("\n", "*"*56)
            print("경기에 진지하게 임해주세요. +ㅁ+")
            print("*"*56,"\n")
            return None
        

if __name__ =='__main__':       # 실행파일 이름=> __main__
    print("--TEST--")

    # print(f'결과: {check_Bat("31")}')
    print(f'결과: {check_Bat("no")}')