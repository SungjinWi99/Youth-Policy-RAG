import os
import sys
import json
import argparse
import time
import requests
from dotenv import load_dotenv
from src.config import load_config

load_dotenv()
config = load_config()

# 신규 오픈 API URL (JSON 규격)
API_URL = "https://www.youthcenter.go.kr/go/ythip/getPlcy"

def fetch_page(api_key, page_num, page_size=100):
    """
    특정 페이지의 정책 데이터를 API(JSON)로부터 가져와 딕셔너리로 반환합니다.
    """
    params = {
        "apiKeyNm": api_key,
        "pageNum": page_num,
        "pageSize": page_size,
        "rtnType": "json"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(API_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\n[오류] API 호출 중 네트워크 예외가 발생했습니다: {e}")
        return None

    try:
        data = response.json()
        return data
    except ValueError as e:
        print(f"\n[오류] JSON 파싱 실패: {e}")
        print(f"응답 본문 일부: {response.text[:200]}")
        return None

def main():
    parser = argparse.ArgumentParser(description="온통청년 OpenAPI 청년정책 데이터 수집기")
    parser.add_argument("--limit-test", action="store_true", help="1페이지(10개)만 다운로드하여 테스트 실행")
    parser.add_argument("--display", type=int, default=100, help="한 번에 가져올 정책 수 (기본값: 100)")
    args = parser.parse_args()

    # API 인증키 확인
    api_key = os.getenv("YOUTH_API_KEY")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("[오류] .env 파일에 YOUTH_API_KEY가 설정되어 있지 않습니다.")
        print(".env 파일에 올바른 API 인증키를 입력한 후 다시 실행해주세요.")
        sys.exit(1)

    print("=" * 60)
    print(" 온통청년 청년정책 데이터 수집을 시작합니다. (JSON API)")
    print("=" * 60)

    # 1. 최초 호출을 통해 전체 개수 및 구조 파악
    display_cnt = 10 if args.limit_test else args.display
    print(f"[*] API 연결 및 최초 데이터 정보 조회 중... (pageSize={display_cnt})")

    first_data = fetch_page(api_key, page_num=1, page_size=display_cnt)
    if first_data is None:
        print("[오류] 첫 페이지 조회에 실패하여 수집을 중단합니다.")
        sys.exit(1)

    res_code = first_data.get("resultCode")
    res_msg = first_data.get("resultMessage")

    if res_code != 200:
        print(f"[오류] API 에러가 반환되었습니다. (코드: {res_code}, 메시지: {res_msg})")
        sys.exit(1)

    pagging = first_data.get("result", {}).get("pagging", {})
    total_cnt = pagging.get("totCount", 0)
    print(f"[+] 조회된 전체 정책 수: {total_cnt}건")

    if total_cnt == 0:
        print("[경고] 조회된 정책 수가 0건입니다. 인증키 상태를 다시 확인해주세요.")
        sys.exit(1)

    # 2. 루프 돌며 데이터 수집
    all_policies = []

    if args.limit_test:
        print("[*] 테스트 모드로 실행되어 1페이지(최대 10개)만 수집합니다.")
        policies = first_data.get("result", {}).get("youthPolicyList", [])
        all_policies.extend(policies)
        print(f"[+] 테스트 수집 완료: {len(all_policies)}개 정책 수집됨")
    else:
        # 전체 페이지 수 계산
        total_pages = (total_cnt + display_cnt - 1) // display_cnt
        print(f"[*] 총 {total_pages}회 요청을 통해 데이터 수집을 진행합니다.")

        # 첫 페이지 데이터 추가
        first_policies = first_data.get("result", {}).get("youthPolicyList", [])
        all_policies.extend(first_policies)
        print(f" -> [1/{total_pages}] 완료 ({len(all_policies)}/{total_cnt})")

        # 2페이지부터 순차적으로 호출
        for page in range(2, total_pages + 1):
            # API 서버 부하 방지를 위한 약간의 딜레이
            time.sleep(0.3)

            page_data = fetch_page(api_key, page_num=page, page_size=display_cnt)
            if page_data is None or page_data.get("resultCode") != 200:
                print(f"\n[경고] {page}페이지 요청 실패. 건너뜁니다.")
                continue

            policies = page_data.get("result", {}).get("youthPolicyList", [])
            all_policies.extend(policies)

            # 진행 상태 표시
            sys.stdout.write(f"\r -> [{page}/{total_pages}] 완료 ({len(all_policies)}/{total_cnt})")
            sys.stdout.flush()
        print("\n[+] 데이터 수집 완료!")

    # 3. JSON 파일로 저장
    print(f"[*] 수집한 데이터를 파일에 저장하는 중: {config.data.raw}")
    try:
        with open(config.data.raw, "w", encoding="utf-8") as f:
            json.dump(all_policies, f, ensure_ascii=False, indent=2)
        print(f"[+] 성공적으로 저장되었습니다! (총 {len(all_policies)}개 정책 저장됨)")
    except Exception as e:
        print(f"[오류] 파일 저장 실패: {e}")
        sys.exit(1)

    print("=" * 60)
    print(" 데이터 수집이 성공적으로 종료되었습니다.")
    print("=" * 60)

if __name__ == "__main__":
    main()
