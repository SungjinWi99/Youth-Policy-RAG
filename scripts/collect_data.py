"""온통청년 API 정책 스냅샷을 안전하게 수집한다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.policy.corpus import write_policy_snapshot_atomically
from src.policy.source import fetch_policies


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="온통청년 OpenAPI 청년정책 데이터 수집기"
    )
    parser.add_argument(
        "--limit-test",
        action="store_true",
        help="첫 페이지 10건만 다운로드하여 연결과 저장을 확인",
    )
    parser.add_argument(
        "--display",
        type=int,
        default=100,
        help="API 페이지당 정책 수 (기본값: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "저장 경로. 생략하면 전체 수집은 config.data.raw, "
            "연결 테스트는 data/raw/youth_policies.sample.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.display < 1:
        parser.error("--display는 1 이상이어야 합니다.")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = str(os.getenv("YOUTH_API_KEY") or "").strip()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise SystemExit(".env에 올바른 YOUTH_API_KEY를 설정해주세요.")

    config = load_config()
    default_output = (
        "data/raw/youth_policies.sample.json"
        if args.limit_test
        else config.data.raw
    )
    raw_path = Path(
        config.path(str(args.output or default_output))
    )
    page_size = 10 if args.limit_test else args.display
    policies = fetch_policies(
        api_key=api_key,
        page_size=page_size,
        max_pages=1 if args.limit_test else None,
    )
    if not policies:
        raise RuntimeError("API에서 수집된 정책이 없습니다.")

    write_policy_snapshot_atomically(raw_path, policies)
    mode = "연결 테스트" if args.limit_test else "전체 수집"
    print(
        f"{mode} 완료: count={len(policies)}, raw={raw_path}"
    )


if __name__ == "__main__":
    main()
