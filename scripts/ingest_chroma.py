import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factory import create_embedding_model
from policy.utils import (
    build_age_metadata,
    build_application_period_metadata,
    build_income_metadata,
)
from policy.utils import build_region_metadata


DEFAULT_RAW_PATH = PROJECT_ROOT / "data/raw/youth_policies.json"
DEFAULT_COLLECTION_NAME = "youth_policies_rag"
DEFAULT_BATCH_SIZE = 270
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def to_policy_end_date(value: Any) -> int:
    normalized = str(value or "").strip()
    if len(normalized) == 8 and normalized.isdigit():
        return int(normalized)
    return 99991231


def load_raw_policies(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as policy_file:
        policies = json.load(policy_file)
    if not isinstance(policies, list) or not policies:
        raise ValueError(f"{path}에는 비어 있지 않은 JSON 배열이 필요합니다.")

    policy_ids = []
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            raise ValueError(f"{path}[{index}] 정책이 JSON 객체가 아닙니다.")
        policy_id = str(policy.get("plcyNo") or "").strip()
        if not policy_id:
            raise ValueError(f"{path}[{index}]에 plcyNo가 없습니다.")
        policy_ids.append(policy_id)
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError(f"{path}에 중복 plcyNo가 있습니다.")
    return policies


def build_documents(
    policies: list[dict[str, Any]],
) -> tuple[list[Document], list[str]]:
    documents = []
    ids = []

    for item in policies:
        policy_id = str(item["plcyNo"]).strip()
        content = f"""
정책명: {item.get("plcyNm", "")}
키워드: {item.get("plcyKywdNm", "")}
카테고리: {item.get("lclsfNm", "")} > {item.get("mclsfNm", "")}
정책 설명: {item.get("plcyExplnCn", "")}
지원 내용: {item.get("plcySprtCn", "")}
""".strip()

        metadata = {
            "plcyNo": policy_id,
            "lclsfNm": str(item.get("lclsfNm") or ""),
            "mclsfNm": str(item.get("mclsfNm") or ""),
            "refUrlAddr1": str(item.get("refUrlAddr1") or ""),
            "refUrlAddr2": str(item.get("refUrlAddr2") or ""),
            "aplyUrlAddr": str(item.get("aplyUrlAddr") or ""),
            "sprvsnInstCdNm": str(item.get("sprvsnInstCdNm") or ""),
            "operInstCdNm": str(item.get("operInstCdNm") or ""),
            "bizPrdBgngYmd": str(item.get("bizPrdBgngYmd") or ""),
            "bizPrdEndYmd": to_policy_end_date(item.get("bizPrdEndYmd")),
            "bizPrdEtcCn": str(item.get("bizPrdEtcCn") or ""),
            "aplyYmd": str(item.get("aplyYmd") or ""),
            "plcyAplyMthdCn": str(item.get("plcyAplyMthdCn") or ""),
            "ptcpPrpTrgtCn": str(item.get("ptcpPrpTrgtCn") or ""),
            "addAplyQlfcCndCn": str(
                item.get("addAplyQlfcCndCn") or ""
            ),
            "sbmsnDcmntCn": str(item.get("sbmsnDcmntCn") or ""),
            "srngMthdCn": str(item.get("srngMthdCn") or ""),
            "region": str(item.get("rgtrInstCdNm") or ""),
            "zipCd": str(item.get("zipCd") or ""),
            "jobCd": str(item.get("jobCd") or ""),
            "mrgSttsCd": str(item.get("mrgSttsCd") or ""),
        }
        metadata.update(
            build_age_metadata(
                item.get("sprtTrgtMinAge"),
                item.get("sprtTrgtMaxAge"),
            )
        )
        metadata.update(
            build_income_metadata(
                item.get("earnMinAmt"),
                item.get("earnMaxAmt"),
            )
        )
        metadata.update(
            build_application_period_metadata(
                item.get("aplyYmd"),
                item.get("aplyPrdSeCd"),
            )
        )
        metadata.update(build_region_metadata(item.get("zipCd")))

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )
        ids.append(policy_id)

    return documents, ids


def create_passage_embedding_model(
    provider: str,
    model_name: str,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
):
    kwargs = {}
    if provider == "ollama":
        kwargs["base_url"] = ollama_base_url
    return create_embedding_model(
        provider=provider,
        model_name=model_name,
        **kwargs,
    )


def prepare_vector_store(
    chroma_dir: Path,
    collection_name: str,
    embedding_model: Any,
    distance_metric: str,
    recreate: bool,
) -> Chroma:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    existing_names = {
        collection.name
        for collection in client.list_collections()
    }
    if collection_name in existing_names:
        if not recreate:
            raise FileExistsError(
                f"{chroma_dir}에 collection '{collection_name}'이 이미 "
                "존재합니다. 다시 만들려면 --recreate를 사용하세요."
            )
        client.delete_collection(collection_name)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(chroma_dir),
        collection_metadata={"hnsw:space": distance_metric},
    )


def ingest_documents(
    vector_store: Chroma,
    documents: list[Document],
    ids: list[str],
    batch_size: int,
    sleep_seconds: float,
) -> None:
    for start in tqdm(
        range(0, len(documents), batch_size),
        desc="Chroma 적재",
    ):
        end = start + batch_size
        vector_store.add_documents(
            documents=documents[start:end],
            ids=ids[start:end],
        )
        if end < len(documents) and sleep_seconds > 0:
            time.sleep(sleep_seconds)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "config.yaml과 무관하게 provider/model/path를 지정해 "
            "청년정책 ChromaDB를 생성합니다."
        )
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("google", "ollama", "openai", "upstage"),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="문서 적재용 passage embedding 모델명",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        required=True,
        help="생성할 ChromaDB 디렉터리",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
    )
    parser.add_argument(
        "--distance-metric",
        choices=("cosine", "l2", "ip"),
        default="cosine",
        help="세 실험에서 같은 값을 사용해야 합니다(기본값: cosine)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="API rate limit 대응을 위한 batch 사이 대기 시간",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="동일 경로의 동일 collection을 삭제하고 다시 생성",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds는 0 이상이어야 합니다.")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    raw_path = project_path(args.raw_path)
    chroma_dir = project_path(args.chroma_dir)
    policies = load_raw_policies(raw_path)
    documents, ids = build_documents(policies)
    embedding_model = create_passage_embedding_model(
        provider=args.provider,
        model_name=args.model,
        ollama_base_url=args.ollama_base_url,
    )
    vector_store = prepare_vector_store(
        chroma_dir=chroma_dir,
        collection_name=args.collection,
        embedding_model=embedding_model,
        distance_metric=args.distance_metric,
        recreate=args.recreate,
    )
    ingest_documents(
        vector_store=vector_store,
        documents=documents,
        ids=ids,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )

    stored_count = vector_store._collection.count()
    if stored_count != len(documents):
        raise RuntimeError(
            f"적재 건수가 일치하지 않습니다: expected={len(documents)}, "
            f"stored={stored_count}"
        )
    print(
        f"Chroma ready: provider={args.provider}, model={args.model}, "
        f"path={chroma_dir}, collection={args.collection}, "
        f"metric={args.distance_metric}, count={stored_count}"
    )


if __name__ == "__main__":
    main()
