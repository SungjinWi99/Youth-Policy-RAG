import asyncio
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Protocol

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.rag.state import RAGUserProfile
from src.policy.utils import region_metadata_key, region_name_to_code

TOKEN_PATTERN = re.compile(r"[0-9a-zA-Z]+|[가-힣]+")
KIWI_CONTENT_TAGS = frozenset({"NNG", "NNP", "VV", "VA", "VX", "XR", "SL", "SH", "SN"})


def build_filter_from_profile(
    user: RAGUserProfile,
    *,
    exclude_expired: bool,
    today: date,
) -> dict | None:
    today_yyyymmdd = int(today.strftime("%Y%m%d"))
    filters: list[dict] = []
    age = user.get("age")
    if age is not None:
        filters.append({
            "$or": [
                {"agePolicy": {"$in": ["all", "unknown"]}},
                {
                    "$and": [
                        {"agePolicy": {"$eq": "specific"}},
                        {"sprtTrgtMinAge": {"$lte": age}},
                        {"sprtTrgtMaxAge": {"$gte": age}},
                    ]
                },
            ]
        })

    income = user.get("income")
    if income is not None:
        filters.append({
            "$or": [
                {
                    "incomePolicy": {
                        "$in": ["all", "unknown"]
                    }
                },
                {
                    "$and": [
                        {"incomePolicy": {"$eq": "specific"}},
                        {"earnMinAmt": {"$lte": income}},
                        {"earnMaxAmt": {"$gte": income}},
                    ]
                },
            ]
        })

    region_code = region_name_to_code(user.get("region"))
    if region_code:
        filters.append({
            region_metadata_key(region_code): {"$eq": True}
        })

    if exclude_expired:
        filters.append({
            "$or": [
                {
                    "applicationPolicy": {
                        "$in": ["rolling", "unknown"]
                    }
                },
                {
                    "$and": [
                        {
                            "applicationPolicy": {
                                "$in": ["fixed", "multi"]
                            }
                        },
                        {
                            "applicationEndYmd": {
                                "$gte": today_yyyymmdd
                            }
                        },
                    ]
                },
            ]
        })

    if not filters:
        return None

    if len(filters) == 1:
        return filters[0]

    return {"$and": filters}


def tokenize_korean_legacy(text: str) -> list[str]:
    """Split text into words and Korean character bigrams without dependencies."""
    tokens: list[str] = []
    for chunk in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(chunk)
        if not re.fullmatch(r"[가-힣]+", chunk) or len(chunk) <= 2:
            continue
        tokens.extend(
            chunk[index:index + 2]
            for index in range(len(chunk) - 1)
        )
    return tokens


@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    """Load the Kiwi model once per process."""
    return Kiwi()


def tokenize_korean_lexical(text: str) -> list[str]:
    """Tokenize Korean text with Kiwi and retain contiguous compound nouns."""
    tokens: list[str] = []
    noun_run: list = []

    def flush_noun_run() -> None:
        if len(noun_run) > 1:
            tokens.append("".join(token.form.lower() for token in noun_run))
        noun_run.clear()

    for token in _get_kiwi().tokenize(text):
        tag = token.tag.split("-", maxsplit=1)[0]
        if tag not in KIWI_CONTENT_TAGS:
            flush_noun_run()
            continue

        if tag not in {"NNG", "NNP"}:
            flush_noun_run()
        elif noun_run:
            previous = noun_run[-1]
            if previous.start + previous.len != token.start:
                flush_noun_run()

        tokens.append(token.form.lower())
        if tag in {"NNG", "NNP"}:
            noun_run.append(token)

    flush_noun_run()
    return tokens


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    user_profile: RAGUserProfile
    exclude_expired: bool


class PolicyRetriever(Protocol):
    search_k: int

    def retrieve(self, request: RetrievalRequest) -> list[Document]: ...

    async def aretrieve(self, request: RetrievalRequest) -> list[Document]: ...


class DensePolicyRetriever:
    def __init__(
        self,
        vector_store: Any,
        search_k: int,
        today_provider: Callable[[], date] = date.today,
    ):
        if search_k < 1:
            raise ValueError("search_k는 1 이상이어야 합니다.")
        self.vector_store = vector_store
        self.search_k = search_k

        # 만료일 필터 테스트를 재현할 수 있도록 날짜 제공자를 주입한다.
        self.today_provider = today_provider

    def _build_retriever(
        self,
        metadata_filter: dict | None,
    ):
        search_kwargs: dict[str, Any] = {
            "k": self.search_k,
        }

        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )

    def retrieve(self, request: RetrievalRequest) -> list[Document]:
        metadata_filter = build_filter_from_profile(
            request.user_profile,
            exclude_expired=request.exclude_expired,
            today=self.today_provider(),
        )
        retriever = self._build_retriever(metadata_filter)
        return retriever.invoke(request.query)

    async def aretrieve(self, request: RetrievalRequest) -> list[Document]:
        metadata_filter = build_filter_from_profile(
            request.user_profile,
            exclude_expired=request.exclude_expired,
            today=self.today_provider(),
        )
        retriever = self._build_retriever(metadata_filter)
        return await retriever.ainvoke(request.query)


class BM25DocumentIndex:
    def __init__(
        self,
        documents: Iterable[Document],
        *,
        tokenizer: Callable[[str], list[str]] = tokenize_korean_lexical,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("BM25 k1은 0보다 커야 합니다.")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b는 0과 1 사이여야 합니다.")

        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.documents: dict[str, Document] = {}
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.document_lengths: dict[str, int] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)

        for document in documents:
            policy_id = str(document.metadata.get("plcyNo") or "").strip()
            if not policy_id:
                raise ValueError("BM25 문서에 plcyNo 메타데이터가 필요합니다.")
            if policy_id in self.documents:
                raise ValueError(f"중복 plcyNo: {policy_id}")
            term_frequency = Counter(tokenizer(document.page_content))
            self.documents[policy_id] = document
            self.term_frequencies[policy_id] = term_frequency
            self.document_lengths[policy_id] = sum(term_frequency.values())
            for term in term_frequency:
                self.postings[term].add(policy_id)

        if not self.documents:
            raise ValueError("BM25 인덱스에 문서가 필요합니다.")
        self.average_document_length = (
            sum(self.document_lengths.values()) / len(self.document_lengths)
        )

    def search(
        self,
        query: str,
        *,
        limit: int,
        allowed_policy_ids: set[str] | None = None,
    ) -> list[tuple[Document, float]]:
        if limit < 1:
            raise ValueError("BM25 limit은 1 이상이어야 합니다.")
        query_terms = set(self.tokenizer(query))
        if not query_terms:
            return []

        allowed = (
            set(self.documents)
            if allowed_policy_ids is None
            else set(allowed_policy_ids)
        )
        scores: dict[str, float] = defaultdict(float)
        document_count = len(self.documents)
        for term in query_terms:
            matching_policy_ids = self.postings.get(term, set()) & allowed
            if not matching_policy_ids:
                continue
            document_frequency = len(self.postings[term])
            inverse_document_frequency = math.log(
                1 + (
                    document_count - document_frequency + 0.5
                ) / (document_frequency + 0.5)
            )
            for policy_id in matching_policy_ids:
                term_frequency = self.term_frequencies[policy_id][term]
                document_length = self.document_lengths[policy_id]
                denominator = term_frequency + self.k1 * (
                    1 - self.b
                    + self.b * document_length / self.average_document_length
                )
                scores[policy_id] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1) / denominator
                )

        ranked_policy_ids = sorted(
            scores,
            key=lambda policy_id: (-scores[policy_id], policy_id),
        )[:limit]

        return [
            (self.documents[policy_id], scores[policy_id])
            for policy_id in ranked_policy_ids
        ]

def load_chroma_documents(collection: Any) -> list[Document]:
    stored = collection.get(include=["documents", "metadatas"])
    return [
        Document(
            page_content=page_content or "",
            metadata=dict(metadata or {}, plcyNo=policy_id),
        )
        for policy_id, page_content, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"],
            strict=True,
        )
    ]


class BM25PolicyRetriever:
    def __init__(
        self,
        collection: Chroma,
        search_k: int,
        today_provider: Callable[[], date] = date.today,
        tokenizer: Callable[[str], list[str]] = tokenize_korean_lexical,
    ):
        if search_k < 1:
            raise ValueError("search_k는 1 이상이어야 합니다.")
        self.collection = collection
        self.index = BM25DocumentIndex(
            load_chroma_documents(collection),
            tokenizer=tokenizer,
        )
        self.search_k = search_k
        self.today_provider = today_provider

    def _eligible_policy_ids(self, request: RetrievalRequest) -> set[str]:
        metadata_filter = build_filter_from_profile(
            request.user_profile,
            exclude_expired=request.exclude_expired,
            today=self.today_provider(),
        )
        get_kwargs: dict[str, Any] = {"include": []}
        if metadata_filter:
            get_kwargs["where"] = metadata_filter
        return set(self.collection.get(**get_kwargs)["ids"])

    def retrieve(self, request: RetrievalRequest) -> list[Document]:
        result = self.index.search(
            request.query,
            limit=self.search_k,
            allowed_policy_ids=self._eligible_policy_ids(request),
        )
        return [document for document, _ in result]

    async def aretrieve(self, request: RetrievalRequest) -> list[Document]:
        filtered_ids = await asyncio.to_thread(
            self._eligible_policy_ids,
            request,
        )
        result = await asyncio.to_thread(
            self.index.search,
            request.query,
            limit=self.search_k,
            allowed_policy_ids=filtered_ids,
        )
        return [document for document, _ in result]


class EnsemblePolicyRetriever:
    def __init__(
        self,
        retrievers: Sequence[PolicyRetriever],
        weights: Sequence[float],
        search_k: int,
        rrf_k: int,
    ):
        if not retrievers:
            raise ValueError("하나 이상의 retriever가 필요합니다.")
        if len(retrievers) != len(weights):
            raise ValueError("retrievers와 weights의 길이가 같아야 합니다.")
        if any(weight < 0 for weight in weights):
            raise ValueError("weight는 0 이상이어야 합니다.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("하나 이상의 weight가 0보다 커야 합니다.")
        if search_k < 1:
            raise ValueError("search_k는 1 이상이어야 합니다.")
        if rrf_k < 1:
            raise ValueError("rrf_k는 1 이상이어야 합니다.")

        self.retrievers = list(retrievers)
        self.weights = [float(weight) for weight in weights]
        self.search_k = search_k
        self.rrf_k = rrf_k

    def fuse_results(
        self,
        source_results: Sequence[Sequence[Document]],
    ) -> list[Document]:
        if len(source_results) != len(self.retrievers):
            raise ValueError("source_results와 retrievers의 길이가 같아야 합니다.")

        scores: dict[str, float] = defaultdict(float)
        documents_by_id: dict[str, Document] = {}
        source_ranks: dict[str, list[float]] = {}
        first_seen: dict[str, int] = {}
        seen_order = 0

        for source_index, (retrieved_documents, weight) in enumerate(zip(
            source_results,
            self.weights,
            strict=True,
        )):
            seen_in_source = set()
            for rank, document in enumerate(retrieved_documents, start=1):
                policy_id = str(document.metadata.get("plcyNo") or "").strip()
                if not policy_id:
                    raise ValueError("Ensemble 문서에 plcyNo 메타데이터가 필요합니다.")
                if policy_id in seen_in_source:
                    continue
                seen_in_source.add(policy_id)
                documents_by_id.setdefault(policy_id, document)
                source_ranks.setdefault(
                    policy_id,
                    [math.inf] * len(source_results),
                )[source_index] = rank
                if policy_id not in first_seen:
                    first_seen[policy_id] = seen_order
                    seen_order += 1
                scores[policy_id] += weight / (self.rrf_k + rank)

        ranked_ids = sorted(
            scores,
            key=lambda policy_id: (
                -scores[policy_id],
                *source_ranks[policy_id],
                first_seen[policy_id],
                policy_id,
            ),
        )[:self.search_k]
        return [documents_by_id[policy_id] for policy_id in ranked_ids]

    def retrieve_with_sources(
        self,
        request: RetrievalRequest,
    ) -> tuple[list[Document], list[list[Document]]]:
        source_results = [
            retriever.retrieve(request)
            for retriever in self.retrievers
        ]
        return self.fuse_results(source_results), source_results

    def retrieve(self, request: RetrievalRequest) -> list[Document]:
        documents, _ = self.retrieve_with_sources(request)
        return documents

    async def aretrieve_with_sources(
        self,
        request: RetrievalRequest,
    ) -> tuple[list[Document], list[list[Document]]]:
        source_results = await asyncio.gather(*[
            retriever.aretrieve(request)
            for retriever in self.retrievers
        ])
        return self.fuse_results(source_results), source_results

    async def aretrieve(self, request: RetrievalRequest) -> list[Document]:
        documents, _ = await self.aretrieve_with_sources(request)
        return documents
