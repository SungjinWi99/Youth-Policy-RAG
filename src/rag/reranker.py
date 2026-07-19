from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import requests
from langchain_core.documents import Document


@dataclass(frozen=True)
class RerankedDocument:
    document: Document
    original_rank: int
    relevance_score: float


class LlamaCppReranker:
    """llama.cpp의 Jina 호환 rerank endpoint를 호출한다."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 60.0,
        session: Any | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[Document],
    ) -> list[RerankedDocument]:
        if not documents:
            return []

        response = self.session.post(
            f"{self.base_url}/v1/rerank",
            json={
                "model": self.model,
                "query": query,
                "documents": [document.page_content for document in documents],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("reranker 응답에 results 목록이 없습니다.")

        reranked = []
        seen_indexes = set()
        for result in raw_results:
            if not isinstance(result, dict):
                raise ValueError("reranker result가 객체가 아닙니다.")
            index = result.get("index")
            score = result.get("relevance_score")
            if (
                not isinstance(index, int)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                raise ValueError("reranker result의 index 또는 score가 유효하지 않습니다.")
            if index < 0 or index >= len(documents) or index in seen_indexes:
                raise ValueError("reranker result index가 범위를 벗어나거나 중복됩니다.")
            seen_indexes.add(index)
            reranked.append(RerankedDocument(
                document=documents[index],
                original_rank=index + 1,
                relevance_score=float(score),
            ))

        if len(reranked) != len(documents):
            raise ValueError(
                "reranker가 일부 문서를 누락했습니다: "
                f"expected={len(documents)}, actual={len(reranked)}"
            )

        return sorted(
            reranked,
            key=lambda result: (
                -result.relevance_score,
                result.original_rank,
            ),
        )
