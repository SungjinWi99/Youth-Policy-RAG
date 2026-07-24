from typing import Any, Callable
from datetime import date
from langchain_core.documents import Document

from src.rag.retrievers.base import RetrievalRequest
from src.rag.retrievers.filter import (
    add_policy_exclusion,
    build_filter_from_profile,
)


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
        metadata_filter = add_policy_exclusion(
            build_filter_from_profile(
                request.user_profile,
                exclude_expired=request.exclude_expired,
                today=self.today_provider(),
            ),
            request.excluded_policy_ids,
        )
        retriever = self._build_retriever(metadata_filter)
        return retriever.invoke(request.query)

    async def aretrieve(self, request: RetrievalRequest) -> list[Document]:
        metadata_filter = add_policy_exclusion(
            build_filter_from_profile(
                request.user_profile,
                exclude_expired=request.exclude_expired,
                today=self.today_provider(),
            ),
            request.excluded_policy_ids,
        )
        retriever = self._build_retriever(metadata_filter)
        return await retriever.ainvoke(request.query)
