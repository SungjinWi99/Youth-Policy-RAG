from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.documents import Document

from src.rag.state import RAGUserProfile
from src.policy.utils import region_metadata_key, region_name_to_code


def build_user_filter(
    user: RAGUserProfile,
    *,
    exclude_expired: bool,
    today_yyyymmdd: int,
) -> dict | None:
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


class PolicyRetriever:
    def __init__(
        self,
        vector_store: Any,
        search_k: int,
        today_provider: Callable[[], date] = date.today,
    ):
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

    def _build_filter(
        self,
        user_profile: RAGUserProfile,
        exclude_expired: bool,
        *,
        today: date | None = None,
    ) -> dict | None:
        effective_today = (
            today
            if today is not None
            else self.today_provider()
        )
        today_yyyymmdd = int(
            effective_today.strftime("%Y%m%d")
        )

        return build_user_filter(
            user_profile,
            exclude_expired=exclude_expired,
            today_yyyymmdd=today_yyyymmdd,
        )

    def retrieve(
        self,
        query: str,
        user_profile: RAGUserProfile,
        exclude_expired: bool = True,
    ) -> list[Document]:
        metadata_filter = self._build_filter(
            user_profile,
            exclude_expired,
        )

        retriever = self._build_retriever(metadata_filter)
        return retriever.invoke(query)

    async def aretrieve(
        self,
        query: str,
        user_profile: RAGUserProfile,
        exclude_expired: bool = True,
    ) -> list[Document]:
        metadata_filter = self._build_filter(
            user_profile,
            exclude_expired,
        )

        retriever = self._build_retriever(metadata_filter)
        return await retriever.ainvoke(query)
