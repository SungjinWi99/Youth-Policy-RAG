from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.documents import Document

from src.rag.state import PolicySearchProfile


def normalize_user_gender(gender: str | None) -> str | None:
    if not gender:
        return None

    value = gender.strip().lower()

    if value in {"female", "woman", "women", "f", "여성", "여자"}:
        return "female"

    if value in {"male", "man", "men", "m", "남성", "남자"}:
        return "male"

    return None


def build_user_filter(
    user: PolicySearchProfile,
    *,
    exclude_expired: bool,
    today_yyyymmdd: int,
) -> dict | None:
    filters: list[dict] = []
    age = user.get("age")
    if age is not None:
        filters.extend([
            {"sprtTrgtMinAge": {"$lte": age}},
            {"sprtTrgtMaxAge": {"$gte": age}},
        ])

    gender = normalize_user_gender(user.get("gender"))
    if gender:
        filters.append({
            "genderPolicy": {"$in": ["all", gender]}
        })

    income = user.get("income")
    if income is not None:
        filters.append({
            "$or": [
                {"incomePolicy": {"$eq": "all"}},
                {
                    "$and": [
                        {"earnMinAmt": {"$lte": income}},
                        {"earnMaxAmt": {"$gte": income}},
                    ]
                },
            ]
        })

    region = user.get("region")
    if region:
        filters.append({
            "regionPolicy": {"$in": ["all", region]}
        })

    if exclude_expired:
        filters.append({
            "bizPrdEndYmd": {"$gte": today_yyyymmdd}
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
        user_profile: PolicySearchProfile,
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
        user_profile: PolicySearchProfile,
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
        user_profile: PolicySearchProfile,
        exclude_expired: bool = True,
    ) -> list[Document]:
        metadata_filter = self._build_filter(
            user_profile,
            exclude_expired,
        )

        retriever = self._build_retriever(metadata_filter)
        return await retriever.ainvoke(query)
