import re
from datetime import date

from src.policy.utils import region_metadata_key, region_name_to_code
from src.rag.state import RAGUserProfile

TOKEN_PATTERN = re.compile(r"[0-9a-zA-Z]+|[가-힣]+")

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


def add_policy_exclusion(
    metadata_filter: dict | None,
    excluded_policy_ids: frozenset[str],
) -> dict | None:
    if not excluded_policy_ids:
        return metadata_filter

    exclusion_filter = {
        "plcyNo": {
            "$nin": sorted(excluded_policy_ids),
        }
    }
    if metadata_filter is None:
        return exclusion_filter
    return {
        "$and": [
            metadata_filter,
            exclusion_filter,
        ]
    }
