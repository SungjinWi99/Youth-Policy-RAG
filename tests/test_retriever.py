import unittest
from datetime import date

from src.rag.retriever import PolicyRetriever, build_user_filter


class RetrieverTest(unittest.TestCase):
    def test_build_user_filter_combines_profile_conditions(self):
        result = build_user_filter(
            {
                "age": 25,
                "gender": "여성",
                "income": 3000,
                "region": "서울",
            },
            exclude_expired=True,
            today_yyyymmdd=20260701,
        )

        self.assertEqual(
            result,
            {
                "$and": [
                    {
                        "$or": [
                            {
                                "agePolicy": {
                                    "$in": ["all", "unknown"]
                                }
                            },
                            {
                                "$and": [
                                    {
                                        "agePolicy": {
                                            "$eq": "specific"
                                        }
                                    },
                                    {
                                        "sprtTrgtMinAge": {
                                            "$lte": 25
                                        }
                                    },
                                    {
                                        "sprtTrgtMaxAge": {
                                            "$gte": 25
                                        }
                                    },
                                ]
                            },
                        ]
                    },
                    {
                        "$or": [
                            {
                                "incomePolicy": {
                                    "$in": ["all", "unknown"]
                                }
                            },
                            {
                                "$and": [
                                    {
                                        "incomePolicy": {
                                            "$eq": "specific"
                                        }
                                    },
                                    {"earnMinAmt": {"$lte": 3000}},
                                    {"earnMaxAmt": {"$gte": 3000}},
                                ]
                            },
                        ]
                    },
                    {
                        "region_11": {"$eq": True}
                    },
                    {
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
                                            "$gte": 20260701
                                        }
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
        )

    def test_gender_does_not_create_metadata_filter(self):
        result = build_user_filter(
            {"gender": "여성"},
            exclude_expired=False,
            today_yyyymmdd=20260701,
        )

        self.assertIsNone(result)

    def test_build_user_filter_normalizes_legacy_region_name(self):
        result = build_user_filter(
            {"region": "충청북도"},
            exclude_expired=False,
            today_yyyymmdd=20260701,
        )

        self.assertEqual(result, {"region_43": {"$eq": True}})

    def test_build_user_filter_ignores_unknown_legacy_region(self):
        result = build_user_filter(
            {"region": "알 수 없는 지역"},
            exclude_expired=False,
            today_yyyymmdd=20260701,
        )

        self.assertIsNone(result)

    def test_build_user_filter_returns_none_without_conditions(self):
        self.assertIsNone(
            build_user_filter(
                {},
                exclude_expired=False,
                today_yyyymmdd=20260701,
            )
        )

    def test_explicit_today_overrides_today_provider(self):
        provider_calls = []
        retriever = PolicyRetriever(
            vector_store=object(),
            search_k=3,
            today_provider=lambda: (
                provider_calls.append(True)
                or date(2026, 1, 1)
            ),
        )

        result = retriever._build_filter(
            {},
            True,
            today=date(2026, 7, 1),
        )

        self.assertEqual(
            result,
            {
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
                                    "$gte": 20260701
                                }
                            },
                        ]
                    },
                ]
            },
        )
        self.assertEqual(provider_calls, [])


if __name__ == "__main__":
    unittest.main()
