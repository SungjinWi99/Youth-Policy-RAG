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
                    {"sprtTrgtMinAge": {"$lte": 25}},
                    {"sprtTrgtMaxAge": {"$gte": 25}},
                    {
                        "genderPolicy": {
                            "$in": ["all", "female"]
                        }
                    },
                    {
                        "$or": [
                            {"incomePolicy": {"$eq": "all"}},
                            {
                                "$and": [
                                    {"earnMinAmt": {"$lte": 3000}},
                                    {"earnMaxAmt": {"$gte": 3000}},
                                ]
                            },
                        ]
                    },
                    {
                        "regionPolicy": {
                            "$in": ["all", "서울"]
                        }
                    },
                    {
                        "bizPrdEndYmd": {
                            "$gte": 20260701
                        }
                    },
                ]
            },
        )

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
            {"bizPrdEndYmd": {"$gte": 20260701}},
        )
        self.assertEqual(provider_calls, [])


if __name__ == "__main__":
    unittest.main()
