import json
import unittest

import chromadb

from src.config import load_config
from src.rag.retriever import build_user_filter
from policy.utils import (
    REGION_CODES,
    REGION_NAMES,
    region_metadata_key,
    region_name_to_code,
)


TEST_TODAY = 20260703


def is_eligible(
    metadata: dict,
    profile: dict,
    *,
    exclude_expired: bool,
    today_yyyymmdd: int,
) -> bool:
    age = profile.get("age")
    if age is not None:
        age_policy = metadata.get("agePolicy")
        if age_policy == "specific" and not (
            metadata["sprtTrgtMinAge"]
            <= age
            <= metadata["sprtTrgtMaxAge"]
        ):
            return False
        if age_policy not in {"all", "unknown", "specific"}:
            return False

    income = profile.get("income")
    if income is not None:
        income_policy = metadata.get("incomePolicy")
        if income_policy == "specific" and not (
            metadata["earnMinAmt"]
            <= income
            <= metadata["earnMaxAmt"]
        ):
            return False
        if income_policy not in {"all", "unknown", "specific"}:
            return False

    region_code = region_name_to_code(profile.get("region"))
    if (
        region_code is not None
        and metadata.get(region_metadata_key(region_code)) is not True
    ):
        return False

    if exclude_expired:
        application_policy = metadata.get("applicationPolicy")
        if application_policy in {"fixed", "multi"}:
            if metadata["applicationEndYmd"] < today_yyyymmdd:
                return False
        elif application_policy not in {"rolling", "unknown"}:
            return False

    return True


class ChromaFilterIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config()
        client = chromadb.PersistentClient(
            path=config.path(config.data.chroma_dir)
        )
        cls.collection = client.get_collection(
            config.data.chroma_collection_name
        )
        records = cls.collection.get(include=["metadatas"])
        cls.metadata_by_id = dict(zip(
            records["ids"],
            records["metadatas"],
        ))

        with open(
            config.path(config.data.raw),
            encoding="utf-8",
        ) as raw_file:
            cls.raw_policy_ids = {
                item["plcyNo"]
                for item in json.load(raw_file)
            }

    def actual_ids(
        self,
        profile: dict,
        *,
        exclude_expired: bool,
    ) -> set[str]:
        metadata_filter = build_user_filter(
            profile,
            exclude_expired=exclude_expired,
            today_yyyymmdd=TEST_TODAY,
        )
        if metadata_filter is None:
            return set(self.metadata_by_id)
        return set(self.collection.get(
            where=metadata_filter,
            include=[],
        )["ids"])

    def expected_ids(
        self,
        profile: dict,
        *,
        exclude_expired: bool,
    ) -> set[str]:
        return {
            policy_id
            for policy_id, metadata in self.metadata_by_id.items()
            if is_eligible(
                metadata,
                profile,
                exclude_expired=exclude_expired,
                today_yyyymmdd=TEST_TODAY,
            )
        }

    def test_chroma_filter_matches_python_oracle(self):
        profiles = [
            {},
            *({"age": age} for age in [0, 18, 25, 39, 40, 99]),
            *(
                {"income": income}
                for income in [0, 3000, 4500, 5000, 40_000_000]
            ),
            *({"region": region} for region in REGION_NAMES),
            {
                "age": 25,
                "income": 3000,
                "region": "서울",
            },
            {
                "age": 40,
                "income": 5000,
                "region": "부산",
            },
            {
                "age": 18,
                "region": "제주",
                "gender": "여성",
            },
            {"gender": "여성"},
        ]

        for profile in profiles:
            for exclude_expired in [False, True]:
                with self.subTest(
                    profile=profile,
                    exclude_expired=exclude_expired,
                ):
                    self.assertEqual(
                        self.actual_ids(
                            profile,
                            exclude_expired=exclude_expired,
                        ),
                        self.expected_ids(
                            profile,
                            exclude_expired=exclude_expired,
                        ),
                    )

    def test_unknown_metadata_is_always_included(self):
        unknown_age_ids = {
            policy_id
            for policy_id, metadata in self.metadata_by_id.items()
            if metadata["agePolicy"] == "unknown"
        }
        unknown_income_ids = {
            policy_id
            for policy_id, metadata in self.metadata_by_id.items()
            if metadata["incomePolicy"] == "unknown"
        }
        unknown_application_ids = {
            policy_id
            for policy_id, metadata in self.metadata_by_id.items()
            if metadata["applicationPolicy"] == "unknown"
        }

        for age in [0, 25, 120]:
            self.assertLessEqual(
                unknown_age_ids,
                self.actual_ids(
                    {"age": age},
                    exclude_expired=False,
                ),
            )

        for income in [0, 3000, 40_000_000]:
            self.assertLessEqual(
                unknown_income_ids,
                self.actual_ids(
                    {"income": income},
                    exclude_expired=False,
                ),
            )

        self.assertLessEqual(
            unknown_application_ids,
            self.actual_ids({}, exclude_expired=True),
        )

    def test_expiration_toggle_and_boundaries(self):
        all_ids = self.actual_ids({}, exclude_expired=False)
        active_ids = self.actual_ids({}, exclude_expired=True)
        closed_ids = {
            policy_id
            for policy_id, metadata in self.metadata_by_id.items()
            if (
                metadata["applicationPolicy"] in {"fixed", "multi"}
                and metadata["applicationEndYmd"] < TEST_TODAY
            )
        }
        today_or_future_ids = {
            policy_id
            for policy_id, metadata in self.metadata_by_id.items()
            if (
                metadata["applicationPolicy"] in {"fixed", "multi"}
                and metadata["applicationEndYmd"] >= TEST_TODAY
            )
        }

        self.assertLessEqual(active_ids, all_ids)
        self.assertTrue(active_ids.isdisjoint(closed_ids))
        self.assertLessEqual(today_or_future_ids, active_ids)

    def test_chroma_metadata_schema_is_current(self):
        self.assertEqual(
            set(self.metadata_by_id),
            self.raw_policy_ids,
        )

        for policy_id, metadata in self.metadata_by_id.items():
            with self.subTest(policy_id=policy_id):
                self.assertIn(
                    metadata.get("agePolicy"),
                    {"all", "specific", "unknown"},
                )
                self.assertIn(
                    metadata.get("incomePolicy"),
                    {"all", "specific", "unknown"},
                )
                self.assertIn(
                    metadata.get("applicationPolicy"),
                    {"fixed", "multi", "rolling", "unknown"},
                )
                self.assertNotIn("genderPolicy", metadata)
                self.assertTrue(any(
                    metadata.get(region_metadata_key(code)) is True
                    for code in REGION_CODES
                ))


if __name__ == "__main__":
    unittest.main()
