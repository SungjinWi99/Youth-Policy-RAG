import unittest

from pydantic import ValidationError

from policy.utils import (
    REGION_CODES,
    build_region_metadata,
    extract_sido_codes,
    region_name_to_code,
)
from src.user.schemas import UserCreate, UserUpdate


class RegionTest(unittest.TestCase):
    def test_user_schema_accepts_only_canonical_region_names(self):
        user = UserCreate(user_id="user-1", region="서울")
        update = UserUpdate(region="부산")

        self.assertEqual(user.region, "서울")
        self.assertEqual(update.region, "부산")

        with self.assertRaises(ValidationError):
            UserCreate(user_id="user-2", region="서울특별시")

        with self.assertRaises(ValidationError):
            UserUpdate(region="임의 지역")

    def test_region_name_to_code_supports_existing_full_names(self):
        self.assertEqual(region_name_to_code("서울"), "11")
        self.assertEqual(region_name_to_code("서울특별시"), "11")
        self.assertEqual(region_name_to_code(" 충청북도 "), "43")
        self.assertIsNone(region_name_to_code("임의 지역"))

    def test_extract_sido_codes_handles_single_multi_and_legacy_codes(self):
        self.assertEqual(
            extract_sido_codes("11110,11140,26110"),
            {"11", "26"},
        )
        self.assertEqual(
            extract_sido_codes("42110,45110"),
            {"51", "52"},
        )
        self.assertEqual(
            extract_sido_codes("invalid,,123"),
            set(),
        )

    def test_build_region_metadata_marks_each_applicable_region(self):
        metadata = build_region_metadata("11110,26110")

        self.assertEqual(len(metadata), len(REGION_CODES))
        self.assertTrue(metadata["region_11"])
        self.assertTrue(metadata["region_26"])
        self.assertFalse(metadata["region_27"])


if __name__ == "__main__":
    unittest.main()
