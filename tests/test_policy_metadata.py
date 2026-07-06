import unittest

from policy.utils import (
    build_age_metadata,
    build_application_period_metadata,
    build_income_metadata,
)


class PolicyMetadataTest(unittest.TestCase):
    def test_zero_age_range_means_all_ages(self):
        self.assertEqual(
            build_age_metadata("0", "0"),
            {
                "sprtTrgtMinAge": 0,
                "sprtTrgtMaxAge": 0,
                "agePolicy": "all",
            },
        )

    def test_valid_and_invalid_age_ranges_are_distinguished(self):
        self.assertEqual(
            build_age_metadata("19", "39")["agePolicy"],
            "specific",
        )
        self.assertEqual(
            build_age_metadata("39", "19")["agePolicy"],
            "unknown",
        )
        self.assertEqual(
            build_age_metadata("", "")["agePolicy"],
            "unknown",
        )

    def test_zero_income_range_means_all_incomes(self):
        self.assertEqual(
            build_income_metadata("0", "0"),
            {
                "earnMinAmt": 0,
                "earnMaxAmt": 0,
                "incomePolicy": "all",
            },
        )

    def test_valid_and_invalid_income_ranges_are_distinguished(self):
        self.assertEqual(
            build_income_metadata("0", "4500")["incomePolicy"],
            "specific",
        )
        self.assertEqual(
            build_income_metadata("7500", "5000")["incomePolicy"],
            "unknown",
        )
        self.assertEqual(
            build_income_metadata("", "")["incomePolicy"],
            "unknown",
        )

    def test_fixed_application_period_uses_explicit_dates(self):
        self.assertEqual(
            build_application_period_metadata(
                "20260401 ~ 20260414",
                "0057001",
            ),
            {
                "applicationPolicy": "fixed",
                "applicationStartYmd": 20260401,
                "applicationEndYmd": 20260414,
            },
        )

    def test_multi_application_period_uses_first_start_and_last_end(self):
        self.assertEqual(
            build_application_period_metadata(
                (
                    "20260615 ~ 20260706\\N"
                    "20260907 ~ 20260928\\N"
                    "20261026 ~ 20261116"
                ),
                "0057001",
            ),
            {
                "applicationPolicy": "multi",
                "applicationStartYmd": 20260615,
                "applicationEndYmd": 20261116,
            },
        )

    def test_rolling_and_unknown_application_periods_are_distinguished(self):
        self.assertEqual(
            build_application_period_metadata("", "0057002"),
            {
                "applicationPolicy": "rolling",
                "applicationStartYmd": 0,
                "applicationEndYmd": 0,
            },
        )
        self.assertEqual(
            build_application_period_metadata("", "0057003"),
            {
                "applicationPolicy": "unknown",
                "applicationStartYmd": 0,
                "applicationEndYmd": 0,
            },
        )

    def test_invalid_application_period_is_unknown(self):
        self.assertEqual(
            build_application_period_metadata(
                "20260230 ~ 20260301",
                "0057001",
            )["applicationPolicy"],
            "unknown",
        )


if __name__ == "__main__":
    unittest.main()
