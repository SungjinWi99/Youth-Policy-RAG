import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.policy.models import Policy, _load_policy_index
from src.policy.router import policy_router
from src.policy.schemas import PolicyDetail


class PolicyModelTest(unittest.TestCase):
    def tearDown(self):
        _load_policy_index.cache_clear()

    def test_get_policy_preserves_extra_source_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "policies.json"
            source_path.write_text(
                json.dumps(
                    [
                        {
                            "plcyNo": "policy-1",
                            "plcyNm": "테스트 정책",
                            "customField": "원본 필드",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            policy = Policy.get("policy-1", source_path)

        self.assertEqual(policy.plcyNm, "테스트 정책")
        self.assertEqual(policy.model_dump()["customField"], "원본 필드")

    def test_get_policy_raises_404_for_unknown_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "policies.json"
            source_path.write_text("[]", encoding="utf-8")

            with self.assertRaises(HTTPException) as context:
                Policy.get("unknown", source_path)

        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(
            context.exception.detail,
            "정책을 찾을 수 없습니다.",
        )


class PolicyRouterTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(policy_router)
        self.client = TestClient(app)

    @patch("src.policies.router.Policy.get")
    def test_get_policy_endpoint(self, get_policy):
        get_policy.return_value = PolicyDetail(
            plcyNo="policy-1",
            plcyNm="테스트 정책",
            customField="원본 필드",
        )

        response = self.client.get("/policies/policy-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["plcyNo"], "policy-1")
        self.assertEqual(response.json()["customField"], "원본 필드")
        get_policy.assert_called_once_with("policy-1")


if __name__ == "__main__":
    unittest.main()
