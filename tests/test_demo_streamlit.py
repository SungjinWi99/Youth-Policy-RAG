import unittest
from unittest.mock import patch

from demo_streamlit import (
    parse_sse_line,
    reset_conversation_history,
)


class StreamlitDemoTest(unittest.TestCase):
    def test_parse_metadata_event(self):
        event = parse_sse_line(
            'data: {"type":"metadata","data":{"contexts":["context-1"],'
            '"retrieved_policy_ids":["policy-1"]}}'
        )

        self.assertEqual(event["type"], "metadata")
        self.assertEqual(event["data"]["contexts"], ["context-1"])
        self.assertEqual(
            event["data"]["retrieved_policy_ids"],
            ["policy-1"],
        )

    def test_ignore_non_data_sse_line(self):
        self.assertIsNone(parse_sse_line(""))
        self.assertIsNone(parse_sse_line(": keep-alive"))

    @patch("demo_streamlit.request_json")
    def test_reset_conversation_history_calls_delete_api(
        self,
        request_json,
    ):
        request_json.return_value = (
            True,
            {"message": "대화 기록이 초기화되었습니다."},
        )

        result = reset_conversation_history(" user/name ")

        request_json.assert_called_once_with(
            "DELETE",
            "/chat/history/user%2Fname",
        )
        self.assertTrue(result[0])


if __name__ == "__main__":
    unittest.main()
