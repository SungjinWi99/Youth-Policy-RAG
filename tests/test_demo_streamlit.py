import unittest

from demo_streamlit import parse_sse_line


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


if __name__ == "__main__":
    unittest.main()
