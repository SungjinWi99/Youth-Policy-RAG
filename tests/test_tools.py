import unittest

from src.rag.retriever import PolicyRetriever
from src.rag.tools import create_search_policies_tool


class ToolsTest(unittest.TestCase):
    def test_search_tool_exposes_only_query_to_model(self):
        tool = create_search_policies_tool(
            PolicyRetriever(
                vector_store=object(),
                search_k=3,
            )
        )

        schema = tool.tool_call_schema.model_json_schema()

        self.assertEqual(
            set(schema["properties"]),
            {"query"},
        )
        self.assertNotIn("user_profile", schema["properties"])
        self.assertNotIn("exclude_expired", schema["properties"])
        self.assertIn(
            "사용자 프로필 값은 넣지 않는다",
            tool.description,
        )


if __name__ == "__main__":
    unittest.main()
