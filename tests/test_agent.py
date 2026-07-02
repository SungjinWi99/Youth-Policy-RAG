import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.rag.agent import PolicyAgent


class FakeSearchTool:
    name = "search_policies"


class RecordingToolModel(RunnableLambda):
    def __init__(self):
        super().__init__(
            lambda _: AIMessage(content="최종 답변")
        )
        self.bind_calls = []

    def bind_tools(
        self,
        tools,
        *,
        tool_choice=None,
        parallel_tool_calls=None,
    ):
        self.bind_calls.append({
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        })
        if tool_choice == "search_policies":
            return RunnableLambda(
                lambda _: AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_policies",
                            "args": {
                                "query": "청년 주거비 지원 정책"
                            },
                            "id": "forced-search",
                            "type": "tool_call",
                        }
                    ],
                )
            )
        return RunnableLambda(
            lambda _: AIMessage(content="기존 문서 답변")
        )


class PolicyAgentTest(unittest.TestCase):
    def setUp(self):
        self.model = RecordingToolModel()
        self.tool = FakeSearchTool()
        self.agent = PolicyAgent(self.model, [self.tool])
        self.request = {
            "user_profile": {"age": 25, "region": "서울특별시"},
            "documents": [],
            "messages": [HumanMessage(content="내가 받을 수 있어?")],
            "conversation_summary": "",
            "last_retrieval_query": "청년 주거 정책",
        }

    def test_binds_named_required_and_auto_tool_modes(self):
        self.assertEqual(
            [
                call["tool_choice"]
                for call in self.model.bind_calls
            ],
            ["search_policies", "auto"],
        )
        self.assertTrue(
            all(
                call["parallel_tool_calls"] is None
                for call in self.model.bind_calls
            )
        )

    def test_required_mode_forces_search_tool(self):
        response = self.agent.invoke(
            self.request,
            retrieval_mode="required",
        )

        self.assertEqual(
            response.tool_calls[0]["name"],
            "search_policies",
        )
        self.assertEqual(
            response.tool_calls[0]["args"]["query"],
            "청년 주거비 지원 정책",
        )

    def test_prompt_separates_semantic_query_from_profile_filter(self):
        prompt_messages = self.agent.build_prompt_messages(
            self.request
        )
        system_prompt = prompt_messages[0].text

        self.assertIn(
            "사용자 프로필에 저장된 값은 query에 넣지",
            system_prompt,
        )
        self.assertIn(
            "metadata filter로 별도 적용",
            system_prompt,
        )
        self.assertIn(
            '좋은 query: "월세 주거비 지원"',
            system_prompt,
        )

    def test_optional_mode_can_answer_without_tool(self):
        response = self.agent.invoke(
            self.request,
            retrieval_mode="optional",
        )

        self.assertEqual(response.text, "기존 문서 답변")
        self.assertFalse(response.tool_calls)

    def test_disabled_mode_uses_plain_model(self):
        response = self.agent.invoke(
            self.request,
            retrieval_mode="disabled",
        )

        self.assertEqual(response.text, "최종 답변")
        self.assertFalse(response.tool_calls)

    def test_unknown_retrieval_mode_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "retrieval mode",
        ):
            self.agent.invoke(
                self.request,
                retrieval_mode="unknown",
            )


if __name__ == "__main__":
    unittest.main()
