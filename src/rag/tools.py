from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from src.rag.retriever import PolicyRetriever


SEARCH_POLICIES_DESCRIPTION = """
사용자 조건에 맞는 청년정책을 새로 검색한다. query에는 정책 주제와 지원 목적만
간결하게 작성하고, 나이·성별·소득·거주 지역 등 사용자 프로필 값은 넣지 않는다.
프로필 조건과 만료 여부는 Tool이 metadata filter로 별도 적용한다.
현재 검색된 정책 문서만으로 답할 수 없거나, 질문 주제가 바뀌었거나,
사용자가 다른 정책·추가 추천·최신 정보를 요청한 경우에만 사용한다.
""".strip()


def create_search_policies_tool(
    retriever: PolicyRetriever,
) -> StructuredTool:
    def build_command(
        query: str,
        runtime: ToolRuntime,
        documents,
    ) -> Command:
        state = runtime.state
        policy_ids = [
            document.metadata["plcyNo"]
            for document in documents
            if document.metadata.get("plcyNo")
        ]
        return Command(
            update={
                "documents": documents,
                "retrieval_mode": "disabled",
                "last_retrieval_query": query,
                "last_retrieval_profile": dict(
                    state["user_profile"]
                ),
                "last_retrieval_exclude_expired": (
                    state["exclude_expired"]
                ),
                "messages": [
                    ToolMessage(
                        content=(
                            "정책 검색 완료. 검색된 정책번호: "
                            f"{policy_ids or '없음'}"
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    def search_policies(
        query: str,
        runtime: ToolRuntime,
    ) -> Command:
        """청년정책을 검색한다."""
        documents = retriever.retrieve(
            query=query,
            user_profile=runtime.state["user_profile"],
            exclude_expired=runtime.state["exclude_expired"],
        )
        return build_command(query, runtime, documents)

    async def asearch_policies(
        query: str,
        runtime: ToolRuntime,
    ) -> Command:
        """청년정책을 비동기로 검색한다."""
        documents = await retriever.aretrieve(
            query=query,
            user_profile=runtime.state["user_profile"],
            exclude_expired=runtime.state["exclude_expired"],
        )
        return build_command(query, runtime, documents)

    return StructuredTool.from_function(
        func=search_policies,
        coroutine=asearch_policies,
        name="search_policies",
        description=SEARCH_POLICIES_DESCRIPTION,
    )
