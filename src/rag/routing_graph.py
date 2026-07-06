from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from src.rag.router import ContextRouter, RouteName, RoutingDecision
from src.rag.state import (
    RoutingGraphInput,
    RoutingGraphOutput,
    RoutingGraphState,
)


RouteHandler = Callable[[RoutingGraphState], dict[str, Any]]


class RoutingGraph:
    """검색 구현과 분리된 대화 라우팅 그래프."""

    def __init__(
        self,
        *,
        router: ContextRouter,
        reuse_handler: RouteHandler,
        search_handler: RouteHandler,
        clarify_handler: RouteHandler,
    ):
        self.router = router
        self.handlers = {
            "reuse": reuse_handler,
            "search": search_handler,
            "clarify": clarify_handler,
        }
        self.graph = self._compile()

    def _compile(self):
        workflow = StateGraph(
            state_schema=RoutingGraphState,
            input_schema=RoutingGraphInput,
            output_schema=RoutingGraphOutput,
        )
        workflow.add_node("route", self._route_node)
        for route, handler in self.handlers.items():
            workflow.add_node(route, handler)

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges(
            "route",
            self._select_route,
            {
                "reuse": "reuse",
                "search": "search",
                "clarify": "clarify",
            },
        )
        for route in self.handlers:
            workflow.add_edge(route, END)

        return workflow.compile(name="youth_policy_routing")

    def _route_node(
        self,
        state: RoutingGraphState,
    ) -> dict[str, RoutingDecision]:
        documents = state["documents"]
        if not documents:
            decision = RoutingDecision(
                route="search",
                reason="활성 문서가 없어 정책 검색이 필요합니다.",
            )
        else:
            decision = self.router.decide(
                current_question=state["current_question"],
                documents=documents,
            )

        if not isinstance(decision, RoutingDecision):
            raise TypeError(
                "router.decide()는 RoutingDecision을 반환해야 합니다."
            )

        return {"routing_decision": decision}

    def _select_route(
        self,
        state: RoutingGraphState,
    ) -> RouteName:
        return state["routing_decision"].route

    def invoke(
        self,
        *,
        current_question: str,
        documents: list[Document],
    ) -> RoutingGraphOutput:
        return self.graph.invoke({
            "current_question": current_question,
            "documents": documents,
        })
