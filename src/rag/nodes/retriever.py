from langchain_core.runnables import RunnableLambda
from src.rag.retrievers import PolicyRetriever, RetrievalRequest
from src.rag.state import RAGGraphState


REJECTED_VERDICTS = frozenset({"indirect", "mismatch"})


def _policy_id(document) -> str:
    return str(document.metadata.get("plcyNo") or "").strip()


def _excluded_policy_ids(state: RAGGraphState) -> frozenset[str]:
    active_policy_ids = {
        policy_id
        for document in state.get("active_policies", [])
        if (policy_id := _policy_id(document))
    }
    rejected_policy_ids = {
        policy_id
        for item in state.get("checked_policies", [])
        if item["verdict"] in REJECTED_VERDICTS
        if (policy_id := _policy_id(item["document"]))
    }
    return frozenset(active_policy_ids | rejected_policy_ids)


def make_retriever_node(retriever: PolicyRetriever):
    def build_request(state: RAGGraphState) -> RetrievalRequest:
        return RetrievalRequest(
            query=state.get("retrieval_query") or state["user_input"],
            user_profile=state["user_profile"],
            exclude_expired=state["exclude_expired"],
            excluded_policy_ids=_excluded_policy_ids(state),
        )

    def build_update(state: RAGGraphState, documents) -> dict:
        return {
            "retrieved_policies": list(documents),
            "retrieval_count": state.get("retrieval_count", 0) + 1,
        }

    def retriever_node(state: RAGGraphState):
        documents = retriever.retrieve(build_request(state))
        return build_update(state, documents)

    async def aretriever_node(state: RAGGraphState):
        documents = await retriever.aretrieve(build_request(state))
        return build_update(state, documents)

    return RunnableLambda(retriever_node, afunc=aretriever_node)
