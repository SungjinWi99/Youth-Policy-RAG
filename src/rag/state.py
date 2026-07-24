from typing import Annotated, Literal, TypedDict
import operator
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


CHECKER_VERDICT_METADATA_KEY = "_checker_verdict"
CHECKER_REASONING_METADATA_KEY = "_checker_reasoning"


class RAGUserProfile(TypedDict, total=False):
    age: int | None
    gender: str | None
    job: str | None
    income: int | None
    region: str | None


class RAGGraphInput(TypedDict):
    user_input: str
    user_profile: RAGUserProfile
    exclude_expired: bool
    messages: Annotated[list[AnyMessage], add_messages]
    retrieval_count: int
    retrieved_policies: list[Document]
    checked_policies: Annotated[list["CheckedPolicy"], operator.add]


class RAGGraphState(RAGGraphInput, total=False):
    user_requirement: str
    needs_retrieval: bool
    retrieval_reason: str
    retrieval_query: str
    retrieval_count: int

    retrieved_policies: list[Document]
    checked_policies: Annotated[list["CheckedPolicy"], operator.add]
    active_policies: list[Document]
    documents: list[Document]
    answer: str
    selection_reason: str


PolicyVerdict = Literal[
    "direct_fit",
    "fit_needs_clarification",
    "indirect",
    "mismatch",
]


class CheckedPolicy(TypedDict):
    verdict: PolicyVerdict
    document: Document
    reasoning: str
    retrieval_rank: int
    retrieval_round: int

class RAGGraphOutput(TypedDict):
    documents: list[Document]
    answer: str


class RAGResult(BaseModel):
    answer: str
    contexts: list[str]
    retrieved_policy_ids: list[str]
