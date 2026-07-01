from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class RAGResult(BaseModel):
    answer: str
    contexts: list[str]
    retrieved_policy_ids: list[str]


class PolicySearchProfile(TypedDict, total=False):
    age: int | None
    gender: str | None
    job: str | None
    income: int | None
    region: str | None


class RAGGraphInput(TypedDict):
    user_input: str
    user_profile: PolicySearchProfile
    exclude_expired: bool
    messages: Annotated[list[AnyMessage], add_messages]


class RAGGraphState(RAGGraphInput, total=False):
    documents: list[Document]
    answer: str
    conversation_summary: str


class RAGGraphOutput(TypedDict):
    documents: list[Document]
    answer: str
