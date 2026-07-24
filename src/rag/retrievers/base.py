from typing import Protocol
from dataclasses import dataclass
from langchain_core.documents import Document

from src.rag.state import RAGUserProfile

@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    user_profile: RAGUserProfile
    exclude_expired: bool
    excluded_policy_ids: frozenset[str] = frozenset()


class PolicyRetriever(Protocol):
    search_k: int

    def retrieve(self, request: RetrievalRequest) -> list[Document]: ...

    async def aretrieve(self, request: RetrievalRequest) -> list[Document]: ...
