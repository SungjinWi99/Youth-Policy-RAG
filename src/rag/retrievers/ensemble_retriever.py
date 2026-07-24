import math
from collections import defaultdict
from collections.abc import Sequence
from langchain_core.documents import Document

from src.rag.retrievers.base import PolicyRetriever, RetrievalRequest

class EnsemblePolicyRetriever:
    def __init__(
        self,
        retrievers: Sequence[PolicyRetriever],
        weights: Sequence[float],
        search_k: int,
        rrf_k: int,
    ):
        if not retrievers:
            raise ValueError("하나 이상의 retriever가 필요합니다.")
        if len(retrievers) != len(weights):
            raise ValueError("retrievers와 weights의 길이가 같아야 합니다.")
        if any(weight < 0 for weight in weights):
            raise ValueError("weight는 0 이상이어야 합니다.")
        if not any(weight > 0 for weight in weights):
            raise ValueError("하나 이상의 weight가 0보다 커야 합니다.")
        if search_k < 1:
            raise ValueError("search_k는 1 이상이어야 합니다.")
        if rrf_k < 1:
            raise ValueError("rrf_k는 1 이상이어야 합니다.")

        self.retrievers = list(retrievers)
        self.weights = [float(weight) for weight in weights]
        self.search_k = search_k
        self.rrf_k = rrf_k

    def fuse_results(
        self,
        source_results: Sequence[Sequence[Document]],
    ) -> list[Document]:
        if len(source_results) != len(self.retrievers):
            raise ValueError("source_results와 retrievers의 길이가 같아야 합니다.")

        scores: dict[str, float] = defaultdict(float)
        documents_by_id: dict[str, Document] = {}
        source_ranks: dict[str, list[float]] = {}
        first_seen: dict[str, int] = {}
        seen_order = 0

        for source_index, (retrieved_documents, weight) in enumerate(zip(
            source_results,
            self.weights,
            strict=True,
        )):
            seen_in_source = set()
            for rank, document in enumerate(retrieved_documents, start=1):
                policy_id = str(document.metadata.get("plcyNo") or "").strip()
                if not policy_id:
                    raise ValueError("Ensemble 문서에 plcyNo 메타데이터가 필요합니다.")
                if policy_id in seen_in_source:
                    continue
                seen_in_source.add(policy_id)
                documents_by_id.setdefault(policy_id, document)
                source_ranks.setdefault(
                    policy_id,
                    [math.inf] * len(source_results),
                )[source_index] = rank
                if policy_id not in first_seen:
                    first_seen[policy_id] = seen_order
                    seen_order += 1
                scores[policy_id] += weight / (self.rrf_k + rank)

        ranked_ids = sorted(
            scores,
            key=lambda policy_id: (
                -scores[policy_id],
                *source_ranks[policy_id],
                first_seen[policy_id],
                policy_id,
            ),
        )[:self.search_k]
        return [documents_by_id[policy_id] for policy_id in ranked_ids]

    def retrieve_with_sources(
        self,
        request: RetrievalRequest,
    ) -> tuple[list[Document], list[list[Document]]]:
        source_results = [
            [
                document
                for document in retriever.retrieve(request)
                if str(
                    document.metadata.get("plcyNo") or ""
                ).strip() not in request.excluded_policy_ids
            ]
            for retriever in self.retrievers
        ]
        return self.fuse_results(source_results), source_results

    def retrieve(self, request: RetrievalRequest) -> list[Document]:
        documents, _ = self.retrieve_with_sources(request)
        return documents

    async def aretrieve_with_sources(
        self,
        request: RetrievalRequest,
    ) -> tuple[list[Document], list[list[Document]]]:
        import asyncio

        source_results = await asyncio.gather(*[
            retriever.aretrieve(request)
            for retriever in self.retrievers
        ])
        source_results = [
            [
                document
                for document in documents
                if str(
                    document.metadata.get("plcyNo") or ""
                ).strip() not in request.excluded_policy_ids
            ]
            for documents in source_results
        ]
        return self.fuse_results(source_results), source_results

    async def aretrieve(self, request: RetrievalRequest) -> list[Document]:
        documents, _ = await self.aretrieve_with_sources(request)
        return documents
