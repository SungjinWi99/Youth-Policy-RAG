from fastapi import Request

from src.database import get_session
from src.rag.graph import RAGGraph

def get_rag_graph(request: Request) -> RAGGraph:
    return request.app.state.rag_graph

def get_db():
    yield from get_session()
