from fastapi import Request
from src.chat.rag import RAGPipeline
from src.database import get_session

def get_rag_service(request: Request) -> RAGPipeline:
    return request.app.state.rag_service

def get_db():
    yield from get_session()
