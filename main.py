from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.chat.router import chat_router
from src.user.router import user_router
from src.config import load_config
from src.database import create_db_and_tables
from src.factory import build_rag_graph

load_dotenv()
config = load_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    rag_graph = build_rag_graph(config)
    app.state.rag_graph = rag_graph
    try:
        yield
    finally:
        rag_graph.close()

app = FastAPI(title="청년정책 RAG API", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(user_router)
