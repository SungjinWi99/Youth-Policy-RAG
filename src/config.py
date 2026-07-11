from typing import Literal
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAME = "config.yaml"
class DataConfig(BaseModel):
  raw: str
  sqlite_db: str
  conversation_db: str
  chroma_dir: str
  chroma_collection_name: str

class RetrieverConfig(BaseModel):
  provider: Literal['google', 'openai', 'upstage', 'ollama']
  query_model: str
  passage_model: str
  search_k: int

class LLMConfig(BaseModel):
  provider: Literal['google', 'openai', 'upstage', 'anthropic', 'deepseek']
  model: str

class EvaluationConfig(BaseModel):
  example_path: str
  provider: Literal['google', 'openai', 'upstage', 'anthropic']
  model: str
  dataset_name: str
  experiment_prefix: str
  max_concurrency: int = Field(default=1, ge=1)

class DatabaseConfig(BaseModel):
  echo: bool = False

class PlannerRuntimeConfig(BaseModel):
  history_window: int = Field(default=6, ge=0)

class AgentRuntimeConfig(BaseModel):
  history_window: int = Field(default=10, ge=0)

class RAGRuntimeConfig(BaseModel):
  planner: PlannerRuntimeConfig = Field(default_factory=PlannerRuntimeConfig)
  agent: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)

class ApplicationConfig(BaseModel):
  release: str = Field(min_length=1, max_length=200)
  environment: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")

class AppConfig(BaseModel):
  app: ApplicationConfig
  data: DataConfig
  retriever: RetrieverConfig
  llm: LLMConfig
  evaluation: EvaluationConfig
  database: DatabaseConfig
  rag: RAGRuntimeConfig = Field(default_factory=RAGRuntimeConfig)

  def path(self, value: str) -> str:
      return str((PROJECT_ROOT / value).resolve())

@lru_cache
def load_config() -> AppConfig:
  with open(PROJECT_ROOT / CONFIG_FILENAME, encoding="utf-8") as f:
    return AppConfig.model_validate(yaml.safe_load(f))
