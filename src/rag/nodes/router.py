from collections.abc import Sequence
from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.rag.utils.formatting import format_docs
from src.rag.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT


class RouterOutput(BaseModel):
  route: Literal['retriever', 'agent'] = Field(
     description=(
        "다음 실행 분기."
        "'agent'는 현재 활성 정책 문서만으로 현재 질문에 답변이 가능한 경우"
        "'retriever'는 새 정책 검색이 필요하거나 현재 활성 문서가 부족한 경우."
        )
  )
  route_reason: str = Field(
     description=(
        "분기 결정 이유. 현재 사용자 질문과 현재 활성 정책 문서의 관계를 기준으로"
        "1문장으로 간결하게 작성"
     )
  )


class PolicyRouter:
  def __init__(self, llm: BaseChatModel):
    self.llm = llm.with_structured_output(RouterOutput)
    self.prompt = ChatPromptTemplate.from_messages([
       ("system", ROUTER_SYSTEM_PROMPT),
       MessagesPlaceholder("chat_history", optional=True),
       ("human", ROUTER_USER_PROMPT)
    ])
    self.chain = self.prompt | self.llm


  def _build_chain_input(self,
                         *,
                         current_question: str,
                         documents: Sequence[Document] | None = None,
                         chat_history: Sequence[BaseMessage] | None = None
                         ) -> dict:
     return {
        "documents": format_docs(documents) if documents else [],
        "current_question": current_question,
        "chat_history": list(chat_history or [])
     }


  def decide(self,
               *,
               current_question: str,
               documents: Sequence[Document],
               chat_history: Sequence[BaseMessage] | None = None
               ) -> RouterOutput:

      if not documents:
         return RouterOutput(
            route="retriever",
            route_reason="documents가 비어있습니다: 강제 Retrieval 진행"
         )
      chain_input = self._build_chain_input(
       current_question = current_question,
       documents = documents,
       chat_history = chat_history
     )
      return self.chain.invoke(chain_input)

  async def adecide(self,
                *,
                current_question: str,
                documents: Sequence[Document],
                chat_history: Sequence[BaseMessage] | None = None
                ) -> RouterOutput:
    if not documents:
         return RouterOutput(
            route="retriever",
            route_reason="documents가 비어있습니다: 강제 Retrieval 진행"
         )

    chain_input = self._build_chain_input(
       current_question = current_question,
       documents = documents,
       chat_history = chat_history
     )
    return await self.chain.ainvoke(chain_input)
