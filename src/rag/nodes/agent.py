from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from src.rag.prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT
from src.rag.prompts import ANSWER_STRATEGY_INSTRUCTIONS
from src.rag.state import RAGUserProfile
from src.rag.utils.formatting import format_user_profile, format_docs

class PolicyAgent:
  def __init__(self, llm: BaseChatModel):
    self.llm = llm
    self.prompt = ChatPromptTemplate.from_messages([
       ("system", AGENT_SYSTEM_PROMPT),
       MessagesPlaceholder("chat_history", optional=True),
       ("human", AGENT_USER_PROMPT)
    ])
    self.chain = self.prompt | self.llm | StrOutputParser()

  def build_chain_input(self,
                  user_input: str,
                  user_profile: RAGUserProfile,
                  documents: list[Document],
                  chat_history: list[BaseMessage],
                  answer_strategy: str = "policy_recommendation",
                  ):
    answer_strategy_instruction = ANSWER_STRATEGY_INSTRUCTIONS.get(
      answer_strategy,
      ANSWER_STRATEGY_INSTRUCTIONS["policy_recommendation"]
    )
    return {
      "user_input": user_input,
      "user_profile": format_user_profile(user_profile),
      "documents": format_docs(documents) if documents else [],
      "chat_history": list(chat_history or []),
      "answer_strategy": answer_strategy,
      "answer_strategy_instruction": answer_strategy_instruction,
    }
  
  def invoke(self,
             user_input: str,
             user_profile: RAGUserProfile,
             documents: list[Document] | None = None,
             chat_history: list[BaseMessage] | None = None,
             answer_strategy: str = "policy_recommendation",
             ):
    chain_input = self.build_chain_input(user_input = user_input,
                                         user_profile = user_profile,
                                         documents = documents,
                                         chat_history = chat_history,
                                         answer_strategy = answer_strategy,
    )
    return self.chain.invoke(chain_input)
  
  async def ainvoke(self,
             user_input: str,
             user_profile: RAGUserProfile,
             documents: list[Document] | None = None,
             chat_history: list[BaseMessage] | None = None,
             answer_strategy: str = "policy_recommendation",
             ):
    chain_input = self.build_chain_input(user_input = user_input,
                                         user_profile = user_profile,
                                         documents = documents,
                                         chat_history = chat_history,
                                         answer_strategy = answer_strategy,
    )
    return await self.chain.ainvoke(chain_input)
