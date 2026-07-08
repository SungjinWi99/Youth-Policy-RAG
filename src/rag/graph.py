import json
from collections.abc import AsyncIterator
from typing import Literal
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.state import (
    RAGGraphInput,
    RAGGraphOutput,
    RAGGraphState,
    RAGResult,
    RAGUserProfile,
)
from src.rag.nodes.retriever import PolicyRetriever
from src.rag.nodes.agent import PolicyAgent
from src.rag.nodes.router import PolicyRouter
from src.checkpointer import AsyncCompatibleSqliteSaver
from src.rag.utils.formatting import format_doc

class PolicyRagGraph:
  def __init__(self,
               router: PolicyRouter,
               retriever: PolicyRetriever,
               agent: PolicyAgent,
               checkpointer: AsyncCompatibleSqliteSaver
               ):
    self.router = router
    self.retriever = retriever
    self.agent = agent
    self.checkpointer = checkpointer
    self.graph = self._compile_graph()

  def _compile_graph(self):
    workflow = StateGraph(
      state_schema=RAGGraphState,
      input_schema=RAGGraphInput,
      output_schema=RAGGraphOutput
    )

    workflow.add_node(
      "router",
      RunnableLambda(
        self._router_node,
        afunc = self._arouter_node
      )
    )
    workflow.add_node(
      "retriever",
      RunnableLambda(
        self._retrieve_node,
        afunc = self._aretrieve_node
      )
    )
    workflow.add_node(
      "agent",
      RunnableLambda(
        self._agent_node,
        afunc = self._aagent_node
      )
    )

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", self._select_route,
                                   {
                                     "retriever": "retriever",
                                     "agent": "agent"
                                   })
    workflow.add_edge("retriever", "agent")
    workflow.add_edge("agent", END)
    return workflow.compile(
      checkpointer=self.checkpointer,
      name="youth_policy_rag"
    )

  def _build_graph_config(self,
                          user_id: str | None,
                          ) -> dict:
        thread_id = user_id or str(uuid4())
        return {"configurable": {"thread_id": thread_id}}

  def build_graph_input(self,
                        *,
                        user_input: str,
                        user_profile: RAGUserProfile,
                        exclude_expired: bool
  ) -> RAGGraphInput:
    return {
      "user_input": user_input,
      "user_profile": dict(user_profile),
      "exclude_expired": exclude_expired,
      "messages": [HumanMessage(content=user_input)]
    }
  def build_result(self,
                         *,
                         answer: str,
                         documents: list[Document]
  ) -> RAGResult:
    return RAGResult(
      answer = answer,
      contexts=[
        format_doc(document, index)
        for index, document in enumerate(documents, start=1)
      ],
      retrieved_policy_ids=[
        document.metadata['plcyNo']
        for document in documents
        if document.metadata.get('plcyNo')
      ]
    )

  # Router
  def _router_node(self, state: RAGGraphState):
    documents = state.get('documents', [])
    chat_history = state.get('messages', [])[:-1]
    result = self.router.decide(
      current_question=state['user_input'],
      documents = documents,
      chat_history = chat_history
    )
    return {
      "route": result.route,
      "route_reason": result.route_reason
    }

  async def _arouter_node(self, state: RAGGraphState):
    documents = state.get('documents', [])
    chat_history = state.get('messages', [])[:-1]
    result = await self.router.adecide(
      current_question=state['user_input'],
      documents = documents,
      chat_history = chat_history
    )
    return {
      "route": result.route,
      "route_reason": result.route_reason
    }

  def _select_route(self, state:RAGGraphState) -> Literal['retriever', 'agent']:
    return state['route']

  # Retriever
  def _retrieve_node(self, state: RAGGraphState):
    documents = self.retriever.retrieve(query=state['user_input'],
                                   user_profile=state['user_profile'],
                                   exclude_expired = state['exclude_expired'])
    return {"documents": documents}

  async def _aretrieve_node(self, state: RAGGraphState):
    documents = await self.retriever.aretrieve(query=state['user_input'],
                                   user_profile=state['user_profile'],
                                   exclude_expired = state['exclude_expired'])
    return {"documents": documents}

  # Agent
  def _agent_node(self, state: RAGGraphState):
    documents = state.get('documents', [])
    chat_history = state.get('messages', [])[:-1]
    result = self.agent.invoke(user_input = state['user_input'],
                             user_profile = state['user_profile'],
                             documents = documents,
                             chat_history = chat_history)
    return {
      "answer": result,
      "messages": [AIMessage(content=result)]
    }


  async def _aagent_node(self, state: RAGGraphState):
    documents = state.get('documents', [])
    chat_history = state.get('messages', [])[:-1]
    result = await self.agent.ainvoke(user_input = state['user_input'],
                             user_profile = state['user_profile'],
                             documents = documents,
                             chat_history = chat_history)
    return {
      "answer": result,
      "messages": [AIMessage(content=result)]
    }


  def generate_answer(self,
                      user_input: str,
                      user_profile: RAGUserProfile,
                      user_id: str | None = None,
                      exclude_expired: bool = True
  ) -> RAGResult:
    graph_output = self.graph.invoke(
      self.build_graph_input(
        user_input=user_input,
        user_profile=user_profile,
        exclude_expired=exclude_expired
      ),
      config = self._build_graph_config(user_id)
    )
    return self.build_result(
      answer=graph_output['answer'],
      documents=graph_output['documents']
    )

  async def agenerate_answer(self,
                      user_input: str,
                      user_profile: RAGUserProfile,
                      user_id: str | None = None,
                      exclude_expired: bool = True
  ) -> RAGResult:
    graph_output = await self.graph.ainvoke(
      self.build_graph_input(
        user_input=user_input,
        user_profile=user_profile,
        exclude_expired=exclude_expired
      ),
      config = self._build_graph_config(user_id)
    )
    return self.build_result(
      answer=graph_output['answer'],
      documents=graph_output['documents']
    )

  async def stream_answer(self,
                          user_input: str,
                          user_profile: RAGUserProfile,
                          user_id: str | None = None,
                          exclude_expired: bool = True
  ) -> AsyncIterator[str]:
    graph_input = self.build_graph_input(
      user_input=user_input,
      user_profile=user_profile,
      exclude_expired=exclude_expired
    )
    streamed_answer = ""

    config = self._build_graph_config(user_id)
    previous_snapshot = await self.graph.aget_state(config)
    previous_documents = previous_snapshot.values.get("documents", [])
    metadata_sent = False

    async for part in self.graph.astream(
      graph_input,
      config=config,
      stream_mode=["updates", "messages"],
      version="v2"
    ):
      if part['type'] == "updates":
        router_update = part["data"].get("router")
        if (
          router_update
          and router_update.get("route") == "agent"
        ):
          result_metadata = self.build_result(
            answer="",
            documents=previous_documents
          )
          yield self._sse_event(
            event_type="metadata",
            data={
              "contexts": result_metadata.contexts,
              "retrieved_policy_ids": (
                result_metadata.retrieved_policy_ids
              )
            }
          )
          metadata_sent = True

        retrieve_update = part["data"].get('retriever')
        if retrieve_update:
          result_metadata = self.build_result(
            answer="",
            documents=retrieve_update['documents'])
          yield self._sse_event(event_type="metadata",
                                data={
                                   "contexts": result_metadata.contexts,
                                   "retrieved_policy_ids": (
                                      result_metadata.retrieved_policy_ids
                                   )
                                })
          metadata_sent = True
        agent_update = part['data'].get('agent')
        if agent_update and not streamed_answer:
           answer = agent_update['answer']
           streamed_answer = answer
           yield self._sse_event("chunk", answer)

      if part['type'] == 'messages':
         message_chunk, metadata = part['data']
         if (
            metadata.get('langgraph_node') != 'agent'
            or metadata.get('ls_model_type') != 'chat'
         ): continue

         chunk = message_chunk.text
         if not chunk: continue

         streamed_answer += chunk
         yield self._sse_event(event_type="chunk", data=chunk)

    if not metadata_sent:
      snapshot = await self.graph.aget_state(config)
      result_metadata = self.build_result(
        answer="",
        documents=snapshot.values.get("documents", [])
      )
      yield self._sse_event(
        event_type="metadata",
        data={
          "contexts": result_metadata.contexts,
          "retrieved_policy_ids": result_metadata.retrieved_policy_ids
        }
      )

    yield self._sse_event("done")

  @staticmethod
  def _sse_event(
        event_type: str,
        data=None,
    ) -> str:
    event = {"type": event_type}
    if data is not None:
        event["data"] = data
    return (
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    )

  def delete_conversation(self, user_id: str) -> None:
      self.checkpointer.delete_thread(user_id)

  def close(self) -> None:
      close = getattr(self.checkpointer, "close", None)
      if close:
          close()
