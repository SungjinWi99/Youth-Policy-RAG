import json
from datetime import date
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from src.user.models import UserProfile


SYSTEM_PROMPT = """
당신은 대한민국 청년정책을 안내하는 RAG 기반 정책 상담 어시스턴트입니다.

역할:
- 사용자의 나이, 성별, 소득수준, 주거지, 직업 등 프로필 조건을 고려해 관련 청년정책을 설명합니다.
- 제공된 검색 문서(context)에 근거해서만 답변합니다.
- 문서에 없는 내용은 추측하지 말고 "제공된 자료만으로는 확인하기 어렵습니다"라고 말합니다.
- 정책 조건이 사용자에게 맞는지 가능한 범위에서 판단하되, 최종 신청 가능 여부는 공식 신청 페이지나 담당 기관 확인이 필요하다고 안내합니다.

답변 원칙:
1. 먼저 사용자의 질문 의도를 한 문장으로 파악합니다.
2. 관련 정책이 있으면 정책명, 지원 내용, 대상 조건, 신청 기간 또는 사업 종료일, 신청 방법 또는 참고 URL을 정리합니다.
3. 사용자의 프로필과 정책 조건을 비교해 "적합해 보이는 이유"와 "추가 확인이 필요한 조건"을 구분합니다.
4. 여러 정책이 있으면 우선순위를 매겨 2~4개 정도만 추천합니다.
5. 검색된 문서에 마감된 정책이 포함되어 있다면, 마감된 정책임을 명확히 표시하고 현재 신청 가능 여부 확인이 필요하다고 안내합니다.
6. 정책 정보가 부족하면 부족한 필드를 명확히 말합니다.
7. 답변은 한국어로, 친절하지만 과장 없이 작성합니다.

출력 형식:
- 질문 요약
- 추천 정책
- 사용자 조건과의 적합성
- 추가 확인 필요 사항
- 다음 행동 제안

사용자 프로필:
{user_profile}

검색된 정책 문서:
{context}
""".strip()

HUMAN_PROMPT = """
사용자 질문:
{question}

위 사용자 프로필과 검색된 정책 문서를 바탕으로 답변해주세요.
""".strip()


def today_yyyymmdd() -> int:
  return int(date.today().strftime("%Y%m%d"))

def normalize_user_gender(gender: str | None) -> str | None:
  if not gender:
    return None

  value = gender.strip().lower()
  if value in {"female", "woman", "women", "f", "여성", "여자"}:
    return "female"
  if value in {"male", "man", "men", "m", "남성", "남자"}:
    return "male"
  return None

def format_optional(value) -> str:
  return str(value) if value not in (None, "") else "미입력"

def format_context_value(value) -> str:
  return str(value) if value not in (None, "") else "미제공"

def format_context_unit_value(value, unit: str) -> str:
  formatted_value = format_context_value(value)
  return formatted_value if formatted_value == "미제공" else f"{formatted_value}{unit}"

def format_context_range(start, end, unit: str = "") -> str:
  if start in (None, "") and end in (None, ""):
    return "미제공"
  return f"{format_context_unit_value(start, unit)} ~ {format_context_unit_value(end, unit)}"

def format_income_condition(metadata: dict) -> str:
  if metadata.get("incomePolicy") == "all":
    return "제한 없음"
  return format_context_range(metadata.get("earnMinAmt"), metadata.get("earnMaxAmt"))

def format_user_profile(user: UserProfile) -> str:
  return "\n".join([
      f"나이: {format_optional(user.age)}",
      f"성별: {format_optional(user.gender)}",
      f"소득수준: {format_optional(user.income)}",
      f"주거지: {format_optional(user.region)}",
      f"직업: {format_optional(user.job)}",
  ])


class RAGResult(BaseModel):
  answer: str
  contexts: list[str]
  retrieved_policy_ids: list[str]


class RAGPipeline:
  def __init__(self, llm, vector_store, search_k):
    self.llm = llm
    self.vector_store = vector_store
    self.search_k = search_k
    self.prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
    ])
    self.generation_chain = self.prompt | self.llm | StrOutputParser()

  def format_doc(self, doc, index: int) -> str:
    metadata = doc.metadata
    return f"""
[검색 결과 {index}]
정책번호: {format_context_value(metadata.get("plcyNo"))}
대분류: {format_context_value(metadata.get("lclsfNm"))}
중분류: {format_context_value(metadata.get("mclsfNm"))}
주관기관: {format_context_value(metadata.get("sprvsnInstCdNm"))}
운영기관: {format_context_value(metadata.get("operInstCdNm"))}
지역: {format_context_value(metadata.get("region"))}
지원 연령: {format_context_range(metadata.get("sprtTrgtMinAge"), metadata.get("sprtTrgtMaxAge"), "세")}
소득 조건: {format_income_condition(metadata)}
사업 기간: {format_context_range(metadata.get("bizPrdBgngYmd"), metadata.get("bizPrdEndYmd"))}
사업 기간 기타 설명: {format_context_value(metadata.get("bizPrdEtcCn"))}
신청 기간: {format_context_value(metadata.get("aplyYmd"))}
신청 방법: {format_context_value(metadata.get("plcyAplyMthdCn"))}
신청 URL: {format_context_value(metadata.get("aplyUrlAddr"))}
참고 URL 1: {format_context_value(metadata.get("refUrlAddr1"))}
참고 URL 2: {format_context_value(metadata.get("refUrlAddr2"))}
참여 대상: {format_context_value(metadata.get("ptcpPrpTrgtCn"))}
추가 신청 자격: {format_context_value(metadata.get("addAplyQlfcCndCn"))}
제출 서류: {format_context_value(metadata.get("sbmsnDcmntCn"))}
심사 방법: {format_context_value(metadata.get("srngMthdCn"))}
직업 코드: {format_context_value(metadata.get("jobCd"))}
혼인 상태 코드: {format_context_value(metadata.get("mrgSttsCd"))}

검색 문서:
{doc.page_content}
""".strip()

  def format_docs(self, docs) -> str:
    return "\n\n---\n\n".join(
      self.format_doc(doc, index)
      for index, doc in enumerate(docs, start=1)
    )

  def build_user_filter(self, user: UserProfile, exclude_expired: bool = True) -> dict | None:
    filters = []

    if user.age is not None:
      filters.extend([
        {"sprtTrgtMinAge": {"$lte": user.age}},
        {"sprtTrgtMaxAge": {"$gte": user.age}},
      ])

    user_gender = normalize_user_gender(user.gender)
    if user_gender:
      filters.append({"genderPolicy": {"$in": ["all", user_gender]}})

    if user.income is not None:
      filters.append(
        {
          "$or": [
            {"incomePolicy": {"$eq": "all"}},
            {
              "$and": [
                {"earnMinAmt": {"$lte": user.income}},
                {"earnMaxAmt": {"$gte": user.income}},
              ]
            },
          ]
        }
      )

    if user.region:
      filters.append({"regionPolicy": {"$in": ["all", user.region]}})

    if exclude_expired:
      filters.append({"bizPrdEndYmd": {"$gte": today_yyyymmdd()}})

    if not filters:
      return None

    if len(filters) == 1:
        return filters[0]

    return {"$and": filters}

  def _build_retriever(self, metadata_filter: dict | None = None):
    search_kwargs = {"k": self.search_k}
    if metadata_filter:
      search_kwargs["filter"] = metadata_filter

    return self.vector_store.as_retriever(search_kwargs=search_kwargs)

  def _build_chain_input(
      self,
      user_input: str,
      user_profile: UserProfile,
      documents,
  ) -> dict:
    return {
      "question": user_input,
      "user_profile": format_user_profile(user_profile),
      "context": self.format_docs(documents),
    }

  def _build_result(self, answer: str, documents) -> RAGResult:
    return RAGResult(
      answer=answer,
      contexts=[
        self.format_doc(document, index)
        for index, document in enumerate(documents, start=1)
      ],
      retrieved_policy_ids=[
        document.metadata["plcyNo"]
        for document in documents
        if document.metadata.get("plcyNo")
      ],
    )

  def retrieve_documents(
      self,
      user_input: str,
      user_profile: UserProfile,
      exclude_expired: bool = True,
  ):
    metadata_filter = self.build_user_filter(
      user=user_profile,
      exclude_expired=exclude_expired,
    )
    retriever = self._build_retriever(metadata_filter)
    return retriever.invoke(user_input)

  async def aretrieve_documents(
      self,
      user_input: str,
      user_profile: UserProfile,
      exclude_expired: bool = True,
  ):
    metadata_filter = self.build_user_filter(
      user=user_profile,
      exclude_expired=exclude_expired,
    )
    retriever = self._build_retriever(metadata_filter)
    return await retriever.ainvoke(user_input)

  def generate_answer(
      self,
      user_input: str,
      user_profile: UserProfile,
      exclude_expired: bool = True,
  ) -> RAGResult:
    documents = self.retrieve_documents(
      user_input=user_input,
      user_profile=user_profile,
      exclude_expired=exclude_expired,
    )
    chain_input = self._build_chain_input(
      user_input=user_input,
      user_profile=user_profile,
      documents=documents,
    )
    answer = self.generation_chain.invoke(chain_input)
    return self._build_result(answer, documents)

  async def agenerate_answer(
      self,
      user_input: str,
      user_profile: UserProfile,
      exclude_expired: bool = True,
  ) -> RAGResult:
    documents = await self.aretrieve_documents(
      user_input=user_input,
      user_profile=user_profile,
      exclude_expired=exclude_expired,
    )
    chain_input = self._build_chain_input(
      user_input=user_input,
      user_profile=user_profile,
      documents=documents,
    )
    answer = await self.generation_chain.ainvoke(chain_input)
    return self._build_result(answer, documents)

  async def stream_answer(
      self,
      user_input: str,
      user_profile: UserProfile,
      exclude_expired: bool = True,
  ):
    documents = await self.aretrieve_documents(
      user_input=user_input,
      user_profile=user_profile,
      exclude_expired=exclude_expired,
    )
    result_metadata = self._build_result("", documents)
    metadata_event = {
      'type': 'metadata',
      'data': {
        'contexts': result_metadata.contexts,
        'retrieved_policy_ids': result_metadata.retrieved_policy_ids,
      },
    }
    yield f"data: {json.dumps(metadata_event, ensure_ascii=False)}\n\n"

    chain_input = self._build_chain_input(
      user_input=user_input,
      user_profile=user_profile,
      documents=documents,
    )
    async for chunk in self.generation_chain.astream(chain_input):
      yield f"data: {json.dumps({'type': 'chunk', 'data': chunk}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
