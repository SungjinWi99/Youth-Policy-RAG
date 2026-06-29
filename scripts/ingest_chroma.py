import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import load_config
from src.factory import create_embedding_model

BATCH_SIZE = 270
TIME_SLEEP = 10
load_dotenv()
config = load_config()

def to_int(value, default=0):
  try:
    value = str(value or "").strip()
    return int(value) if value else default
  except ValueError:
    return default

def to_policy_end_date(value):
  value = str(value or "").strip()
  if len(value) == 8 and value.isdigit():
    return int(value)
  return 99991231

def normalize_income_policy(item):
  min_income = to_int(item.get("earnMinAmt"), 0)
  max_income = to_int(item.get("earnMaxAmt"), 0)
  if min_income == 0 and max_income == 0:
    return "all"
  return "specific"

def normalize_region_policy(item):
  zip_codes = [
      zip_code.strip()
      for zip_code in str(item.get("zipCd") or "").split(",")
      if zip_code.strip()
  ]

  if not zip_codes or len(zip_codes) >= 100:
    return "all"
  return item.get("rgtrInstCdNm") or "all"

def normalize_gender_policy(item):
  text = " ".join([
      str(item.get("plcyNm") or ""),
      str(item.get("plcyKywdNm") or ""),
      str(item.get("plcyExplnCn") or ""),
      str(item.get("plcySprtCn") or ""),
      str(item.get("addAplyQlfcCndCn") or ""),
      str(item.get("ptcpPrpTrgtCn") or ""),
  ])

  female_terms = ["여성", "여자", "여대생", "경력단절여성"]
  male_terms = ["남성", "남자", "남대생"]

  has_female = any(term in text for term in female_terms)
  has_male = any(term in text for term in male_terms)

  if has_female and not has_male:
    return "female"
  if has_male and not has_female:
    return "male"
  return "all"

def main():
  with open(config.data.raw, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

  assert isinstance(raw_data, list)

  documents = []
  ids = []
  for item in raw_data:
      content = f"""
        정책명: {item.get('plcyNm', '')}
        키워드: {item.get('plcyKywdNm', '')}
        카테고리: {item.get('lclsfNm', '')} > {item.get('mclsfNm', '')}
        정책 설명: {item.get('plcyExplnCn', '')}
        지원 내용: {item.get('plcySprtCn', '')}
        """

      metadata = {
          "plcyNo": item.get("plcyNo"),  # 정책 고유 번호
          "lclsfNm": item.get("lclsfNm"),  # 정책 대분류명
          "mclsfNm": item.get("mclsfNm"),  # 정책 중분류명
          "refUrlAddr1": item.get("refUrlAddr1"),  # 정책 참고 URL 1
          "refUrlAddr2": item.get("refUrlAddr2") or "",  # 정책 참고 URL 2
          "aplyUrlAddr": item.get("aplyUrlAddr") or "",  # 정책 신청 URL
          "sprvsnInstCdNm": item.get("sprvsnInstCdNm"),  # 정책 주관 기관명
          "operInstCdNm": item.get("operInstCdNm") or "",  # 정책 운영 기관명
          "sprtTrgtMinAge": to_int(item.get("sprtTrgtMinAge"), 0),  # 지원 대상 최소 연령
          "sprtTrgtMaxAge": to_int(item.get("sprtTrgtMaxAge"), 99),  # 지원 대상 최대 연령
          "earnMinAmt": to_int(item.get("earnMinAmt"), 0),  # 소득 조건 최소 금액
          "earnMaxAmt": to_int(item.get("earnMaxAmt"), 0),  # 소득 조건 최대 금액
          "incomePolicy": normalize_income_policy(item),  # 소득 필터용 정규화 값
          "bizPrdBgngYmd": item.get("bizPrdBgngYmd") or "",  # 사업 기간 시작일
          "bizPrdEndYmd": to_policy_end_date(item.get("bizPrdEndYmd")),  # 사업 기간 종료일
          "bizPrdEtcCn": item.get("bizPrdEtcCn") or "",  # 사업 기간 기타 설명
          "aplyYmd": item.get("aplyYmd") or "",  # 신청 기간
          "plcyAplyMthdCn": item.get("plcyAplyMthdCn") or "",  # 정책 신청 방법
          "ptcpPrpTrgtCn": item.get("ptcpPrpTrgtCn") or "",  # 참여 제안 대상 설명
          "addAplyQlfcCndCn": item.get("addAplyQlfcCndCn") or "",  # 추가 신청 자격 조건
          "sbmsnDcmntCn": item.get("sbmsnDcmntCn") or "",  # 제출 서류
          "srngMthdCn": item.get("srngMthdCn") or "",  # 심사 방법
          "region": item.get("rgtrInstCdNm") or "",  # 정책 등록 기관명
          "regionPolicy": normalize_region_policy(item),  # 지역 필터용 정규화 값
          "zipCd": item.get("zipCd") or "",  # 정책 적용 지역 우편번호/행정구역 코드 목록
          "jobCd": item.get("jobCd") or "",  # 참여 대상 직업 코드
          "genderPolicy": normalize_gender_policy(item),  # 성별 필터용 정규화 값
          "mrgSttsCd": item.get("mrgSttsCd") or "",  # 혼인 상태 코드
      }

      doc = Document(page_content=content.strip(), metadata=metadata)
      documents.append(doc)
      ids.append(item.get("plcyNo"))

  embedding_model = create_embedding_model(
      provider=config.retriever.provider,
      model_name=config.retriever.passage_model,
  )
  vector_store = Chroma(
        collection_name=config.data.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=config.path(config.data.chroma_dir)
    )
  for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="데이터 적재 중..."):
    batch_docs = documents[i : i + BATCH_SIZE]
    batch_ids = ids[i : i + BATCH_SIZE]
    vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    if i + BATCH_SIZE < len(documents):
      time.sleep(TIME_SLEEP)

if __name__ == '__main__':
  main()
