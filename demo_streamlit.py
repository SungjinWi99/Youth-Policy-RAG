import json
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("YOUTH_RAG_API_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT = 20


def api_url(path: str) -> str:
    return f"{st.session_state.api_base_url.rstrip('/')}{path}"


def parse_optional_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "")}


def request_json(method: str, path: str, **kwargs: Any) -> tuple[bool, Any]:
    try:
        response = requests.request(
            method,
            api_url(path),
            timeout=DEFAULT_TIMEOUT,
            **kwargs,
        )
    except requests.RequestException as exc:
        return False, {"detail": f"API 연결 실패: {exc}"}

    try:
        data = response.json()
    except ValueError:
        data = response.text

    if response.ok:
        return True, data
    return False, data


def parse_sse_line(raw_line: str) -> dict[str, Any] | None:
    if not raw_line or not raw_line.startswith("data: "):
        return None
    event = json.loads(raw_line.removeprefix("data: "))
    if not isinstance(event, dict):
        raise ValueError("SSE event는 JSON object여야 합니다.")
    return event


def stream_chat(user_id: str, user_input: str, exclude_expired: bool):
    payload = {
        "user_id": user_id,
        "user_input": user_input,
        "exclude_expired": exclude_expired,
    }
    with requests.post(
        api_url("/chat"),
        json=payload,
        stream=True,
        timeout=(5, None),
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            event = parse_sse_line(raw_line)
            if event is None:
                continue

            yield event
            if event.get("type") == "done":
                break


def init_state() -> None:
    st.session_state.setdefault("api_base_url", DEFAULT_API_BASE_URL)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("active_user_id", "")
    st.session_state.setdefault("exclude_expired", True)


def render_user_form() -> None:
    st.subheader("User Profile")

    with st.form("user_profile_form"):
        user_id = st.text_input("user_id", value=st.session_state.active_user_id)
        col1, col2 = st.columns(2)
        with col1:
            age = st.text_input("age", placeholder="예: 25")
            gender = st.selectbox("gender", ["", "여성", "남성"], index=0)
        with col2:
            income = st.text_input("income", placeholder="예: 3000")
            region = st.text_input("region", placeholder="예: 서울특별시")
        job = st.text_input("job", placeholder="선택 입력")

        action = st.radio(
            "action",
            ["register", "update", "load", "delete"],
            horizontal=True,
        )
        submitted = st.form_submit_button("Run")

    if not submitted:
        return

    if not user_id.strip():
        st.error("user_id를 입력하세요.")
        return

    st.session_state.active_user_id = user_id.strip()

    try:
        payload = compact_payload(
            {
                "user_id": user_id.strip(),
                "age": parse_optional_int(age),
                "gender": gender,
                "income": parse_optional_int(income),
                "region": region.strip(),
                "job": job.strip(),
            }
        )
    except ValueError:
        st.error("age와 income은 숫자로 입력하세요.")
        return

    if action == "register":
        ok, data = request_json("POST", "/user/registration", json=payload)
    elif action == "update":
        payload.pop("user_id", None)
        ok, data = request_json("POST", f"/user/{user_id.strip()}", json=payload)
    elif action == "load":
        ok, data = request_json("GET", f"/user/{user_id.strip()}")
    else:
        ok, data = request_json("DELETE", f"/user/{user_id.strip()}")

    if ok:
        st.success("API 요청 성공")
        st.json(data)
    else:
        st.error("API 요청 실패")
        st.json(data)


def render_retrieval_metadata(
    contexts: list[str],
    retrieved_policy_ids: list[str],
) -> None:
    if not contexts and not retrieved_policy_ids:
        return

    with st.expander("검색 근거", expanded=False):
        if retrieved_policy_ids:
            st.caption(f"검색된 정책 ID {len(retrieved_policy_ids)}건")
            st.code("\n".join(retrieved_policy_ids), language=None)

        for index, context in enumerate(contexts, start=1):
            st.markdown(f"**Context {index}**")
            st.text(context)


def render_chat() -> None:
    st.subheader("Streaming Chat")

    st.session_state.exclude_expired = st.checkbox(
        "마감된 정책 제외",
        value=st.session_state.exclude_expired,
        help="끄면 정책 종료일이 지난 문서도 검색 후보에 포함합니다.",
    )

    if st.button("Clear chat"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_retrieval_metadata(
                    message.get("contexts", []),
                    message.get("retrieved_policy_ids", []),
                )

    user_input = st.chat_input("청년정책 질문을 입력하세요")
    if not user_input:
        return

    user_id = st.session_state.active_user_id.strip()
    if not user_id:
        st.error("먼저 User Profile에서 user_id를 입력하거나 조회하세요.")
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = ""
    contexts: list[str] = []
    retrieved_policy_ids: list[str] = []
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        metadata_placeholder = st.empty()
        try:
            for event in stream_chat(
                user_id=user_id,
                user_input=user_input,
                exclude_expired=st.session_state.exclude_expired,
            ):
                event_type = event.get("type")
                if event_type == "metadata":
                    metadata = event.get("data") or {}
                    contexts = metadata.get("contexts") or []
                    retrieved_policy_ids = (
                        metadata.get("retrieved_policy_ids") or []
                    )
                    with metadata_placeholder.container():
                        render_retrieval_metadata(
                            contexts,
                            retrieved_policy_ids,
                        )
                elif event_type == "chunk":
                    answer += event.get("data", "")
                    answer_placeholder.markdown(answer)
                elif event_type == "done":
                    break
        except requests.HTTPError as exc:
            answer = f"API 오류: {exc.response.status_code} {exc.response.text}"
            answer_placeholder.error(answer)
        except requests.RequestException as exc:
            answer = f"API 연결 실패: {exc}"
            answer_placeholder.error(answer)
        except (json.JSONDecodeError, ValueError) as exc:
            answer = f"SSE 응답 파싱 실패: {exc}"
            answer_placeholder.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "contexts": contexts,
            "retrieved_policy_ids": retrieved_policy_ids,
        }
    )


def main() -> None:
    st.set_page_config(page_title="청년정책 RAG API Demo", layout="wide")
    init_state()

    st.title("청년정책 RAG API Demo")
    st.caption("FastAPI 서버를 먼저 실행한 뒤 이 앱에서 유저 프로필과 스트리밍 채팅을 테스트합니다.")

    with st.sidebar:
        st.header("API")
        st.session_state.api_base_url = st.text_input(
            "Base URL",
            value=st.session_state.api_base_url,
        )
        st.code(f"{st.session_state.api_base_url.rstrip('/')}/docs")

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        render_user_form()
    with right:
        render_chat()


if __name__ == "__main__":
    main()
