"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ApiError, apiJson } from "@/lib/api";
import type {
  ChatMessage,
  ConversationSnapshot,
  PolicyDetail,
  Profile,
  SessionStatus,
} from "@/lib/types";
import { AnswerFeedback } from "@/components/answer-feedback";
import { Modal } from "@/components/modal";
import { PolicyCard } from "@/components/policy-card";
import { ProfileForm } from "@/components/profile-form";

const SUGGESTED_QUESTIONS = [
  "지금 신청할 수 있는 취업 지원 정책을 알려주세요.",
  "청년 월세 지원 조건이 궁금해요.",
  "창업 준비 중인데 받을 수 있는 지원이 있나요?",
];

type AppState = "loading" | "onboarding" | "ready";

type SseEvent = {
  type: "metadata" | "chunk" | "done";
  data?: unknown;
};

function messageId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function formatProfile(profile: Profile) {
  const items = [
    profile.age != null ? `${profile.age}세` : null,
    profile.region,
    profile.job,
    profile.income != null ? `연 ${profile.income.toLocaleString()}만원` : null,
  ].filter(Boolean);
  return items.length ? items.join(" · ") : "입력된 조건 없음";
}

function dataLine(block: string) {
  return block
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.replace(/^data:\s?/, ""))
    .join("\n");
}

export function ChatClient() {
  const searchParams = useSearchParams();
  const [appState, setAppState] = useState<AppState>("loading");
  const [profile, setProfile] = useState<Profile | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activePolicies, setActivePolicies] = useState<PolicyDetail[]>([]);
  const [draft, setDraft] = useState(searchParams.get("question") ?? "");
  const [excludeExpired, setExcludeExpired] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [pageError, setPageError] = useState("");
  const [policyError, setPolicyError] = useState("");
  const messageEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    async function bootstrap() {
      try {
        const session = await apiJson<SessionStatus>("/sessions/current");
        setProfile(session.profile);
        const snapshot = await apiJson<ConversationSnapshot>(
          "/me/conversation",
        );
        setMessages(
          snapshot.messages.map((message, index) => ({
            ...message,
            id: `restored-${message.role}-${index}`,
          })),
        );
        await loadPolicies(snapshot.active_policy_ids);
        setAppState("ready");
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          setAppState("onboarding");
          return;
        }
        setPageError(
          error instanceof Error
            ? error.message
            : "상담 정보를 불러오지 못했습니다.",
        );
        setAppState("onboarding");
      }
    }

    void bootstrap();
  }, []);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({
      behavior: messages.length > 2 ? "smooth" : "auto",
      block: "nearest",
    });
  }, [messages]);

  async function loadPolicies(policyIds: string[]) {
    setPolicyError("");
    if (!policyIds.length) {
      setActivePolicies([]);
      return;
    }
    try {
      const policies = await apiJson<PolicyDetail[]>("/policies/batch", {
        method: "POST",
        body: JSON.stringify({ policy_ids: policyIds }),
      });
      setActivePolicies(policies);
    } catch (error) {
      setActivePolicies([]);
      setPolicyError(
        error instanceof Error
          ? error.message
          : "정책 정보를 불러오지 못했습니다.",
      );
    }
  }

  async function startSession(
    nextProfile: Profile,
    acceptedStorage: boolean,
  ) {
    setIsSavingProfile(true);
    try {
      const session = await apiJson<SessionStatus>("/sessions/anonymous", {
        method: "POST",
        body: JSON.stringify({
          ...nextProfile,
          accepted_storage: acceptedStorage,
        }),
      });
      setProfile(session.profile);
      setMessages([]);
      setActivePolicies([]);
      setPageError("");
      setAppState("ready");
    } finally {
      setIsSavingProfile(false);
    }
  }

  async function saveProfile(nextProfile: Profile) {
    setIsSavingProfile(true);
    try {
      const updated = await apiJson<Profile>("/me/profile", {
        method: "PATCH",
        body: JSON.stringify(nextProfile),
      });
      setProfile(updated);
      setShowProfile(false);
    } finally {
      setIsSavingProfile(false);
    }
  }

  async function clearConversation() {
    if (
      !window.confirm(
        "현재 상담 대화와 활성 정책을 삭제할까요? 프로필은 유지됩니다.",
      )
    ) {
      return;
    }
    try {
      await apiJson<{ message: string }>("/me/conversation", {
        method: "DELETE",
      });
      setMessages([]);
      setActivePolicies([]);
      setPageError("");
    } catch (error) {
      setPageError(
        error instanceof Error
          ? error.message
          : "대화 기록을 삭제하지 못했습니다.",
      );
    }
  }

  async function deleteAllData() {
    if (
      !window.confirm(
        "프로필과 상담 기록을 모두 삭제할까요? 삭제한 정보는 복구할 수 없습니다.",
      )
    ) {
      return;
    }
    try {
      await apiJson<{ message: string }>("/me/data", { method: "DELETE" });
      setMessages([]);
      setActivePolicies([]);
      setProfile(null);
      setShowProfile(false);
      setAppState("onboarding");
    } catch (error) {
      setPageError(
        error instanceof Error
          ? error.message
          : "저장된 정보를 삭제하지 못했습니다.",
      );
    }
  }

  function handleSseEvent(event: SseEvent, assistantId: string) {
    if (event.type === "metadata") {
      const data = event.data as
        | { retrieved_policy_ids?: string[]; trace_id?: string }
        | undefined;
      if (data?.trace_id) {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? { ...message, traceId: data.trace_id }
              : message,
          ),
        );
      }
      void loadPolicies(data?.retrieved_policy_ids ?? []);
      return;
    }
    if (event.type === "chunk" && typeof event.data === "string") {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: message.content + event.data }
            : message,
        ),
      );
    }
  }

  async function sendMessage(question: string) {
    const normalized = question.trim();
    if (!normalized || isStreaming) {
      return;
    }

    const userMessage: ChatMessage = {
      id: messageId("user"),
      role: "user",
      content: normalized,
    };
    const assistantId = messageId("assistant");
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setDraft("");
    setPageError("");
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/me/chat", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_input: normalized,
          exclude_expired: excludeExpired,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        let detail = "답변을 생성하지 못했습니다.";
        try {
          const data = (await response.json()) as { detail?: string };
          detail = data.detail || detail;
        } catch {
          // Keep fallback message.
        }
        if (response.status === 401) {
          setAppState("onboarding");
        }
        throw new ApiError(detail, response.status);
      }
      if (!response.body) {
        throw new Error("스트리밍 응답을 받을 수 없습니다.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let boundary = buffer.indexOf("\n\n");

        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const data = dataLine(block);
          if (data) {
            handleSseEvent(JSON.parse(data) as SseEvent, assistantId);
          }
          boundary = buffer.indexOf("\n\n");
        }

        if (done) {
          break;
        }
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId && !message.content
              ? {
                  ...message,
                  content: "사용자가 답변 생성을 중지했습니다.",
                }
              : message,
          ),
        );
      } else {
        const message =
          error instanceof Error
            ? error.message
            : "답변을 불러오지 못했습니다.";
        setPageError(`${message} 잠시 후 다시 시도해 주세요.`);
        setMessages((current) =>
          current.map((item) =>
            item.id === assistantId && !item.content
              ? {
                  ...item,
                  content: "답변을 불러오지 못했습니다. 다시 질문해 주세요.",
                }
              : item,
          ),
        );
      }
    } finally {
      abortRef.current = null;
      setIsStreaming(false);
    }
  }

  function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage(draft);
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage(draft);
    }
  }

  if (appState === "loading") {
    return (
      <div className="chat-loading" role="status">
        <span className="loading-spinner" aria-hidden="true" />
        <strong>상담 정보를 불러오고 있습니다</strong>
        <p>잠시만 기다려 주세요.</p>
      </div>
    );
  }

  if (appState === "onboarding") {
    return (
      <section className="onboarding-card">
        <div className="onboarding-heading">
          <p className="eyebrow">상담 준비</p>
          <h1>내게 맞는 정책을 찾기 위한 정보를 알려주세요</h1>
          <p>
            모든 항목은 선택 사항입니다. 비워둔 조건은 상담 중에 추가로
            설명할 수 있습니다.
          </p>
        </div>
        {pageError ? (
          <div className="alert alert-error" role="alert">
            {pageError}
          </div>
        ) : null}
        <ProfileForm
          requireConsent
          busy={isSavingProfile}
          submitLabel="동의하고 상담 시작"
          onSubmit={startSession}
        />
      </section>
    );
  }

  return (
    <>
      <div className="chat-toolbar">
        <div>
          <span className="toolbar-label">내 상담 조건</span>
          <strong>{profile ? formatProfile(profile) : "입력된 조건 없음"}</strong>
        </div>
        <div className="toolbar-actions">
          <button
            className="button button-tertiary button-small"
            type="button"
            onClick={() => setShowProfile(true)}
          >
            조건 수정
          </button>
          <button
            className="button button-tertiary button-small"
            type="button"
            onClick={() => void clearConversation()}
          >
            새 상담
          </button>
        </div>
      </div>

      {pageError ? (
        <div className="alert alert-error" role="alert">
          {pageError}
        </div>
      ) : null}

      <div className="chat-grid">
        <section className="chat-card" aria-label="청년정책 상담">
          <div className="chat-card-header">
            <div>
              <span className="status-dot" aria-hidden="true" />
              <strong>청년정책 상담</strong>
            </div>
            <span>공공데이터 기반 안내</span>
          </div>

          <div className="message-list" aria-live="polite">
            {messages.length === 0 ? (
              <div className="chat-welcome">
                <div className="bot-avatar" aria-hidden="true">
                  상담
                </div>
                <h1>어떤 정책이 필요한지 말씀해 주세요</h1>
                <p>
                  현재 상황과 필요한 지원을 구체적으로 알려주시면 관련 정책을
                  찾아드릴게요.
                </p>
                <div className="suggestion-list">
                  {SUGGESTED_QUESTIONS.map((question) => (
                    <button
                      type="button"
                      onClick={() => void sendMessage(question)}
                      key={question}
                    >
                      <span>{question}</span>
                      <span aria-hidden="true">→</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <article
                  className={`message message-${message.role}`}
                  key={message.id}
                >
                  {message.role === "assistant" ? (
                    <div className="bot-avatar" aria-hidden="true">
                      상담
                    </div>
                  ) : null}
                  <div className="message-body">
                    <span className="message-author">
                      {message.role === "assistant" ? "정책 상담" : "나"}
                    </span>
                    {message.role === "assistant" && !message.content ? (
                      <div className="typing-status" role="status">
                        <span className="typing-dots" aria-hidden="true">
                          <i />
                          <i />
                          <i />
                        </span>
                        관련 정책을 확인하고 있습니다.
                      </div>
                    ) : (
                      <>
                        <div className="markdown">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                        {message.role === "assistant" &&
                        message.traceId &&
                        !isStreaming ? (
                          <AnswerFeedback traceId={message.traceId} />
                        ) : null}
                      </>
                    )}
                  </div>
                </article>
              ))
            )}
            <div ref={messageEndRef} />
          </div>

          <form className="chat-composer" onSubmit={submitMessage}>
            <label className="sr-only" htmlFor="chat-question">
              청년정책 질문
            </label>
            <textarea
              id="chat-question"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="필요한 지원이나 현재 상황을 입력해 주세요."
              rows={3}
              maxLength={4000}
              disabled={isStreaming}
            />
            <div className="composer-footer">
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={excludeExpired}
                  onChange={(event) => setExcludeExpired(event.target.checked)}
                  disabled={isStreaming}
                />
                신청 마감 정책 제외
              </label>
              <div className="composer-actions">
                {isStreaming ? (
                  <button
                    className="button button-secondary button-small"
                    type="button"
                    onClick={() => abortRef.current?.abort()}
                  >
                    답변 중지
                  </button>
                ) : null}
                <button
                  className="button button-primary"
                  type="submit"
                  disabled={isStreaming || !draft.trim()}
                >
                  질문 보내기
                  <span aria-hidden="true">↑</span>
                </button>
              </div>
            </div>
          </form>
        </section>

        <aside className="active-policy-panel" aria-labelledby="active-title">
          <div className="active-policy-heading">
            <div>
              <p className="eyebrow">답변 근거</p>
              <h2 id="active-title">현재 상담 중인 정책</h2>
            </div>
            {activePolicies.length ? (
              <span className="count-badge">{activePolicies.length}</span>
            ) : null}
          </div>
          <p className="active-policy-description">
            현재 답변에서 참고하고 있는 정책입니다. 새로운 검색이 진행되면
            목록이 바뀔 수 있습니다.
          </p>
          {policyError ? (
            <div className="alert alert-error" role="alert">
              {policyError}
            </div>
          ) : activePolicies.length ? (
            <div className="active-policy-list">
              {activePolicies.map((policy) => (
                <PolicyCard compact policy={policy} key={policy.plcyNo} />
              ))}
            </div>
          ) : (
            <div className="empty-policies">
              <span aria-hidden="true">⌕</span>
              <strong>아직 활성 정책이 없습니다</strong>
              <p>질문을 보내면 답변에 사용된 정책이 여기에 표시됩니다.</p>
            </div>
          )}
          <div className="policy-caution">
            <strong>확인해 주세요</strong>
            <p>
              추천 결과는 신청 자격을 확정하지 않습니다. 신청 전 공식
              페이지의 최신 조건을 확인해 주세요.
            </p>
          </div>
        </aside>
      </div>

      <div className="data-actions">
        <button type="button" onClick={() => void deleteAllData()}>
          내 프로필과 상담 기록 모두 삭제
        </button>
        <span>마지막 이용일로부터 30일 후 자동 삭제됩니다.</span>
      </div>

      {showProfile && profile ? (
        <Modal
          title="내 상담 조건 수정"
          description="변경한 조건은 다음 질문부터 정책 검색에 반영됩니다."
          onClose={() => setShowProfile(false)}
        >
          <ProfileForm
            initialProfile={profile}
            busy={isSavingProfile}
            submitLabel="변경사항 저장"
            onSubmit={saveProfile}
          />
        </Modal>
      ) : null}
    </>
  );
}
