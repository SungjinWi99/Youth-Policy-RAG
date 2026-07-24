"use client";

import { FormEvent, useState } from "react";

import type { Profile } from "@/lib/types";

const REGIONS = [
  "서울",
  "부산",
  "대구",
  "인천",
  "광주",
  "대전",
  "울산",
  "세종",
  "경기",
  "강원",
  "충북",
  "충남",
  "전북",
  "전남",
  "경북",
  "경남",
  "제주",
];

const EMPTY_PROFILE: Profile = {
  age: null,
  gender: null,
  job: null,
  income: null,
  region: null,
};

type ProfileFormProps = {
  initialProfile?: Profile;
  requireConsent?: boolean;
  busy?: boolean;
  submitLabel: string;
  onSubmit: (profile: Profile, acceptedStorage: boolean) => Promise<void>;
};

export function ProfileForm({
  initialProfile = EMPTY_PROFILE,
  requireConsent = false,
  busy = false,
  submitLabel,
  onSubmit,
}: ProfileFormProps) {
  const [profile, setProfile] = useState<Profile>(initialProfile);
  const [acceptedStorage, setAcceptedStorage] = useState(false);
  const [error, setError] = useState("");

  function setField<K extends keyof Profile>(key: K, value: Profile[K]) {
    setProfile((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    if (requireConsent && !acceptedStorage) {
      setError("상담을 시작하려면 정보 저장 안내에 동의해 주세요.");
      return;
    }
    try {
      await onSubmit(profile, acceptedStorage);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "프로필을 저장하지 못했습니다.",
      );
    }
  }

  return (
    <form className="profile-form" onSubmit={handleSubmit}>
      <div className="form-grid">
        <label className="field">
          <span>
            나이 <small>선택</small>
          </span>
          <span className="input-with-unit">
            <input
              inputMode="numeric"
              min="0"
              max="120"
              name="age"
              type="number"
              value={profile.age ?? ""}
              onChange={(event) =>
                setField(
                  "age",
                  event.target.value ? Number(event.target.value) : null,
                )
              }
              placeholder="예: 27"
            />
            <span>세</span>
          </span>
        </label>

        <label className="field">
          <span>
            거주 지역 <small>선택</small>
          </span>
          <select
            name="region"
            value={profile.region ?? ""}
            onChange={(event) =>
              setField("region", event.target.value || null)
            }
          >
            <option value="">선택하지 않음</option>
            {REGIONS.map((region) => (
              <option value={region} key={region}>
                {region}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>
            현재 상태·직업 <small>선택</small>
          </span>
          <input
            name="job"
            type="text"
            value={profile.job ?? ""}
            onChange={(event) => setField("job", event.target.value || null)}
            placeholder="예: 취업준비생, 직장인"
          />
        </label>

        <label className="field">
          <span>
            연 소득 <small>선택</small>
          </span>
          <span className="input-with-unit">
            <input
              inputMode="numeric"
              min="0"
              name="income"
              type="number"
              value={profile.income ?? ""}
              onChange={(event) =>
                setField(
                  "income",
                  event.target.value ? Number(event.target.value) : null,
                )
              }
              placeholder="예: 3000"
            />
            <span>만원</span>
          </span>
        </label>

        <label className="field field-full">
          <span>
            성별 <small>선택</small>
          </span>
          <select
            name="gender"
            value={profile.gender ?? ""}
            onChange={(event) =>
              setField("gender", event.target.value || null)
            }
          >
            <option value="">선택하지 않음</option>
            <option value="여성">여성</option>
            <option value="남성">남성</option>
          </select>
        </label>
      </div>

      <div className="profile-info-box">
        입력한 정보는 관련 정책을 찾는 데만 사용됩니다. 정보가 없거나
        정확하지 않은 항목은 비워두고 상담 중에 설명할 수 있습니다.
      </div>

      {requireConsent ? (
        <label className="consent-field">
          <input
            type="checkbox"
            checked={acceptedStorage}
            onChange={(event) => setAcceptedStorage(event.target.checked)}
          />
          <span>
            프로필과 상담 기록이 마지막 이용일로부터 30일간 저장되는 것에
            동의합니다. 언제든 직접 삭제할 수 있습니다.
          </span>
        </label>
      ) : null}

      {error ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : null}

      <button
        className="button button-primary button-block"
        type="submit"
        disabled={busy}
      >
        {busy ? "저장 중..." : submitLabel}
      </button>
    </form>
  );
}
