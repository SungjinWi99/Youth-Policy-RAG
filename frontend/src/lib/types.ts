export type Profile = {
  age: number | null;
  gender: string | null;
  job: string | null;
  income: number | null;
  region: string | null;
};

export type SessionStatus = {
  expires_at: string;
  profile: Profile;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  traceId?: string;
};

export type ConversationSnapshot = {
  messages: Array<Pick<ChatMessage, "role" | "content">>;
  active_policy_ids: string[];
};

export type PolicyDetail = {
  plcyNo: string;
  plcyNm: string;
  plcyKywdNm?: string;
  plcyExplnCn?: string;
  plcySprtCn?: string;
  lclsfNm?: string;
  mclsfNm?: string;
  sprvsnInstCdNm?: string;
  operInstCdNm?: string;
  bizPrdBgngYmd?: string;
  bizPrdEndYmd?: string;
  bizPrdEtcCn?: string;
  aplyYmd?: string;
  plcyAplyMthdCn?: string;
  aplyUrlAddr?: string;
  refUrlAddr1?: string;
  refUrlAddr2?: string;
  sprtTrgtMinAge?: string;
  sprtTrgtMaxAge?: string;
  earnMinAmt?: string;
  earnMaxAmt?: string;
  earnEtcCn?: string;
  ptcpPrpTrgtCn?: string;
  addAplyQlfcCndCn?: string;
  sbmsnDcmntCn?: string;
  srngMthdCn?: string;
  etcMttrCn?: string;
  rgtrInstCdNm?: string;
  zipCd?: string;
};
