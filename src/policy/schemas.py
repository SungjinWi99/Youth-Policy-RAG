from pydantic import BaseModel, ConfigDict, Field


class PolicyBatchRequest(BaseModel):
    policy_ids: list[str] = Field(min_length=1, max_length=10)


class PolicyDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    plcyNo: str
    plcyNm: str
    plcyKywdNm: str = ""
    plcyExplnCn: str = ""
    plcySprtCn: str = ""
    lclsfNm: str = ""
    mclsfNm: str = ""
    sprvsnInstCdNm: str = ""
    operInstCdNm: str = ""
    bizPrdBgngYmd: str = ""
    bizPrdEndYmd: str = ""
    bizPrdEtcCn: str = ""
    aplyYmd: str = ""
    plcyAplyMthdCn: str = ""
    aplyUrlAddr: str = ""
    refUrlAddr1: str = ""
    refUrlAddr2: str = ""
    sprtTrgtMinAge: str = ""
    sprtTrgtMaxAge: str = ""
    earnMinAmt: str = ""
    earnMaxAmt: str = ""
    earnEtcCn: str = ""
    ptcpPrpTrgtCn: str = ""
    addAplyQlfcCndCn: str = ""
    sbmsnDcmntCn: str = ""
    srngMthdCn: str = ""
    etcMttrCn: str = ""
    rgtrInstCdNm: str = ""
    zipCd: str = ""
