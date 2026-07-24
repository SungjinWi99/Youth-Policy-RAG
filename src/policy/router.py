from fastapi import APIRouter

from src.policy.models import Policy
from src.policy.schemas import PolicyBatchRequest, PolicyDetail


policy_router = APIRouter(prefix="/policies", tags=["policies"])


@policy_router.post("/batch", response_model=list[PolicyDetail])
def get_policies(payload: PolicyBatchRequest) -> list[PolicyDetail]:
    return [Policy.get(policy_id) for policy_id in payload.policy_ids]


@policy_router.get("/{policy_id}", response_model=PolicyDetail)
def get_policy(policy_id: str) -> PolicyDetail:
    return Policy.get(policy_id)
