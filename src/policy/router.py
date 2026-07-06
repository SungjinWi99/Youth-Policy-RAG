from fastapi import APIRouter

from src.policy.models import Policy
from src.policy.schemas import PolicyDetail


policy_router = APIRouter(prefix="/policies", tags=["policies"])


@policy_router.get("/{policy_id}", response_model=PolicyDetail)
def get_policy(policy_id: str) -> PolicyDetail:
    return Policy.get(policy_id)
