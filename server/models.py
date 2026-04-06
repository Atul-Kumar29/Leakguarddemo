from pydantic import BaseModel, Field
from typing import List, Literal

class Invoice(BaseModel):
    id: int
    vendor_id: str
    amount: float
    grn_match: bool

class LeakGuardObservation(BaseModel):
    turn_number: int
    pending_invoices: List[Invoice]
    total_revenue_leaked: float
    vendor_trust_score: float

class LeakGuardAction(BaseModel):
    invoice_id: int
    decision: Literal["APPROVE", "FLAG_FOR_AUDIT", "REJECT"]

class LeakGuardState(BaseModel):
    current_turn: int = 0
    max_turns: int = 15
    leaked_revenue: float = 0.0
    trust_score: float = 100.0
    active_invoices: List[Invoice] = Field(default_factory=list)
    invoice_counter: int = 0
