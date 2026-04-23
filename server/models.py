from typing import List, Optional
from pydantic import BaseModel

class Invoice(BaseModel):
    id: int
    vendor_id: str
    item_name: str
    amount: float
    grn_match: bool

class LeakGuardState(BaseModel):
    current_turn: int = 0
    max_turns: int = 20
    trust_score: float = 100.0
    leaked_revenue: float = 0.0
    invoice_counter: int = 0
    active_invoices: List[Invoice] = []
    active_compliance_rules: str = "Standard variance allowed: 2%. Flag severe discrepancies."

class LeakGuardAction(BaseModel):
    decision: str 
    invoice_id: Optional[int] = None
    discount_pct: Optional[float] = None
    vendor_id: Optional[str] = None
    item_name: Optional[str] = None