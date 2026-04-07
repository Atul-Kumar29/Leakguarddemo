import random
from typing import Dict, Any, Tuple

try:
    from openenv import Environment
except ImportError:
    class Environment:
        def reset(self) -> Any:
            pass
        def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
            pass

from server.models import Invoice, LeakGuardState, LeakGuardObservation, LeakGuardAction

class LeakGuardEnvironment(Environment):
    def __init__(self):
        self.state = LeakGuardState()

    def _generate_mock_invoices(self) -> None:
        num_new = random.randint(1, 3)
        for _ in range(num_new):
            self.state.invoice_counter += 1
            is_valid = random.random() > 0.2
            # Amount between $500 and $5000
            amt = round(random.uniform(500, 5000), 2)
            
            invoice = Invoice(
                id=self.state.invoice_counter,
                vendor_id=f"VEND_{random.randint(100, 999)}",
                amount=amt,
                grn_match=is_valid
            )
            self.state.active_invoices.append(invoice)

    def _get_observation(self) -> LeakGuardObservation:
        return LeakGuardObservation(
            turn_number=self.state.current_turn,
            pending_invoices=self.state.active_invoices,
            total_revenue_leaked=self.state.leaked_revenue,
            vendor_trust_score=self.state.trust_score
        )

    def reset(self) -> dict:
        self.state = LeakGuardState()
        self._generate_mock_invoices()
        return self._get_observation().model_dump()

    def step(self, action_dict: dict) -> Tuple[dict, float, bool, dict]:
        action = LeakGuardAction(**action_dict)
        
        target_invoice = next((inv for inv in self.state.active_invoices if inv.id == action.invoice_id), None)
        
        reward = 0.0
        
        if target_invoice:
            if action.decision == "APPROVE" and target_invoice.grn_match:
                self.state.trust_score = min(100.0, self.state.trust_score + 2.0)
            elif action.decision == "APPROVE" and not target_invoice.grn_match:
                self.state.leaked_revenue += target_invoice.amount
                reward -= 0.2
            elif action.decision == "FLAG_FOR_AUDIT" and not target_invoice.grn_match:
                reward += 0.3
            elif action.decision == "FLAG_FOR_AUDIT" and target_invoice.grn_match:
                self.state.trust_score -= 5.0
            elif action.decision == "REJECT" and not target_invoice.grn_match:
                self.state.trust_score -= 2.0
            elif action.decision == "REJECT" and target_invoice.grn_match:
                self.state.trust_score -= 15.0
            
            self.state.active_invoices.remove(target_invoice)

        self.state.current_turn += 1
        self._generate_mock_invoices()
        
        done = self.state.current_turn >= self.state.max_turns
        
        if done:
            final_reward = max(0.0, (self.state.trust_score / 100.0) - min(1.0, self.state.leaked_revenue / 15000.0))
            reward = max(0.0001, min(0.9999, final_reward))

        return self._get_observation().model_dump(), reward, done, {}
