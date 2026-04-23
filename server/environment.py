# server/environment.py
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

from server.models import Invoice, LeakGuardState, LeakGuardAction

MARKET_DATABASE = {
    "Server_Rack": 1200.00,
    "Cisco_Router": 850.00,
    "Office_Chairs_Batch": 450.00,
    "Cloud_Storage_TB": 50.00,
    "GPU_Cluster_Node": 3200.00
}

VENDOR_LEDGER = {
    "VEND_101": {"reliability": 98.5, "past_flags": 1},
    "VEND_102": {"reliability": 75.0, "past_flags": 12},
    "VEND_103": {"reliability": 92.0, "past_flags": 3},
    "VEND_104": {"reliability": 88.5, "past_flags": 4}
}

class LeakGuardEnvironment(Environment):
    def __init__(self):
        self.state = LeakGuardState()

    def _generate_adversarial_invoices(self) -> None:
        num_new = random.randint(1, 3)
        items = list(MARKET_DATABASE.keys())
        vendors = list(VENDOR_LEDGER.keys())
        
        for _ in range(num_new):
            self.state.invoice_counter += 1
            item = random.choice(items)
            vendor = random.choice(vendors)
            
            base_price = MARKET_DATABASE[item]
            adversarial_price = round(base_price * random.uniform(0.95, 1.15), 2)
            is_valid = random.random() > 0.3
            
            invoice = Invoice(
                id=self.state.invoice_counter,
                vendor_id=vendor,
                item_name=item,
                amount=adversarial_price,
                grn_match=is_valid
            )
            self.state.active_invoices.append(invoice)

    def _get_observation(self, tool_response: str = "") -> str:
        obs_lines = [
            f"**Turn:** {self.state.current_turn} / {self.state.max_turns}",
            f"**Trust Score:** {self.state.trust_score:.1f}% | **Leaked Revenue:** ${self.state.leaked_revenue:.2f}",
            f"**Compliance Rules:** {self.state.active_compliance_rules}",
            ""
        ]
        if tool_response:
            obs_lines.extend([f"**Tool Output:** {tool_response}", ""])
            
        obs_lines.append("| ID | Vendor | Item | Amount | GRN Match |")
        obs_lines.append("|---|---|---|---|---|")
        for inv in self.state.active_invoices:
            obs_lines.append(f"| {inv.id} | {inv.vendor_id} | {inv.item_name} | ${inv.amount:.2f} | {inv.grn_match} |")
            
        return "\n".join(obs_lines)

    def reset(self) -> str:
        self.state = LeakGuardState()
        self._generate_adversarial_invoices()
        return self._get_observation()

    def step(self, action_dict: dict) -> Tuple[str, float, bool, dict]:
        action = LeakGuardAction(**action_dict)
        reward = 0.0
        tool_response = ""
        
        if random.random() < 0.05:
            self.state.active_compliance_rules = "UPDATED RULE: Zero variance allowed. All external vendors require strict GRN."

        if action.decision == "SEARCH_WEB":
            reward -= 0.01 
            if action.item_name in MARKET_DATABASE:
                tool_response = f"Market Median for {action.item_name} is ${MARKET_DATABASE[action.item_name]:.2f}"
            else:
                tool_response = "Item not found in market database."
                
        elif action.decision == "QUERY_HISTORY":
            reward -= 0.01
            if action.vendor_id in VENDOR_LEDGER:
                data = VENDOR_LEDGER[action.vendor_id]
                tool_response = f"Ledger - {action.vendor_id}: {data['reliability']}% reliable, {data['past_flags']} past flags."
            else:
                tool_response = "Vendor history not found."
                
        else:
            target_invoice = next((inv for inv in self.state.active_invoices if inv.id == action.invoice_id), None)
            
            if target_invoice:
                if action.decision == "APPROVE" and target_invoice.grn_match:
                    self.state.trust_score = min(100.0, self.state.trust_score + 2.0)
                    reward += 0.1
                elif action.decision == "APPROVE" and not target_invoice.grn_match:
                    self.state.leaked_revenue += target_invoice.amount
                    self.state.trust_score -= 1.0
                    reward -= 0.2
                elif action.decision == "FLAG_FOR_AUDIT" and not target_invoice.grn_match:
                    reward += 0.3
                elif action.decision == "FLAG_FOR_AUDIT" and target_invoice.grn_match:
                    self.state.trust_score -= 5.0
                elif action.decision == "REJECT" and not target_invoice.grn_match:
                    self.state.trust_score -= 2.0
                    reward += 0.1
                elif action.decision == "REJECT" and target_invoice.grn_match:
                    self.state.trust_score -= 15.0
                    reward -= 0.3
                elif action.decision == "NEGOTIATE" and action.discount_pct is not None:
                    if action.discount_pct > 0.20:
                        self.state.trust_score -= 10.0
                        tool_response = f"Vendor rejected {action.discount_pct*100}% discount. Trust penalized."
                        reward -= 0.1
                    else:
                        savings = target_invoice.amount * action.discount_pct
                        self.state.leaked_revenue = max(0.0, self.state.leaked_revenue - savings)
                        self.state.trust_score += 1.0
                        tool_response = f"Vendor accepted {action.discount_pct*100}% discount. Saved ${savings:.2f}."
                        reward += 0.2

                self.state.active_invoices.remove(target_invoice)
            else:
                tool_response = "Invalid or missing invoice ID."

        self.state.current_turn += 1
        self._generate_adversarial_invoices()
        
        done = self.state.current_turn >= self.state.max_turns
        
        if done:
            final_reward = (self.state.trust_score / 100.0) - min(1.0, self.state.leaked_revenue / 15000.0)
            reward += final_reward
            
        clamped_reward = max(0.0001, min(0.9999, float(reward)))
        
        return self._get_observation(tool_response), clamped_reward, done, {}