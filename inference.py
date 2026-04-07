import os
import sys
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = "http://localhost:8000"
NUM_EPISODES = 3

def main():
    if not HF_TOKEN:
        print("Warning: HF_TOKEN environment variable not set. Using dummy token.", file=sys.stderr)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy"
    )

    total_score = 0.0

    for episode in range(1, NUM_EPISODES + 1):
        task_name = "LeakGuard"
        print(f"[START] task={task_name}", flush=True)
        
        try:
            res = requests.post(f"{ENV_URL}/reset")
            res.raise_for_status()
            obs = res.json()["observation"]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to environment... is the server running? {e}", file=sys.stderr)
            return
            
        done = False
        episode_reward = 0.0
        step_counter = 0

        while not done:
            system_prompt = """You are a virtual auditor receiving a stream of incoming vendor invoices.
Your goal is to balance preventing financial loss (leaked revenue) with maintaining supply chain velocity (vendor trust score).

You will receive an observation containing:
- turn_number: Current turn.
- pending_invoices: A list of invoices. An invoice has an 'id', 'vendor_id', 'amount', and 'grn_match'.
  NOTE: grn_match=True means the physical goods match the invoice (valid). False means discrepancy (leak).
- total_revenue_leaked: Cumulative leaked revenue so far.
- vendor_trust_score: Trust score between 0 and 100.

You must output a raw JSON object (without markdown code blocks) representing your action. The JSON must have the following structure:
{
  "invoice_id": <int>,
  "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"
}

Rules:
- Select exactly ONE 'invoice_id' from the 'pending_invoices' list.
- Choose one of the 3 decisions. Approve valid invoices, and flag/reject invalid ones appropriately to maximize final reward."""

            user_prompt = f"Current Observation:\n{json.dumps(obs, indent=2)}\n\nPlease provide your action as a JSON object."

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                
                action_text = response.choices[0].message.content.strip()
                if action_text.startswith("```json"):
                    action_text = action_text[7:-3].strip()
                elif action_text.startswith("```"):
                    action_text = action_text[3:-3].strip()

                action_dict = json.loads(action_text)
                
                step_res = requests.post(f"{ENV_URL}/step", json=action_dict)
                step_res.raise_for_status()
                step_data = step_res.json()
                
                obs = step_data["observation"]
                reward = step_data["reward"]
                done = step_data["done"]
                
                step_counter += 1
                print(f"[STEP] step={step_counter} reward={reward:.4f}", flush=True)
                
                if done:
                    episode_reward = reward
                    
            except Exception as e:
                print(f"Error during interaction: {e}", file=sys.stderr)
                if obs["pending_invoices"]:
                    fallback_action = {
                        "invoice_id": obs["pending_invoices"][0]["id"],
                        "decision": "APPROVE"
                    }
                    print(f"Falling back to action: {fallback_action}", file=sys.stderr)
                    step_res = requests.post(f"{ENV_URL}/step", json=fallback_action)
                    step_data = step_res.json()
                    obs = step_data["observation"]
                    
                    step_reward = step_data.get("reward", 0.0)
                    step_counter += 1
                    print(f"[STEP] step={step_counter} reward={step_reward:.4f}", flush=True)
                    
                    done = step_data["done"]
                    if done:
                        episode_reward = step_data["reward"]
                else:
                    break
        
        print(f"[END] task={task_name} score={episode_reward:.4f} steps={step_counter}", flush=True)
        total_score += episode_reward

    avg_score = total_score / NUM_EPISODES

if __name__ == "__main__":
    main()
