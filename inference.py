import os
import json
import requests
from openai import OpenAI

ENV_URL = "http://localhost:8000"
NUM_EPISODES = 3
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

def main():
    # Read the token from 'HF_TOKEN' environment variable instead of hardcoding to pass HF security scans.
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Warning: API key environment variable not set.")
        return

    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=api_key
    )

    total_score = 0.0

    print(f"Starting Evaluation using model: {MODEL_NAME}")
    print(f"Target Environment URL: {ENV_URL}")
    print("-" * 40)

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n--- Episode {episode} ---")
        
        try:
            res = requests.post(f"{ENV_URL}/reset")
            res.raise_for_status()
            obs = res.json()["observation"]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to environment... is the server running? {e}")
            return
            
        done = False
        episode_reward = 0.0

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
                
                print(f"Turn {obs['turn_number'] - 1} | Action: {action_dict} | Step Reward: {reward:.4f}")
                
                if done:
                    episode_reward = reward
                    
            except Exception as e:
                print(f"Error during interaction: {e}")
                if obs["pending_invoices"]:
                    fallback_action = {
                        "invoice_id": obs["pending_invoices"][0]["id"],
                        "decision": "APPROVE"
                    }
                    print(f"Falling back to action: {fallback_action}")
                    step_res = requests.post(f"{ENV_URL}/step", json=fallback_action)
                    step_data = step_res.json()
                    obs = step_data["observation"]
                    done = step_data["done"]
                    if done:
                        episode_reward = step_data["reward"]
                else:
                    break
        
        print(f"Episode {episode} Final Score (Reward): {episode_reward:.4f}")
        total_score += episode_reward

    avg_score = total_score / NUM_EPISODES
    print("\n" + "=" * 40)
    print(f"Reproducible Baseline Score (Average over {NUM_EPISODES} episodes): {avg_score:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()
