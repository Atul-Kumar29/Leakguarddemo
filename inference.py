# inference.py
import os
import sys
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
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
        print(f"[START] task={task_name} episode={episode}", flush=True)

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
            system_prompt = """You are a virtual auditor managing a multi-agent supply chain. 
Your goal is to balance preventing financial loss (leaked revenue) with maintaining supply chain velocity (vendor trust score).

You have an expanded action space. You must output a raw JSON object (without markdown blocks) representing your action.

Valid Actions:
1. Standard Audit: {"invoice_id": <int>, "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"}
2. Negotiate: {"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>} (Max 0.20 discount)
3. Search Web: {"decision": "SEARCH_WEB", "item_name": "<string>"}
4. Query History: {"decision": "QUERY_HISTORY", "vendor_id": "<string>"}

Rules:
- Analyze the observation table carefully. Ensure you adapt to any updated Compliance Rules.
- Use SEARCH_WEB and QUERY_HISTORY to investigate discrepancies, but they incur a small token penalty (-0.01).
- To process an invoice, use APPROVE, FLAG_FOR_AUDIT, REJECT, or NEGOTIATE."""

            user_prompt = f"Current Observation:\n{obs}\n\nPlease provide your action as a JSON object."

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
                print(f"[STEP] step={step_counter} reward={reward:.4f} action={action_dict.get('decision')}", flush=True)

                if done:
                    episode_reward = reward

            except Exception as e:
                print(f"Error during interaction: {e}", file=sys.stderr)
                
                invoice_id_fallback = 1
                if "ID | Vendor" in obs:
                    try:
                        lines = obs.split("\n")
                        for line in lines:
                            if line.startswith("|") and not line.startswith("| ID") and not line.startswith("|---"):
                                parts = line.split("|")
                                if len(parts) > 1:
                                    invoice_id_fallback = int(parts[1].strip())
                                    break
                    except:
                        pass
                        
                fallback_action = {
                    "invoice_id": invoice_id_fallback,
                    "decision": "APPROVE"
                }
                print(f"Falling back to action: {fallback_action}", file=sys.stderr)
                step_res = requests.post(f"{ENV_URL}/step", json=fallback_action)
                step_data = step_res.json()
                obs = step_data["observation"]

                step_reward = step_data.get("reward", 0.0)
                step_counter += 1
                print(f"[STEP] step={step_counter} reward={step_reward:.4f} (fallback)", flush=True)

                done = step_data["done"]
                if done:
                    episode_reward = step_data["reward"]

        print(f"[END] task={task_name} episode={episode} score={episode_reward:.4f} steps={step_counter}", flush=True)
        total_score += episode_reward

    avg_score = total_score / NUM_EPISODES
    print(f"\n[FINAL] Average Score across {NUM_EPISODES} episodes: {avg_score:.4f}")

if __name__ == "__main__":
    main()