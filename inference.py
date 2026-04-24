# inference.py
import os
import sys
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ENV_URL = os.getenv("ENV_URL", "https://atulk29-lgdemo.hf.space")
NUM_EPISODES = 3

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
ADAPTER_REPO = "AtulK29/LeakGuard-RL-Auditor"

def main():
    print(f"Loading trained model {ADAPTER_REPO}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    
    total_score = 0.0

    for episode in range(1, NUM_EPISODES + 1):
        task_name = "LeakGuard"
        print(f"[START] task={task_name} episode={episode}", flush=True)

        try:
            res = requests.post(f"{ENV_URL}/reset")
            res.raise_for_status()
            obs = res.json()["observation"]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to environment... {e}", file=sys.stderr)
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
2. Negotiate: {"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>}
3. Search Web: {"decision": "SEARCH_WEB", "item_name": "<string>"}
4. Query History: {"decision": "QUERY_HISTORY", "vendor_id": "<string>"}"""

            user_prompt = f"Current Observation:\n{obs}\n\nPlease provide your action as a JSON object."
            prompt = f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Response:\n"

            try:
                inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
                
                input_length = inputs["input_ids"].shape[1]
                action_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

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
                fallback_action = {"invoice_id": 1, "decision": "APPROVE"}
                step_res = requests.post(f"{ENV_URL}/step", json=fallback_action)
                step_data = step_res.json()
                obs = step_data["observation"]
                done = step_data["done"]
                if done:
                    episode_reward = step_data.get("reward", 0.0)

        print(f"[END] task={task_name} episode={episode} score={episode_reward:.4f} steps={step_counter}", flush=True)
        total_score += episode_reward

    avg_score = total_score / NUM_EPISODES
    print(f"\n[FINAL] Average Score across {NUM_EPISODES} episodes: {avg_score:.4f}")

if __name__ == "__main__":
    main()