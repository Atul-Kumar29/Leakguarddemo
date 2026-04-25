import os
import sys
import re
import json
import pydantic
from pydantic import ConfigDict

# CRITICAL: Safety patch for Pydantic/Mergekit conflict
# This must happen before the trl imports
pydantic.main.BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", "unsloth") 

# Ensure the 'server' folder is visible to Python
sys.path.append(os.getcwd())

from server.environment import LeakGuardEnvironment

# 1. Model Initialization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct", 
    max_seq_length = 512,
    load_in_4bit = True,
    fast_inference = True,
)

# 2. Reward Logic
env = LeakGuardEnvironment()

def reward_logic(completions, **kwargs):
    rewards = []
    for content in completions:
        text = content[0]['content'] if isinstance(content, list) else content
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                rewards.append(-1.0)
                continue
            action = json.loads(match.group(0))
            _, reward, done, _ = env.step(action)
            if done: env.reset()
            rewards.append(float(reward))
        except:
            rewards.append(-1.0)
    return rewards

# 3. Dataset Generation
system_prompt = """You are a virtual auditor managing a multi-agent supply chain. 
Output raw JSON only.
Valid Actions:
1. Standard Audit: {"invoice_id": <int>, "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"}
2. Negotiate: {"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>}
3. Search Web: {"decision": "SEARCH_WEB", "item_name": "<string>"}
4. Query History: {"decision": "QUERY_HISTORY", "vendor_id": "<string>"}"""

prompts = []
for _ in range(250):
    obs = env.reset()
    user_prompt = f"Current Observation:\n{obs}\n\nPlease provide your action as a JSON object."
    prompts.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
dataset = Dataset.from_dict({"prompt": prompts})

# 4. Configuration
training_args = GRPOConfig(
    output_dir = "LeakGuard-RL-Auditor",
    learning_rate = 5e-6,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    max_steps = 250,
    logging_steps = 1,
    fp16 = True,     
    num_generations = 4,           
    max_completion_length = 128,   
    push_to_hub = False,
    save_steps = 100,              
)

# 5. Train
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_logic],
    args = training_args,
    train_dataset = dataset,
)

print("Starting RL training loops...")
trainer.train()
