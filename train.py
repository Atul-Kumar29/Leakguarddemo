# 1. Pydantic Safety Patch - MUST be at the very top
import pydantic
from pydantic import ConfigDict
pydantic.BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)

# 2. Unsloth Import - MUST be before transformers/trl
from unsloth import FastLanguageModel

import os
import sys
import re
import json
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# 3. Path Management
sys.path.append(os.getcwd())
from server.environment import LeakGuardEnvironment

# 4. Model Initialization
# Using 4-bit quantization to fit comfortably in Kaggle's T4 RAM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct", 
    max_seq_length = 512,
    load_in_4bit = True,
    fast_inference = True,
)

# 5. Reward Logic
env = LeakGuardEnvironment()

def reward_logic(completions, **kwargs):
    rewards = []
    for content in completions:
        # Handle different TRL output formats
        text = content[0]['content'] if isinstance(content, list) else content
        try:
            # Extract JSON from the model's thought process
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                rewards.append(-1.0)
                continue
                
            action = json.loads(match.group(0))
            _, reward, done, _ = env.step(action)
            
            # Reset environment for the next roll-out if done
            if done: env.reset()
            rewards.append(float(reward))
        except Exception:
            # Penalize invalid JSON or environment crashes
            rewards.append(-1.0)
    return rewards

# 6. Dataset Generation (250 Loops)
obs = env.reset()
system_prompt = """You are a virtual auditor managing a multi-agent supply chain. 
Output raw JSON only.
Valid Actions:
1. Standard Audit: {"invoice_id": <int>, "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"}
2. Negotiate: {"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>}
3. Search Web: {"decision": "SEARCH_WEB", "item_name": "<string>"}
4. Query History: {"decision": "QUERY_HISTORY", "vendor_id": "<string>"}"""

# Prepare the 250-step sequence
prompts = []
for _ in range(250):
    user_prompt = f"Current Observation:\n{obs}\n\nPlease provide your action as a JSON object."
    prompts.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
dataset = Dataset.from_dict({"prompt": prompts})

# 7. Training Configuration
training_args = GRPOConfig(
    output_dir = "LeakGuard-RL-Auditor",
    learning_rate = 5e-6,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    max_steps = 250,
    logging_steps = 1,
    fp16 = True,           # Enable mixed precision for T4 GPU
    num_generations = 4,    # Number of completions to sample per prompt
    max_completion_length = 128,   
    save_steps = 100,              
)

# 8. Trainer Initialization
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_logic],
    args = training_args,
    train_dataset = dataset,
)

print("🚀 Starting LeakGuard RL Training (250 Loops)...")
trainer.train()

# 9. Push to a NEW Repository for HF Jobs
# Change 'AtulK29' to your verified username if needed
NEW_MODEL_ID = "AtulK29/LeakGuard-RL-Final"

print(f"📦 Uploading trained weights to {NEW_MODEL_ID}...")
model.push_to_hub_merged(
    NEW_MODEL_ID, 
    tokenizer, 
    save_method = "lora", 
    token = os.getenv("HF_TOKEN")
)
print("✅ Training and Upload Complete!")
