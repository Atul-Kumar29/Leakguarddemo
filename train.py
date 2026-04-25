# ==============================================================================
# 1. CRITICAL: GLOBAL PATCHES & OPTIMIZED IMPORTS
# ==============================================================================
import pydantic
from pydantic import ConfigDict

# Patch Pydantic to ignore Tensor validation errors before any libraries load
pydantic.BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)

# Unsloth MUST be imported before TRL or Transformers
from unsloth import FastLanguageModel

import os
import sys
import re
import json
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# Fix pathing for the environment module
sys.path.append(os.getcwd())
from server.environment import LeakGuardEnvironment

# ==============================================================================
# 2. MODEL & ENVIRONMENT INITIALIZATION
# ==============================================================================
# Load Qwen 7B in 4-bit for Kaggle T4 GPU efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct", 
    max_seq_length = 512,
    load_in_4bit = True,
    fast_inference = True,
)

env = LeakGuardEnvironment()

# ==============================================================================
# 3. RL REWARD LOGIC
# ==============================================================================
def reward_logic(completions, **kwargs):
    rewards = []
    for content in completions:
        # Extract the text from the completion list
        text = content[0]['content'] if isinstance(content, list) else content
        try:
            # Look for JSON block in the model's output
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                rewards.append(-1.0)
                continue
                
            action = json.loads(match.group(0))
            _, reward, done, _ = env.step(action)
            
            # Reset environment if the sequence ended
            if done: env.reset()
            rewards.append(float(reward))
        except Exception:
            # Penalize formatting errors or invalid actions
            rewards.append(-1.0)
    return rewards

# ==============================================================================
# 4. DATASET PREPARATION (250 LOOPS)
# ==============================================================================
obs = env.reset()
system_prompt = "You are a virtual auditor managing a multi-agent supply chain. Output raw JSON only."

prompts = []
for _ in range(250):
    user_prompt = f"Current Observation:\n{obs}\n\nPlease provide your action as a JSON object."
    prompts.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
dataset = Dataset.from_dict({"prompt": prompts})

# ==============================================================================
# 5. GRPO TRAINING CONFIGURATION
# ==============================================================================
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
    save_steps = 100,              
)

# ==============================================================================
# 6. EXECUTION
# ==============================================================================
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_logic],
    args = training_args,
    train_dataset = dataset,
)

print("🚀 Starting LeakGuard RL Training (250 Steps)...")
trainer.train()

# ==============================================================================
# 7. EXPORT TO NEW REPO FOR HF JOBS
# ==============================================================================
# Replace with your actual HF username if different
NEW_MODEL_ID = "AtulK29/LeakGuard-RL-Final"

print(f"📦 Merging and Pushing to: {NEW_MODEL_ID}")
model.push_to_hub_merged(
    NEW_MODEL_ID, 
    tokenizer, 
    save_method = "lora", 
    token = os.getenv("HF_TOKEN")
)
print("✅ SUCCESS: Model is live for HF Jobs.")
