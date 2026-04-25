import os
import sys
import re
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", "unsloth") 

# Path Fix for Kaggle/Colab
sys.path.append(os.getcwd())
from server.environment import LeakGuardEnvironment

# 1. Load Model
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

# 3. Create 250 Prompts
obs = env.reset()
system_prompt = "You are a virtual auditor. Output raw JSON only."
dataset = Dataset.from_dict({
    "prompt": [[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Current Observation:\n{obs}"}
    ]] * 250
})

# 4. Config (250 Steps)
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
)

# 5. Train
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_logic],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# 6. Push to a NEW Repository for HF Jobs
# Change 'AtulK29' to your exact HF username
NEW_REPO = "AtulK29/LeakGuard-RL-Final"

print(f"📦 Uploading to NEW repo: {NEW_REPO}...")
model.push_to_hub_merged(
    NEW_REPO, 
    tokenizer, 
    save_method = "lora", 
    token = os.getenv("HF_TOKEN")
)
print("✅ Done! Your HF Job is ready to use this model.")
