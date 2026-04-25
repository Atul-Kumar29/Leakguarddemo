import os
import sys
import re
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", "unsloth") 

IS_COLAB = "google.colab" in sys.modules

# Path Management
BASE_PATH = "/content/LeakGuardAI" if IS_COLAB else os.getcwd()
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

from server.environment import LeakGuardEnvironment

# 1. Model Initialization (Matches your Qwen 2.5 7B base)
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

# 3. Dataset Generation (Mirroring your exact inference prompt)
system_prompt = """You are a virtual auditor managing a multi-agent supply chain. 
Your goal is to balance preventing financial loss (leaked revenue) with maintaining supply chain velocity (vendor trust score).

You have an expanded action space. You must output a raw JSON object (without markdown blocks) representing your action.

Valid Actions:
1. Standard Audit: {"invoice_id": <int>, "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"}
2. Negotiate: {"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>}
3. Search Web: {"decision": "SEARCH_WEB", "item_name": "<string>"}
4. Query History: {"decision": "QUERY_HISTORY", "vendor_id": "<string>"}"""

prompts = []
for _ in range(60):
    obs = env.reset()
    user_prompt = f"Current Observation:\n{obs}\n\nPlease provide your action as a JSON object."
    prompts.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
dataset = Dataset.from_dict({"prompt": prompts})

# 4. Adaptive Configuration
output_directory = "/content/drive/MyDrive/LeakGuard-RL-Auditor" if IS_COLAB else "LeakGuard-RL-Auditor"
HF_USERNAME = "AtulK29" # Updated to your HF Username

training_args = GRPOConfig(
    output_dir = output_directory,
    learning_rate = 5e-6,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    max_steps = 60,
    logging_steps = 1,
    
    # Hardware specific precision
    bf16 = not IS_COLAB, 
    fp16 = IS_COLAB,     
    
    # Speed Optimizations
    num_generations = 4,           
    max_completion_length = 128,   
    
    # Hub sync only on HF Jobs
    push_to_hub = not IS_COLAB,
    hub_model_id = f"{HF_USERNAME}/LeakGuard-RL-Auditor" if not IS_COLAB else None, 
    hub_strategy = "every_save" if not IS_COLAB else "none",
    save_steps = 100,              
)

# 5. Execute Training
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_logic],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()
