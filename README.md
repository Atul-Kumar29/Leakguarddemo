---
title: LGDemo
emoji: 🚀
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 🛡️ LeakGuard AI: RL Auditor

An end-to-end supply-chain anomaly detection environment and trained Reinforcement Learning (RL) agent, built on the Meta PyTorch OpenEnv framework.

**Trained Model Weights (LoRA):** [AtulK29/LeakGuard-RL-Auditor](https://huggingface.co/AtulK29/LeakGuard-RL-Auditor)
**Live Environment API:** [AtulK29/LGDemo](https://huggingface.co/spaces/AtulK29/LGDemo)
**Github Repository:** [Leakguarddemo](https://github.com/Atul-Kumar29/Leakguarddemo) 
**Google Colab Notebook:** [LeakGuardEnvironmentDemo](https://colab.research.google.com/drive/1NUuyY5bZZiAqSfbsZmqDdICZbcAWZ-U1?usp=sharing) 


## 📌 Overview
LeakGuard AI tackles revenue leakage within the Procure-to-Pay cycle. This repository contains both the **OpenEnv Simulation Engine** and our custom-trained **RL Agent (Virtual Auditor)**, trained via GRPO (Group Relative Policy Optimization). 

The agent's objective is to balance **preventing financial loss** (catching missing GRNs/fraud) with **maintaining supply chain velocity** (Vendor Trust Score).

---

## 🧠 The RL Agent (Our Submission)

We trained a LoRA adapter on top of `Qwen/Qwen2.5-7B-Instruct` using Unsloth. The agent is trained to operate in a multi-agent framework with an advanced action space, allowing it to perform investigative tasks before making financial decisions.

### Expanded Action Space
The agent responds with a strict JSON format. It can take one of the following actions per turn:
1. **Standard Audit:** `{"invoice_id": <int>, "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"}`
2. **Negotiate:** `{"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>}` *(Max 0.20)*
3. **Search Web:** `{"decision": "SEARCH_WEB", "item_name": "<string>"}`
4. **Query History:** `{"decision": "QUERY_HISTORY", "vendor_id": "<string>"}`

*Note: Investigative actions (Search/Query) incur a minor token penalty to encourage efficiency.*

### State Observation
The agent parses dense state information formatting. The environment provides the observation to the agent as a pre-formatted **Markdown table**, tracking the Turn, Trust Score, Leaked Revenue, Compliance Rules, and current Pending Invoices.

---

## ⚙️ The Environment & Reward Mechanics

The environment runs as a standard `OpenEnv` FastAPI server, providing endpoints for standard RL workflows (`/reset`, `/step`). 

Reward signals force the model to balance aggression and trust:
- **APPROVE:** +Reward for valid invoices; -Reward and revenue leak if GRN is missing.
- **FLAG_FOR_AUDIT:** +Reward for catching leaks; Heavy Trust penalty if used on a valid trusted vendor.
- **NEGOTIATE / REJECT:** Dynamic trust and financial adjustments based on historical context and current compliance rules.

Final scores are normalized between `0.0` and `1.0`.

---

## 🚀 Evaluation & Usage Guide for Judges

### 1. Automated Remote Evaluation (Recommended)
This script pulls our trained LoRA adapter from Hugging Face, initializes the base model, and runs a full evaluation against the live Hugging Face environment.

```bash
# 1. Install required dependencies
pip install torch transformers peft accelerate requests openai

# 2. Run the evaluation script
python inference.py
```

The script handles state formatting, API communication, and outputs the step-by-step reasoning and final normalized score.

### 2. Manual API Testing
If you are integrating our environment into a custom grading pipeline, note that our environment returns observations as Markdown Strings, explicitly formatted for LLM consumption.

1. Reset Environment:
POST https://atulk29-lgdemo.hf.space/reset
```JSON
{
  "observation": "**Turn:** 0 / 20\n**Trust Score:** 100.0% | **Leaked Revenue:** $0.00\n**Compliance Rules:** Standard variance allowed: 2%. Flag severe discrepancies.\n\n| ID | Vendor | Item | Amount | GRN Match |\n|---|---|---|---|---|\n| 1 | VEND_104 | Cloud_Storage_TB | $56.97 | False |"
}

```
2. Take Action (Step):
POST https://atulk29-lgdemo.hf.space/step

Request Payload:
```JSON
{
  "invoice_id": 1,
  "decision": "FLAG_FOR_AUDIT"
}
```
Response Format:
The API will return the environment's state transition, reward, and the agent's action.
```json
{
  "observation": "**Turn:** 1 / 20\n...",
  "reward": 0.3,
  "done": false
}
```
### 3. Running the Environment Locally
If you wish to run the OpenEnv simulation engine on your local machine instead of querying the Hugging Face Space:
```bash
# 1. Install the environment package
pip install -e .

# 2. Launch the FastAPI server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
You can then navigate to http://127.0.0.1:8000/docs to interact manually via the Swagger Dashboard.
