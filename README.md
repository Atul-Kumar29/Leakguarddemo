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

🔗 **Quick Links:**
* **Trained Model Weights (LoRA):** [AtulK29/LeakGuard-RL-Auditor](https://huggingface.co/AtulK29/LeakGuard-RL-Auditor)
* **Live Environment API:** [AtulK29/LGDemo](https://huggingface.co/spaces/AtulK29/LGDemo)
* **Github Repository:** [Leakguarddemo](https://github.com/Atul-Kumar29/Leakguarddemo) 
* **Training Notebook:** [Google Colab Demo](https://colab.research.google.com/drive/1NUuyY5bZZiAqSfbsZmqDdICZbcAWZ-U1?usp=sharing) 

---

## 📌 Overview
LeakGuard AI tackles revenue leakage within the Procure-to-Pay cycle. This repository contains both the **OpenEnv Simulation Engine** and our custom-trained **RL Agent (Virtual Auditor)**, optimized via GRPO (Group Relative Policy Optimization). 

The agent's objective is to balance two critical supply chain metrics:
1. **Preventing Financial Loss** (Catching missing GRNs, anomalies, and fraud)
2. **Maintaining Supply Chain Velocity** (Preserving the Vendor Trust Score)

---

## 🧠 The RL Agent (Our Submission)

We trained a LoRA adapter on top of `Qwen/Qwen2.5-7B-Instruct` using the Unsloth framework for high-speed inference. The agent is designed to operate within a multi-agent framework, utilizing an advanced action space that allows it to perform investigative tasks before executing financial decisions.

### Expanded Action Space
The agent is strictly constrained to output valid JSON. It can take one of the following actions per turn:
1. **Standard Audit:** `{"invoice_id": <int>, "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"}`
2. **Negotiate:** `{"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>}` *(Max 0.20)*
3. **Search Web:** `{"decision": "SEARCH_WEB", "item_name": "<string>"}`
4. **Query History:** `{"decision": "QUERY_HISTORY", "vendor_id": "<string>"}`

*Note: Investigative actions (Search/Query) incur a minor token penalty to encourage efficiency.*

### State Observation
The environment provides the observation to the agent as a pre-formatted **Markdown table**, making dense state information easily digestible for the LLM. It tracks the Turn, Trust Score, Leaked Revenue, active Compliance Rules, and Pending Invoices.

---

## ⚙️ The Environment & Reward Mechanics

The environment runs as a standard `OpenEnv` FastAPI server, exposing RESTful endpoints (`/reset`, `/step`) for seamless RL integration. 

Reward signals force the model to balance aggression and trust:
* **APPROVE:** Positive reward for valid invoices; Heavy negative reward and revenue leak if GRN is missing.
* **FLAG_FOR_AUDIT:** Positive reward for catching leaks; Heavy trust penalty if used unnecessarily on a reliable vendor.
* **NEGOTIATE / REJECT:** Dynamic trust and financial adjustments based on historical context and current compliance rules.

*Final scores are normalized between `0.0` and `1.0`.*

---

## 🚀 Evaluation Guide for Judges

### Method 1: Automated Remote Evaluation (Recommended)
You can run our provided `inference.py` script to automatically evaluate the trained agent against the live Hugging Face environment. The script pulls our LoRA weights, initializes the base model, and runs the episodes.

```bash
# 1. Install required dependencies
pip install torch transformers peft accelerate requests openenv-core

# 2. Run the evaluation script
python inference.py
```
*The script handles state formatting and API communication, outputting the step-by-step reasoning and the final normalized score in the console.*

### Method 2: Direct API Integration
If you are integrating our environment into your own custom grading pipeline, you can interact directly with our hosted FastAPI instance.

**1. Reset Environment:**
`POST https://atulk29-lgdemo.hf.space/reset`
```json
{
  "observation": "**Turn:** 0 / 20\n**Trust Score:** 100.0% | **Leaked Revenue:** $0.00\n**Compliance Rules:** Standard variance allowed: 2%. Flag severe discrepancies.\n\n| ID | Vendor | Item | Amount | GRN Match |\n|---|---|---|---|---|\n| 1 | VEND_104 | Cloud_Storage_TB | $56.97 | False |"
}
```

**2. Take Action (Step):**
`POST https://atulk29-lgdemo.hf.space/step`

*Request Payload:*
```json
{
  "invoice_id": 1,
  "decision": "FLAG_FOR_AUDIT"
}
```

*Response:*
```json
{
  "observation": "**Turn:** 1 / 20\n...",
  "reward": 0.3,
  "done": false
}
```

### Method 3: Running the Environment Locally
If you prefer to run the OpenEnv simulation engine on your local machine instead of querying our Hugging Face Space:

```bash
# 1. Install the environment package locally
pip install -e .

# 2. Launch the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```
*You can then navigate to `http://127.0.0.1:7860/docs` to explore the API and test edge cases manually via the Swagger Dashboard.*
