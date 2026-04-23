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

# LeakGuard AI Simulator

An end-to-end supply-chain anomaly detection environment built on the Meta PyTorch OpenEnv framework.

**Deployed on Hugging Face Spaces**: [AtulK29/LeakGuardAI](https://huggingface.co/spaces/AtulK29/LeakGuardAI)

##  Overview
LeakGuard AI acts as a simulation engine designed to prevent revenue leakage within a typical Procure-to-Pay cycle. In this environment, your Reinforcement Learning (RL) agent acts as a **Virtual Auditor** whose job is to carefully process a continuous stream of incoming vendor invoices.

The core challenge? You must balance **preventing financial loss** (leaked revenue from invalid invoices) while **maintaining supply chain velocity** (represented through a Vendor Trust Score).

## The Environment

The environment runs as a standard `OpenEnv` OpenAPI server, providing endpoints for standard RL workflows (`/reset`, `/step`, `/state`).

An episode lasts a maximum of **15 turns**. During each turn, the environment dynamically generates between 1 and 3 simulated invoices. 

### Observation Space
At each turn, the RL Agent (Auditor) receives:
- **`turn_number`**: Current turn tracked.
- **`pending_invoices`**: An array of `Invoice` objects containing:
  - `id`: Internal Invoice ID.
  - `vendor_id`: Abstracted string representing a specific vendor.
  - `amount`: Financial value of the invoice ranging from $500 to $5000.
  - `grn_match`: Boolean identifying whether physical goods matched the invoice. This represents ground truth reality `(True = Valid, False = Leakage/Discrepancy)`.
- **`total_revenue_leaked`**: Cumulative financial loss thus far.
- **`vendor_trust_score`**: Trust velocity metric bounded between 0 and 100.

### Action Space
The agent replies with a strict JSON format dictating its action for exactly **one** invoice at a time:
```json
{
  "invoice_id": 105,
  "decision": "APPROVE" // OR "FLAG_FOR_AUDIT" OR "REJECT"
}
```

### Reward Mechanics
The internal rules engine distributes scalar reward signals and modifies state variables strictly based on the agent's action and the underlying truth of the `grn_match`:

- **APPROVE**:
  - If `grn_match=True`: Valid approval. Gain +2.0 `trust_score`.
  - If `grn_match=False`: Overlooked leak. Suffer a -0.2 penalty on `reward` and immediately leak the invoice `amount`.
- **FLAG_FOR_AUDIT**:
  - If `grn_match=False`: Safe recovery! Gain +0.3 `reward`.
  - If `grn_match=True`: Vendor gets annoyed by the delay. Penalty of -5.0 `trust_score`.
- **REJECT**:
  - If `grn_match=False`: Valid strictness, but burns relationship. Suffer a -2.0 `trust_score` penalty for outright rejection.
  - If `grn_match=True`: Vendor gets furious. Catastrophic -15.0 `trust_score` penalty!

**Final Reward Mapping:** At the end of 15 turns, the system aggregates the health of the supply chain and normalizes it to a final score strictly constrained between `0.0` and `1.0`:
> `Final Reward = max(0.0, (trust_score / 100.0) - min(1.0, leaked_revenue / 15000.0))`

##  Usage Guide

### Running via Hugging Face Access
To evaluate against the cloud-deployed Hugging Face space remotely, simply update the `ENV_URL` in our provided evaluation script to point to your Space's URL instead of localhost:

1. Clone this repository locally.
2. Edit `inference.py` and replace `ENV_URL = "http://localhost:8000"` with `ENV_URL = "https://your-username-leakguard-ai.hf.space"`.
3. Provide your Hugging Face API key as an environment variable and evaluate using Qwen/Qwen2.5!

```bash
# Example
uv run python inference.py
```

### Running Locally
To launch the FastAPI server locally for validation:
```bash
uv pip install -e .
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://127.0.0.1:8000/docs` to interact manually via the Swagger Dashboard.
