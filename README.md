# Incident Triage AI System

A custom-built AI agent for automated incident triage, designed to identify root causes in distributed systems using structured reasoning, dependency analysis, and multi-step decision making.

---

## Overview

This project simulates real-world production incident scenarios and solves them using a deterministic reasoning agent.

The system:

* Processes alerts, logs, and service dependencies
* Builds a hypothesis model of possible root causes
* Iteratively gathers evidence
* Identifies the most probable root cause efficiently

Unlike typical solutions, this system does **not rely on external LLM APIs**.
All reasoning is implemented from scratch using a custom agent.

---

## Key Features

* Custom AI Agent (no external APIs)
* Multi-step reasoning and decision making
* Dependency-aware root cause analysis
* Structured evaluation with scoring system
* FastAPI backend for environment interaction
* Streamlit UI for interactive visualization (optional)

---

## Project Structure

```
incident-triage-env/
├── env.py           # Environment simulation
├── scenarios.py     # Predefined incident scenarios
├── agent.py         # Custom reasoning agent
├── grader.py        # Evaluation and scoring logic
├── inference.py     # Main evaluation script (validator compliant)
├── main.py          # FastAPI server
├── app.py           # Streamlit UI (demo)
├── requirements.txt
├── Dockerfile
├── openenv.yaml
```

---

## How the Agent Works

The `TriageAgent` follows a structured reasoning pipeline:

### 1. Signal Analysis

Assigns weights to alerts and logs to estimate service impact.

### 2. Hypothesis Modeling

Maintains a score table:

```
service → suspicion score
```

### 3. Dependency Propagation

Propagates failure signals across service dependencies to identify upstream causes.

### 4. Action Planning

Chooses actions based on confidence thresholds:

* Inspect services
* Filter noisy alerts
* Correlate logs
* Identify root cause

---

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Run inference (evaluation)

```bash
python inference.py
```

Expected output format:

```
[START] Task: easy
[STEP] action=inspect_service target=database reward=2.0
[STEP] action=identify_root_cause target=database reward=14.0
[END] correct=True steps=2 score=1.0
```

---

### Run API server

```bash
uvicorn main:app --reload
```

---

### Run UI (optional)

```bash
streamlit run app.py
```

---

## Evaluation

The system is evaluated across three scenarios:

* Easy
* Medium
* Hard

Metrics:

* Correctness
* Number of steps (efficiency)
* Decision quality

---

## Design Philosophy

* Deterministic and explainable decisions
* No reliance on external AI services
* Modular and extensible architecture
* Focus on real-world incident debugging patterns

---

## Future Improvements

* Reinforcement learning-based agent
* More complex and noisy scenarios
* Real-time streaming logs integration
* Advanced visualization of causal graphs

---

## Author

Chethan
