# MSc Group Project – LLM-Based Legal Clause Analysis

## Overview

This group project evaluates Large Language Models (LLMs) for binary legal clause classification (Yes/No).  
Instruction-tuned models are used with prompt-based inference, and performance is measured using standard classification metrics.

---

## Objectives

- Evaluate instruction-tuned LLMs for legal clause classification  
- Compare prompting strategies  
- Analyse performance using Accuracy and F1 Score  
- Maintain reproducible experimentation  

---

## Baseline Model

- **Model:** `microsoft/Phi-3-mini-4k-instruct` (3B parameters)  
- **Framework:** HuggingFace Transformers  
- **Precision:** float16  
- **Decoding:**  
  - `max_new_tokens = 5`  
  - `do_sample = False`  

---

## Dataset

Input file: `test.tsv`

Required columns:
- `text` (legal clause)  
- `answer` or `label` (Yes/No or 1/0)  

Labels are mapped to binary (Yes = 1, No = 0).

---

## Prompt Template

```
You are a legal expert.
Answer ONLY Yes or No.

Clause:
{clause}

Answer:
```

---

## How to Run

```bash
python -m venv llm_env
llm_env\Scripts\activate
pip install -r requirements.txt
python <script_name>.py
```

Output:
- `results_gpu.csv`
- Accuracy
- F1 Score

---

## Repository Structure

```
data/        # Dataset files
scripts/     # Core scripts
experiments/ # Prompt/model variations
results/     # Generated outputs
```

---

## Branching Strategy

- `main` – Stable, reviewed code  
- `feature/*` – Individual task branches  

All feature work is merged via Pull Requests.

---

## Academic Context

MSc Computer Science – University of Liverpool  
Group project focused on applied evaluation of LLMs for legal NLP tasks.
