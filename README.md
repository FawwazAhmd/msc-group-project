# MSc Group Project – Analysing the Effectiveness of LLMs in Legal Tasks

## 📌 Overview

This project investigates the performance of Large Language Models (LLMs) on a range of legal Natural Language Processing (NLP) tasks using a **zero-shot learning approach**.

We evaluate how well instruction-tuned open-weight models generalise across different legal classification tasks without fine-tuning.

This work is part of the MSc Computer Science programme at the University of Liverpool.

---

## 🎯 Project Aim

- Evaluate open-weight LLMs on legal tasks
- Compare performance across multiple models
- Analyse zero-shot generalisation ability
- Benchmark models using structured legal datasets

---

## 🧠 Models Evaluated

- **Phi-3 Mini (3B)**
- **TinyLlama (1.1B)**
- **SmolLM2 (1.7B)**
- **Qwen 2.5 (3B)** _(for prompt engineering analysis)_

All models are evaluated under identical conditions for fair comparison.

---

## ⚙️ Methodology

The project follows a consistent evaluation pipeline:

**Dataset → Prompt → Model → Prediction → Metrics**

- **Zero-shot setting** (no fine-tuning)
- **Task-specific prompts**
- **Temperature tuning (0.01 – 1.0)**
- **Binary classification (Yes/No)**
- **Answer extraction from generated output**

### Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score

---

## 📂 Legal NLP Tasks

We evaluate **7 legal tasks**:

- Contract QA
- Privacy Policy QA
- Definition Classification
- Data Retention
- Data Security
- Do Not Track
- Contract NLI Confidentiality

These tasks cover both **contract analysis** and **privacy policy understanding**.

---

## 📊 Model Performance (F1 Score)

| Task                         | Phi-3 | TinyLlama | SmolLM2 |
| ---------------------------- | ----- | --------- | ------- |
| contract_qa                  | 0.63  | 0.66      | 0.70    |
| contract_nli_confidentiality | 0.70  | 0.65      | 0.42    |
| privacy_policy_qa            | 0.81  | 0.67      | 0.63    |
| definition_classification    | 0.75  | 0.70      | 0.83    |
| opp115_data_retention        | 0.75  | 0.67      | 0.46    |
| opp115_data_security         | 0.83  | 0.58      | 0.72    |
| opp115_do_not_track          | 0.85  | 0.52      | 0.39    |

---

## 🔍 Key Insights

- **Phi-3 Mini** achieved the strongest overall performance and consistency
- **SmolLM2** performed best on structured tasks (e.g., definition classification)
- **TinyLlama** showed higher recall but lower precision
- Smaller models struggled with complex legal reasoning tasks

---

## 🔥 Prompt Engineering Impact

Prompt design significantly influenced performance.

| Model    | Prompt Type | F1 Score |
| -------- | ----------- | -------- |
| Qwen 2.5 | Simple      | 0.61     |
| Qwen 2.5 | Improved    | 0.79     |

> Improved prompts with task-specific guidance led to substantial performance gains.

---

## ⚠️ Challenges

- Prompt sensitivity affected model outputs significantly
- Larger models required more computational resources
- Performance varied across tasks depending on complexity
- Some models produced inconsistent predictions

---

## 🏁 Conclusion

- A unified evaluation pipeline was successfully developed
- Zero-shot LLMs show strong potential in legal NLP tasks
- Model performance varies based on task complexity and structure
- Prompt engineering and decoding strategies play a critical role
- Phi-3 Mini demonstrated the most balanced performance overall

---

## 🚀 Requirements

- Python 3.11
- CUDA-enabled GPU recommended

### Install Dependencies

pip install pandas torch transformers scikit-learn tqdm

---

## 📌 Academic Context

This project was conducted as part of the **COMP530 MSc Group Project** at the University of Liverpool, focusing on applied evaluation of LLMs in legal NLP.

---
