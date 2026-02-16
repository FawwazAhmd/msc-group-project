project:
title: "MSc Group Project – LLM-Based Legal Clause Analysis"
type: "Group Research Project"
university: "University of Liverpool"
degree: "MSc Computer Science"

overview:
description: >
This project investigates the use of Large Language Models (LLMs)
for legal clause classification and analysis. The system evaluates
multiple prompting strategies and model configurations to assess
performance on a binary (Yes/No) legal classification task.

repository_structure:
root: - data/ - models/ - scripts/ - experiments/ - results/ - README.yaml
description:
data: "Dataset files (TSV/CSV)"
models: "Model-specific scripts or configurations"
scripts: "Core execution scripts"
experiments: "Experimental variations (prompts, decoding configs)"
results: "Generated outputs and evaluation files"

models:
baseline:
name: "microsoft/Phi-3-mini-4k-instruct"
parameters: "3B"
framework: "HuggingFace Transformers"
precision: "float16"
device: "GPU (device_map=auto)"
decoding:
max_new_tokens: 5
do_sample: false

datasets:
primary_dataset:
file_format: "TSV"
required_columns: - text - answer_or_label
label_mapping:
yes: 1
no: 0
"1": 1
"0": 0

prompting_strategy:
baseline_prompt: |
You are a legal expert.
Answer ONLY Yes or No.

    Clause:
    {clause}

    Answer:

workflow:
process: - Load dataset - Map labels to binary format - Load tokenizer and model - Generate predictions - Save outputs to CSV - Compute evaluation metrics

evaluation:
metrics: - Accuracy - F1 Score
library: "scikit-learn"
outputs: - results_gpu.csv - printed_accuracy - printed_f1_score

installation:
environment_setup:
create_venv: "python -m venv llm_env"
activate_windows: "llm_env\\Scripts\\activate"
dependencies: - pandas - torch - transformers - scikit-learn - tqdm
install_command: "pip install -r requirements.txt"

execution:
general_run_command: "python <script_name>.py"
requirements: - Dataset must be present in data directory - GPU recommended for faster inference

branching_strategy:
main: "Stable, reviewed code only"
dev: "Integration branch for merging features"
feature_branches: - feature/contract-classification - feature/data-preprocessing - feature/prompt-engineering - feature/model-comparison

reproducibility:
requirements_file: "requirements.txt"
recommended_python_version: "Python 3.11"
hardware: "CUDA-enabled GPU recommended"
