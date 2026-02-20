import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# =====================================
# CONFIGURATION
# =====================================

DATASET_PATH = "contract_nli_confidentiality_test.tsv"
OUTPUT_PATH = "contract_nli_confidentiality_gpu_results.csv"

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# =====================================
# CHECK CUDA
# =====================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Use CPU script instead.")

print("Using GPU:", torch.cuda.get_device_name(0))

# =====================================
# LOAD DATASET
# =====================================

print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH, sep="\t")

TEXT_COLUMN = "text"
LABEL_COLUMN = "answer"

# Convert labels to 0/1
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower().map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

print("Dataset loaded successfully.")

# =====================================
# LOAD MODEL (GPU)
# =====================================

print("\nLoading model on GPU...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print("Model loaded successfully.")

# =====================================
# PROMPT FUNCTION
# =====================================

def create_prompt(clause):
    return f"""
You are a legal expert.

Does the following clause imply that the existence, terms, or discussions 
of the agreement itself must be kept confidential?

Answer ONLY Yes or No.

Clause:
{clause}

Answer:
"""

# =====================================
# RUN INFERENCE
# =====================================

predictions = []

print("\nRunning evaluation...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    prompt = create_prompt(row[TEXT_COLUMN])

    output = pipe(
        prompt,
        max_new_tokens=5,
        do_sample=False
    )[0]["generated_text"]

    output = output.lower()

    # Extract answer safely
    generated = output.split("answer:")[-1].strip()

    if generated.startswith("yes"):
        predictions.append(1)
    elif generated.startswith("no"):
        predictions.append(0)
    else:
        predictions.append(0)  # default fallback

df["prediction"] = predictions

# =====================================
# SAVE RESULTS
# =====================================

df.to_csv(OUTPUT_PATH, index=False)
print("\nResults saved to:", OUTPUT_PATH)

# =====================================
# METRICS
# =====================================

accuracy = accuracy_score(df[LABEL_COLUMN], df["prediction"])
f1 = f1_score(df[LABEL_COLUMN], df["prediction"])
precision = precision_score(df[LABEL_COLUMN], df["prediction"])
recall = recall_score(df[LABEL_COLUMN], df["prediction"])
cm = confusion_matrix(df[LABEL_COLUMN], df["prediction"])

print("\nEvaluation Results:")
print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

print("\nConfusion Matrix:")
print(cm)