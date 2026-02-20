import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# =====================================
# CONFIGURATION
# =====================================

DATASET_PATH = "contract_qa_test.tsv"
OUTPUT_PATH = "contract_qa_gpu_results.csv"

# Small but capable instruction model (3B parameters)
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# =====================================
# LOAD DATASET
# =====================================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH, sep="\t")

print("Columns:", df.columns.tolist())

TEXT_COLUMN = "text"
LABEL_COLUMN = "answer" if "answer" in df.columns else "label"

# Convert labels to 0/1
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower().map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

# =====================================
# LOAD MODEL (4-bit quantized)
# =====================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
  torch_dtype=torch.float16
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
    )[0]["generated_text"].lower()

    if "yes" in output:
        predictions.append(1)
    else:
        predictions.append(0)

df["phi3"] = predictions

# =====================================
# SAVE RESULTS
# =====================================

df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:", OUTPUT_PATH)

# =====================================
# COMPUTE METRICS
# =====================================

accuracy = accuracy_score(df[LABEL_COLUMN], df["phi3"])
f1 = f1_score(df[LABEL_COLUMN], df["phi3"])
precision = precision_score(df[LABEL_COLUMN], df["phi3"])
recall = recall_score(df[LABEL_COLUMN], df["phi3"])
cm = confusion_matrix(df[LABEL_COLUMN], df["phi3"])

print("\nEvaluation Results:")
print("Accuracy:", round(accuracy, 4))
print("F1 Score:", round(f1, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

print("\nConfusion Matrix:")
print(cm)