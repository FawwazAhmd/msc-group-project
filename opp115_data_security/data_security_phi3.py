import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "opp115_data_security_test.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "security_phi3.csv")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Loadind the dataset
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH, sep="\t")

print("Columns:", df.columns.tolist())

TEXT_COLUMN = "text"
LABEL_COLUMN = "answer" if "answer" in df.columns else "label"

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower().map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

# Loading the model and tokenizer
print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# Inputing the prompt template
def create_prompt(clause):
    return f"""
You are a legal expert.

Determine whether the following privacy policy clause explains how user data is protected or secured.

This includes mentions of security measures such as encryption, safeguards, protection methods, or access control.

Answer ONLY Yes or No. Do not explain.

Clause:
{clause}

Answer:
"""

# Temperature evaluation loop
predictions = []

print("\nRunning evaluation...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    prompt = create_prompt(row[TEXT_COLUMN])

    output = pipe(
        prompt,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"].lower()

    if "yes" in output:
        predictions.append(1)
    else:
        predictions.append(0)

df["prediction"] = predictions

# Saving the results to CSV
df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:", OUTPUT_PATH)

# Metrics
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