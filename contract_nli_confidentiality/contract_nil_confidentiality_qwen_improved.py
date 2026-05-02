import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "contract_nli_confidentiality_test.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "qwen_confidentiality_results_improved.csv")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Loadind the dataset
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH, sep="\t")

TEXT_COLUMN = "text"
LABEL_COLUMN = "answer"

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower().map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

# Loading the model and tokenizer
print("\nLoading Qwen model...")

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

# Inputing the improved prompt template
def create_prompt(clause):
    return f"""
You are a legal expert.

Determine whether the following contract clause implies that the existence, terms, or discussions of the agreement must be kept confidential.

Confidentiality of the agreement includes:
- keeping the agreement itself secret
- not disclosing terms or existence
- restricting discussion of the contract

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

    answer_part = output.split("answer:")[-1].strip()

    if answer_part.startswith("yes"):
        predictions.append(1)
    else:
        predictions.append(0)

df["prediction"] = predictions

# Saving the results to CSV
df.to_csv(OUTPUT_PATH, index=False)

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