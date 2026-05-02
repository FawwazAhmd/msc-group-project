import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

from contract_qa.contract_qa_tinyllama_gpu import PLOT_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "contract_qa_test.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "contract_qa_cpu_results2.csv")

# Stating the model name
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Loadind the dataset
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH, sep="\t")

print("Columns:", df.columns.tolist())

TEXT_COLUMN = "text"
LABEL_COLUMN = "answer" if "answer" in df.columns else "label"

# Converting the labels to binary
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower().map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0
})

# Loading the model and tokenizer
print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    torch_dtype=torch.float32
)


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)


print("Model loaded successfully.")

# Inputing the prompt template
def create_prompt(clause):

    return f"""
You are a legal expert.

Answer ONLY Yes or No.

Clause:
{clause}

Answer:
"""

# Running evaluation
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

# Saving the results to CSV
df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved to:", OUTPUT_PATH)

# Metrics
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

# Plotting Temperature vs F1 Score
plt.figure(figsize=(8, 5))
plt.plot(results_df["temperature"], results_df["f1_score"], marker="o", linewidth=2)
plt.xlabel("Temperature")
plt.ylabel("F1 Score")
plt.title("Temperature vs F1 Score (Contract QA)")
plt.grid(True)
plt.savefig(PLOT_PATH)
plt.show()

print("\nPlot saved to:", PLOT_PATH)
