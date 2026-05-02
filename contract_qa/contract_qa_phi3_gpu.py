import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "contract_qa_test.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "contract_qa_temperature_results.csv")
PLOT_PATH = os.path.join(BASE_DIR, "temperature_vs_f1.png")

# Stating the model name
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Temperature values to test
TEMPERATURES = [0.01, 0.2, 0.5, 0.7, 1.0]

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

Answer ONLY Yes or No.

Clause:
{clause}

Answer:
"""

# Temperature evaluation loop
results = []

for temp in TEMPERATURES:

    predictions = []

    print(f"\nRunning evaluation for temperature = {temp}...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        prompt = create_prompt(row[TEXT_COLUMN])

        output = pipe(
            prompt,
            max_new_tokens=5,
            do_sample=True,
            temperature=temp
        )[0]["generated_text"].lower()

        if "yes" in output:
            predictions.append(1)
        else:
            predictions.append(0)

    # Calculate metrics
    accuracy = accuracy_score(df[LABEL_COLUMN], predictions)
    f1 = f1_score(df[LABEL_COLUMN], predictions)
    precision = precision_score(df[LABEL_COLUMN], predictions)
    recall = recall_score(df[LABEL_COLUMN], predictions)

    results.append({
        "temperature": temp,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    })

# Saving the results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nTemperature tuning results saved to:", OUTPUT_PATH)
print("\nResults Summary:")
print(results_df)

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