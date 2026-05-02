import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "contract_qa_test.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "contract_qa_temperature_results_tinyllama.csv")
PLOT_PATH = os.path.join(BASE_DIR, "temperature_vs_f1_tinyllama.png")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TEMPERATURES = [0.01, 0.2, 0.5, 0.7, 1.0]

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

print("Dataset loaded:", len(df))

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Running on CPU.")

# Loading the model and tokenizer
print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )[0]["generated_text"]

        generated_answer = output[len(prompt):].strip().lower()

        if generated_answer.startswith("yes"):
            predictions.append(1)
        else:
            predictions.append(0)

    # Metrics
    accuracy = accuracy_score(df[LABEL_COLUMN], predictions)
    f1 = f1_score(df[LABEL_COLUMN], predictions)
    precision = precision_score(df[LABEL_COLUMN], predictions)
    recall = recall_score(df[LABEL_COLUMN], predictions)
    cm = confusion_matrix(df[LABEL_COLUMN], predictions)

    print(f"\nConfusion Matrix for temperature = {temp}:")
    print(cm)

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

print("\nSaved CSV:", OUTPUT_PATH)
print("\nResults Summary:")
print(results_df)

# Plotting Temperature vs F1 Score
plt.figure(figsize=(8, 5))
plt.plot(results_df["temperature"], results_df["f1_score"], marker="o", linewidth=2)
plt.xlabel("Temperature")
plt.ylabel("F1 Score")
plt.title("Temperature vs F1 Score (TinyLlama Contract QA)")
plt.grid(True)
plt.savefig(PLOT_PATH)
plt.show()

print("\nPlot saved:", PLOT_PATH)