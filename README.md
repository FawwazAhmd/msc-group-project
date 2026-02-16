project:

&nbsp; title: "MSc Group Project – LLM-Based Legal Clause Classification"

&nbsp; author: "Fawwaz Ahmed"

&nbsp; university: "University of Liverpool"

&nbsp; degree: "MSc Computer Science"



overview:

&nbsp; description: >

&nbsp;   This project evaluates a Large Language Model (LLM) on a binary

&nbsp;   legal clause classification task (Yes/No). The model is prompted

&nbsp;   as a legal expert and must respond strictly with "Yes" or "No".

&nbsp; task\_type: "Binary Classification (Yes/No)"

&nbsp; input\_format: "TSV file"

&nbsp; output\_format: "CSV file with predictions"



model:

&nbsp; name: "microsoft/Phi-3-mini-4k-instruct"

&nbsp; provider: "Microsoft"

&nbsp; parameters: "3B"

&nbsp; framework: "HuggingFace Transformers"

&nbsp; precision: "float16"

&nbsp; device: "GPU (device\_map=auto)"

&nbsp; generation:

&nbsp;   max\_new\_tokens: 5

&nbsp;   do\_sample: false

&nbsp;   decoding: "Deterministic"



dataset:

&nbsp; file\_name: "test.tsv"

&nbsp; required\_columns:

&nbsp;   - text

&nbsp;   - answer\_or\_label

&nbsp; label\_mapping:

&nbsp;   yes: 1

&nbsp;   no: 0

&nbsp;   "1": 1

&nbsp;   "0": 0



prompt:

&nbsp; template: |

&nbsp;   You are a legal expert.

&nbsp;   Answer ONLY Yes or No.



&nbsp;   Clause:

&nbsp;   {clause}



&nbsp;   Answer:



pipeline:

&nbsp; steps:

&nbsp;   - Load dataset using pandas

&nbsp;   - Map labels to binary (0/1)

&nbsp;   - Load tokenizer and model

&nbsp;   - Create text-generation pipeline

&nbsp;   - Generate prediction for each clause

&nbsp;   - Extract Yes/No from output

&nbsp;   - Save predictions to CSV

&nbsp;   - Compute evaluation metrics



evaluation:

&nbsp; metrics:

&nbsp;   - Accuracy

&nbsp;   - F1 Score

&nbsp; library: "scikit-learn"

&nbsp; example\_results:

&nbsp;   accuracy: 0.4875

&nbsp;   f1\_score: 0.6555



installation:

&nbsp; environment\_setup:

&nbsp;   create\_venv: "python -m venv llm\_env"

&nbsp;   activate\_windows: "llm\_env\\\\Scripts\\\\activate"

&nbsp; dependencies:

&nbsp;   - pandas

&nbsp;   - torch

&nbsp;   - transformers

&nbsp;   - scikit-learn

&nbsp;   - tqdm

&nbsp; install\_command: "pip install -r requirements.txt"



execution:

&nbsp; run\_command: "python your\_script\_name.py"

&nbsp; requirements:

&nbsp;   - test.tsv must be in project directory

&nbsp;   - GPU recommended



outputs:

&nbsp; generated\_files:

&nbsp;   - results\_gpu.csv

&nbsp; console\_output:

&nbsp;   - Accuracy

&nbsp;   - F1 Score



future\_improvements:

&nbsp; - Improve prompt engineering

&nbsp; - Few-shot prompting

&nbsp; - Structured output parsing

&nbsp; - Larger model comparison



