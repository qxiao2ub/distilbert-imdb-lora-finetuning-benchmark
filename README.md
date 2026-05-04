# DistilBERT IMDB LoRA Fine-Tuning Benchmark

**Recommended GitHub repo name:** `distilbert-imdb-lora-finetuning-benchmark`

**Short description:** Benchmark full fine-tuning versus LoRA parameter-efficient fine-tuning for DistilBERT on IMDB binary sentiment classification.

## GitHub topics / keywords

`distilbert`, `imdb`, `sentiment-analysis`, `lora`, `peft`, `transformers`, `huggingface`, `fine-tuning`, `binary-classification`, `pytorch`

## Project overview

This repository contains a Colab-ready notebook for comparing two DistilBERT-based approaches for IMDB movie-review sentiment classification:

1. **Full fine-tuning** - all DistilBERT parameters and the classification head are trainable.
2. **LoRA PEFT fine-tuning** - low-rank adapters are trained on selected attention projections while keeping most base-model weights frozen.

The main goal is to show how parameter-efficient fine-tuning can approach full fine-tuning performance while training far fewer parameters.

## Source notebook

- `answer_for_problem1.ipynb`

## Dataset

The notebook uses the Hugging Face `imdb` dataset:

- Task: binary sentiment classification
- Labels: negative / positive
- Training source: `imdb["train"]`
- Test source: full `imdb["test"]`
- Training subsample: 10,000 examples
- Validation split: 10 percent of the training subsample
- Test set: full 25,000-example IMDB test split

## Methodology

### Base model

- Model checkpoint: `distilbert-base-uncased`
- Tokenization: DistilBERT tokenizer
- Maximum sequence length: 256
- Task: sequence classification with two labels

### Full fine-tuning setup

| Hyperparameter | Value |
|---|---:|
| Learning rate | 2e-5 |
| Epochs | 3 |
| Train batch size | 16 |
| Eval batch size | 64 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

### LoRA setup

| Hyperparameter | Value |
|---|---:|
| Base model | `distilbert-base-uncased` |
| Target modules | `q_lin`, `v_lin` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.10 |
| Learning rate | 5e-4 |
| Epochs | 3 |
| Train batch size | 16 |
| Eval batch size | 64 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

## Test-set results

| Model | Trainable parameters | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning | 66,955,010 | 0.9037 | 0.8986 | 0.9100 | 0.9043 |
| LoRA | 739,586 | 0.8992 | 0.8931 | 0.9069 | 0.8999 |

## Key finding

LoRA reaches nearly the same test-set performance as full fine-tuning while using only about **1.1 percent** as many trainable parameters. This makes it attractive for resource-constrained experimentation, rapid model iteration, and deployment scenarios where training memory and storage footprint matter.

## Repository structure

```text
.
├── answer_for_problem1.ipynb
├── README.md
└── requirements.txt
```

A minimal `requirements.txt` can contain:

```text
datasets
transformers
accelerate
peft
scikit-learn
pandas
matplotlib
torchao
```

## How to run

### Option 1: Google Colab

1. Upload `answer_for_problem1.ipynb` to Google Colab.
2. Select a GPU runtime.
3. Run the installation cell if needed.
4. Execute the notebook from top to bottom.

### Option 2: Local environment

```bash
git clone https://github.com/<your-username>/distilbert-imdb-lora-finetuning-benchmark.git
cd distilbert-imdb-lora-finetuning-benchmark

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -U pip
pip install -r requirements.txt
jupyter notebook
```

Then open `answer_for_problem1.ipynb` and run all cells.

## Main outputs

The notebook produces:

- IMDB train/validation/test splits
- Tokenized datasets
- Full fine-tuning training log
- LoRA PEFT training log
- Loss curves
- Final test-set comparison table
- Accuracy, precision, recall, and F1 metrics

## Notes

- The notebook is designed for instructional benchmarking rather than production deployment.
- Results can vary slightly depending on GPU type, package versions, random seeds, and training environment.
- The Hugging Face dataset and pretrained model are downloaded at runtime.

## Suggested future improvements

- Add more PEFT methods such as prefix tuning, prompt tuning, and IA3.
- Evaluate runtime, GPU memory, and checkpoint size in addition to predictive metrics.
- Test larger models such as BERT-base, RoBERTa-base, and DeBERTa-v3.
- Add error analysis for misclassified reviews.
- Export the trained LoRA adapter for inference-only deployment.

## License


