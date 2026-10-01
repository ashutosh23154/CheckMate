<div align="center">

# ♟️ CheckMate

### *Explainable Claim Check-Worthiness Detection via Rationality-Guided Multi-Task Learning*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![spaCy](https://img.shields.io/badge/spaCy-en__core__web__sm-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **CheckMate** is a joint deep learning model for *explainable* claim check-worthiness detection. Given a textual claim, it not only predicts whether the claim is **check-worthy**, but also surfaces **why** — via six structured rationality labels — and leverages **Supervised Contrastive Rationality Learning** to cluster semantically similar claims in representation space.

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [BERT Backbone](#1-bert-backbone)
  - [CoNet — Contextual Network](#2-conet--contextual-network)
  - [LiNet — Linguistic Network](#3-linet--linguistic-network)
  - [Rationality Head](#4-rationality-head)
  - [Check-Worthiness Head](#5-check-worthiness-head)
  - [Contrastive Projection Head](#6-contrastive-projection-head-crl)
- [Loss Functions](#loss-functions)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Evaluation](#evaluation)
- [Explainability Demo](#explainability-demo)
- [Rationality Labels](#rationality-labels)
- [Module Reference](#module-reference)
- [Citation](#citation)

---

## Overview

CheckMate implements the model described in:

> **"Leveraging Rationality Labels for Explainable Claim Check-Worthiness"**

The model is trained on the **CheckIt dataset** and solves two tasks simultaneously:

| Task | Output | Description |
|------|--------|-------------|
| **Primary** | `cw_prob ∈ [0, 1]` | Binary check-worthiness probability |
| **Auxiliary** | `rat_probs ∈ [0, 1]⁶` | Six rationality label probabilities (L1–L6) |
| **Contrastive** | `proj ∈ ℝ¹²⁸` | L2-normalised representation for contrastive learning (training only) |

The auxiliary rationality task acts as a structured **explainability layer**: the model's check-worthiness decision is grounded in and explained by six rationality categories, making CheckMate interpretable by design rather than post-hoc.

---

## Architecture

CheckMate is a **5-component joint model** with an optional contrastive projection head active only during training.

```
Raw Claim Text
      │
      ▼
┌─────────────────────────────┐
│   BERT-base-uncased         │  →  (batch, 128, 768)
│   Tokenized, max_length=128 │     last_hidden_state
└─────────────────────────────┘
      │                      spaCy NLP pipeline
      │                           │
      ▼                           ▼
┌──────────────┐       ┌──────────────────────┐
│    CoNet     │       │        LiNet          │
│  6 MHA Blocks│       │  POS (50) + DEP (50) │
│  + Self-Attn │       │  + Emoticon (1) = 101│
│ (batch,6,768)│       │  → Linear → (batch,64)│
└──────────────┘       └──────────────────────┘
      │                           │
      ▼                           │
┌─────────────┐                   │
│ Rationality │  → rat_probs      │
│    Head     │    (batch, 6)     │
│ Linear(768→1│                   │
│ ) × 6 labels│                   │
└─────────────┘                   │
      │                           │
      ▼                           │
┌────────────────────────────────────────┐
│   Flatten CoNet: (batch, 4608)         │
│   Concat LiNet:  (batch, 4672)         │
│   MLP1 → Linear(4672 → 256) + ReLU    │
│   MLP2 → Linear(256  → 1)   + Sigmoid │
└────────────────────────────────────────┘
      │
      ▼
  cw_prob (batch,)

  [During Training Only]
  ┌──────────────────────────────┐
  │   Contrastive Projection     │
  │   Linear(4608 → 256) + ReLU  │
  │   Linear(256  → 128)         │
  │   L2-normalise               │
  └──────────────────────────────┘
        proj (batch, 128)
```

---

### 1. BERT Backbone

- **Model**: `bert-base-uncased` (HuggingFace Transformers)
- **Input**: Tokenized claim text, `max_length = 128`, with padding and truncation
- **Output**: `last_hidden_state` of shape `(batch, 128, 768)` — full contextual sequence embeddings
- **Dropout**: 0.1 applied to the sequence output before passing into CoNet

---

### 2. CoNet — Contextual Network

**File:** [`co_net.py`](co_net.py)

The **Contextual Network** encodes rationality-aware contextual representations from the BERT output.

| Component | Details |
|-----------|---------|
| **MHA Blocks** | 6 parallel `nn.MultiheadAttention` layers (one per rationality label L1–L6) |
| **Attention heads** | 6 heads per MHA block, `embed_dim = 768`, `dropout = 0.1` |
| **Pooling** | Mean-pooling across the sequence dimension: `(batch, 128, 768) → (batch, 768)` |
| **Stacking** | 6 block outputs stacked along a new label dim: `(batch, 6, 768)` |
| **Self-Attention** | A final MHA layer aggregates across the 6 rationality-label representations |
| **Output** | `aggregated: (batch, 6, 768)` |

Each MHA block attends to the full BERT sequence through the "lens" of one rationality label, learning which token-level patterns are relevant to that label's semantics.

---

### 3. LiNet — Linguistic Network

**File:** [`li_net.py`](li_net.py)

The **Linguistic Network** is a lightweight feed-forward module that captures surface-level syntactic and pragmatic signals that BERT alone may not model explicitly.

| Feature Stream | Source | Dimension |
|---------------|--------|-----------|
| POS Tags | spaCy `token.pos` (mod 10000), padded/truncated to 50 | 50 |
| Dependency Parse | spaCy `token.dep` (mod 10000), padded/truncated to 50 | 50 |
| Emoticon Count | Unicode category `So`/`Sm` character count | 1 |
| **Total input** | Concatenated | **101** |
| **Output** | `Linear(101 → 64)` | **64** |

---

### 4. Rationality Head

- Applies a shared `nn.Linear(768 → 1)` across each of the 6 CoNet block outputs
- Produces `rat_logits: (batch, 6)` → `rat_probs: (batch, 6)` via `torch.sigmoid`
- Each scalar is an independent binary probability for one rationality label (L1–L6)
- Provides the structured explainability output of the model

---

### 5. Check-Worthiness Head

- **Input**: CoNet output flattened to `(batch, 4608)` concatenated with LiNet output `(batch, 64)` → `(batch, 4672)`
- **MLP1**: `Linear(4672 → 256)` + ReLU + Dropout(0.1)
- **MLP2**: `Linear(256 → 1)` + Sigmoid
- **Output**: `cw_prob: (batch,)` — scalar check-worthiness probability in [0, 1]
- Threshold `0.5` is used for binary classification

---

### 6. Contrastive Projection Head (CRL)

> **Novel contribution** — Contrastive Rationality Learning (CRL)

A 2-layer MLP projection head maps the flattened CoNet representation into a compact 128-dim space where **Supervised Contrastive Loss** is applied:

```
(batch, 4608)
    → Linear(4608 → 256) → ReLU
    → Linear(256  → 128)
    → L2-normalize
    → (batch, 128)  ← unit sphere
```

- The projected vectors are **L2-normalised** to enable cosine-similarity-based contrastive learning
- **Only used during training** — discarded entirely at inference time
- Based on: *Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020*

---

## Loss Functions

CheckMate is trained with a **three-term composite loss**:

```
L_total = L_cw + L_rat + λ · L_cl
```

| Term | Formula | Description |
|------|---------|-------------|
| `L_cw` | `BCE(cw_prob, label)` | Primary check-worthiness supervision |
| `L_rat` | `Σᵢ BCE(rat_prob[:, i], rat_label[:, i])` | Auxiliary multi-label rationality supervision (sum over L1–L6) |
| `L_cl` | Supervised Contrastive Loss | Pulls same-rationale claims together in projection space |
| **λ** | `0.1` | Contrastive loss weighting factor |

### Supervised Contrastive Loss (CRL)

Two claims are **positive pairs** if their rationality label vectors share ≥ 1 active label (they belong to the same "rationale family"). The loss computation:

1. Computes pairwise cosine-similarity matrix (dot product of L2-normalised projections) scaled by temperature `τ = 0.07`
2. Applies numerical stability: subtracts row-wise max before exponentiation
3. Builds a positive-pair mask: `label_overlap = rat_labels @ rat_labels.T > 0`, excluding self-pairs
4. Computes log-probability for every potential positive pair
5. Averages the loss over all positive pairs per anchor
6. Only includes anchors that have at least one positive pair in the current batch

This forces the model to cluster claims by *intent* before making the final check-worthiness decision, improving generalisation on unseen data.

---

## Dataset

CheckMate uses the **CheckIt dataset** in CSV format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `claim` | `str` | Raw claim text |
| `check_worthy_label` | `float` | Binary check-worthiness label (0 or 1) |
| `verifiable_factual_claim` | `float` | **L1**: Is there a verifiable factual claim? |
| `false_info` | `float` | **L2**: Does it contain false information? |
| `general_public_interest` | `float` | **L3**: Will it impact the general public? |
| `harmful` | `float` | **L4**: Is it harmful to society? |
| `fact_checker_interest` | `float` | **L5**: Should it be verified by a fact-checker? |
| `govt_interest` | `float` | **L6**: Should it get government attention? |

**Data splits:**

| Split | File | Approx. Size |
|-------|------|-------------|
| Train | `train.csv` | ~1.2 MB |
| Dev (Validation) | `dev.csv` | ~246 KB |
| Test | `test.csv` | ~243 KB |

**Preprocessing pipeline (per claim):**

1. BERT tokenization — `bert-base-uncased`, `max_length=128`, padding + truncation
2. spaCy POS tagging — 50-dim integer vector (padded/truncated)
3. spaCy dependency parsing — 50-dim integer vector (padded/truncated)
4. Emoticon counting — Unicode category `So`/`Sm` scan → 1-dim float
5. Rationality labels — NaN → `0.0`, clamped to `[0.0, 1.0]`

> **Note**: If rationality label columns are absent from a CSV, the preprocessor emits a warning and defaults all auxiliary labels to `0.0`, allowing the model to still train on the primary task.

---

## Project Structure

```
CheckMate/
│
├── checkmate.py        # Main model: CheckMate joint architecture
├── co_net.py           # CoNet: 6-block MHA + self-attention contextual encoder
├── li_net.py           # LiNet: POS + DEP + emoticon linguistic feature encoder
├── preprocess.py       # Data loading, BERT tokenization, spaCy feature extraction
├── train.py            # Training loop, loss functions, evaluation, explainability demo
├── utils.py            # Utility: sinusoidal positional encoding
│
├── train.csv           # Training split of the CheckIt dataset
├── dev.csv             # Development / validation split
├── test.csv            # Test split
│
├── README.md           # This file
└── Project Report.pdf  # Full project report
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU training is supported but slow)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CheckMate.git
cd CheckMate
```

### 2. Install Python dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn pandas spacy
```

### 3. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4. BERT weights

The `bert-base-uncased` checkpoint (~440 MB) is **automatically downloaded** from HuggingFace on the first run of `preprocess.py` or `train.py`. No manual step required.

---

## Usage

### Training & Evaluation

```bash
python train.py
```

This single command:
1. Loads and preprocesses all three CSV splits (`train`, `dev`, `test`)
2. Initializes **CheckMate** and moves it to GPU (falls back to CPU if unavailable)
3. Trains for **10 epochs** with AdamW, printing per-epoch average loss
4. Evaluates on the **dev set** (validation metrics)
5. Evaluates on the **test set** (final metrics)
6. Runs an **Explainability Demo** on the first test batch

**Sample console output:**

```
Using device: cuda
Loading datasets...
Processing train set...
Processing test set...
Processing dev set...
Initializing model...
Starting training...
Epoch [1/10], Loss: 0.7812
Epoch [2/10], Loss: 0.6543
...
Epoch [10/10], Loss: 0.3201
Evaluating on dev set...
Validation -> Acc: 82.14% | Prec: 79.33% | Rec: 85.67% | F1: 82.38%
Evaluating on test set...
Test       -> Acc: 81.90% | Prec: 78.91% | Rec: 84.22% | F1: 81.48%

--- EXPLAINABILITY DEMO ---
Model Prediction: Check-Worthy
Check-Worthiness Score: 87.34%
Underlying Rationality (Explainability):
  - L1 - Verifiable Factual Claim: 91.20%
  - L2 - Contains False Information: 76.45%
  - L3 - General Public Interest: 83.11%
  - L4 - Harmful to Society: 61.89%
  - L5 - Needs Fact-Checker Verification: 88.34%
  - L6 - Needs Govt Attention: 42.00%
```

---

## Hyperparameters

| Hyperparameter | Value | Source |
|---------------|-------|--------|
| Epochs | `10` | Paper |
| Batch size | `16` | Paper |
| Learning rate | `2e-5` | Paper (AdamW BERT fine-tuning standard) |
| Weight decay | `0.01` | Paper |
| Adam β₁ | `0.9` | Paper |
| Adam β₂ | `0.999` | Paper |
| Adam ε | `1e-8` | Paper |
| Dropout | `0.1` | Paper |
| BERT max sequence length | `128` | Paper |
| Contrastive temperature τ | `0.07` | Khosla et al. (2020) |
| Contrastive loss weight λ | `0.1` | Novel (CRL contribution) |
| BERT embedding size | `768` | BERT-base-uncased architecture |
| CoNet MHA blocks | `6` | Paper (one per rationality label) |
| CoNet heads per block | `6` | Paper |
| LiNet input dim | `101` | 50 POS + 50 DEP + 1 emoticon |
| LiNet output dim | `64` | Paper |
| Combined dim (CW head input) | `4672` | 6 × 768 (CoNet) + 64 (LiNet) |
| CW head hidden dim | `256` | Paper |
| Projection head output dim | `128` | Novel (CRL contribution) |
| Check-worthy threshold | `0.5` | Standard binary classification |

---

## Evaluation

The model is evaluated on four standard binary classification metrics computed with `sklearn.metrics`:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP + TN) / Total | Overall fraction of correct predictions |
| **Precision** | TP / (TP + FP) | How many predicted check-worthy claims are truly check-worthy |
| **Recall** | TP / (TP + FN) | How many truly check-worthy claims are retrieved |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

All metrics are reported as percentages. `zero_division=0` is used to handle edge cases gracefully. Evaluation runs under `torch.no_grad()` for memory efficiency, and the contrastive projection output is discarded during evaluation.

---

## Explainability Demo

After training, CheckMate runs a built-in explainability demo on the first batch of the test set. For each claim, it produces:

1. **Model Prediction**: `Check-Worthy` or `Non Check-Worthy`
2. **Check-Worthiness Score**: `cw_prob × 100` (%)
3. **Rationality Breakdown** — individual probability scores for each of the 6 rationality dimensions (L1–L6)

This makes CheckMate interpretable **by design**, not post-hoc: the check-worthiness decision is directly grounded in human-understandable rationality dimensions, enabling journalists, fact-checkers, and researchers to understand *why* a claim is flagged.

---

## Rationality Labels

CheckMate predicts six binary rationality labels that correspond to distinct, structured reasons why a claim may be check-worthy:

| Label | Name | Question |
|-------|------|----------|
| **L1** | Verifiable Factual Claim | Does the claim assert a verifiable fact? |
| **L2** | Contains False Information | Does the claim appear to contain false or misleading information? |
| **L3** | General Public Interest | Does the claim concern issues that affect the general public? |
| **L4** | Harmful to Society | Could the claim cause societal harm if left unverified? |
| **L5** | Needs Fact-Checker Verification | Should a professional fact-checker investigate this claim? |
| **L6** | Needs Government Attention | Does the claim require attention from government or regulatory bodies? |

These labels are sourced from the **CheckIt dataset** columns: `verifiable_factual_claim`, `false_info`, `general_public_interest`, `harmful`, `fact_checker_interest`, and `govt_interest`.

---

## Module Reference

### `CheckMate` — [`checkmate.py`](checkmate.py)

```python
CheckMate(
    embed_size=768,       # BERT hidden size
    num_heads=6,          # Attention heads per MHA block
    num_rat_labels=6,     # Number of rationality labels (= number of MHA blocks)
    ling_input_dim=101,   # 50 POS + 50 DEP + 1 emoticon
    ling_output_dim=64,   # LiNet output dimension
)
```

**Forward signature:**

```python
cw_prob, rat_probs, proj = model(
    input_ids,       # (batch, 128)  LongTensor — BERT token IDs
    attention_mask,  # (batch, 128)  LongTensor — BERT attention mask
    pos_tags,        # (batch, 50)   FloatTensor — POS tag features
    dep_parsing,     # (batch, 50)   FloatTensor — dependency parse features
    emoticons,       # (batch, 1)    FloatTensor — emoticon count
)
```

| Return | Shape | Description |
|--------|-------|-------------|
| `cw_prob` | `(batch,)` | Check-worthiness probability ∈ [0, 1] |
| `rat_probs` | `(batch, 6)` | Per-label rationality probabilities ∈ [0, 1] |
| `proj` | `(batch, 128)` | L2-normalised contrastive projection (training only) |

---

### `CoNet` — [`co_net.py`](co_net.py)

```python
CoNet(embed_size=768, num_heads=6, num_rat_labels=6)
```

```
forward(x: Tensor[batch, 128, 768]) → Tensor[batch, 6, 768]
```

---

### `LiNet` — [`li_net.py`](li_net.py)

```python
LiNet(input_dim=101, output_dim=64)
```

```
forward(pos_tags, dep_parsing, emoticons) → Tensor[batch, 64]
```

---

### `SupervisedContrastiveLoss` — [`train.py`](train.py)

```python
SupervisedContrastiveLoss(temperature=0.07)
```

```
forward(
    projections: Tensor[batch, 128],  # L2-normalised contrastive embeddings
    rat_labels:  Tensor[batch, 6],    # Binary rationality label vectors
) → scalar Tensor
```

---

### `preprocess_data` / `load_datasets` — [`preprocess.py`](preprocess.py)

```python
# Load and preprocess a single CSV file
tensors = preprocess_data("train.csv")

# Load all three splits and return PyTorch Dataset objects
train_dataset, test_dataset, dev_dataset = load_datasets(
    "train.csv", "test.csv", "dev.csv"
)
```

---

### `positional_encoding` — [`utils.py`](utils.py)

```python
pe = positional_encoding(seq_len=128, embed_dim=768)
# Returns: Tensor[seq_len, embed_dim] — sinusoidal positional encodings
```

Implements the standard sinusoidal positional encoding from *Vaswani et al., "Attention Is All You Need", NeurIPS 2017*.

---

## Citation

If you use CheckMate in your research, please cite the original paper and this implementation:

```bibtex
@misc{checkmate2026,
  title        = {CheckMate: Explainable Claim Check-Worthiness via Rationality-Guided Multi-Task Learning},
  author       = {Mahen},
  year         = {2026},
  note         = {Implementation of "Leveraging Rationality Labels for Explainable Claim Check-Worthiness"},
  howpublished = {\url{https://github.com/<your-username>/CheckMate}},
}
```

```bibtex
@inproceedings{khosla2020supervised,
  title     = {Supervised Contrastive Learning},
  author    = {Khosla, Prannay and Tian, Yonglong and Wang, Yueqi and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020}
}
```

---

<div align="center">

Made with ♟️ for explainable NLP

</div>
