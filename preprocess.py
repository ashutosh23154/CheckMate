import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import spacy
import unicodedata

# Load spaCy for linguistic features
nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer (paper: BERT-base-uncased)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Constants from paper
MAX_SEQ_LENGTH = 128
MAX_LING_LEN = 50

# Rationality label columns in the CheckIt dataset
RAT_COLS = [
    "verifiable_factual_claim",   # L1: Is there a verifiable factual claim?
    "false_info",                  # L2: Does it contain false information?
    "general_public_interest",     # L3: Will it impact general public?
    "harmful",                     # L4: Is it harmful to society?
    "fact_checker_interest",       # L5: Should it be verified by a fact-checker?
    "govt_interest",               # L6: Should it get govt attention?
]


def count_emoticons(text):
    """Returns count of emoticon/emoji characters in text (as a feature)."""
    count = 0
    for char in text:
        cat = unicodedata.category(char)
        # 'So' = Symbol, other (includes emoji); 'Sm' = Symbol, math
        if cat.startswith("So") or cat.startswith("Sm"):
            count += 1
    return float(count)


def preprocess_data(file_path):
    """Preprocess a CSV file into tensors for the CheckMate model."""
    data = pd.read_csv(file_path)

    # Check which rationality label columns are available
    available_rat_cols = [col for col in RAT_COLS if col in data.columns]
    has_rat_labels = len(available_rat_cols) == len(RAT_COLS)
    if not has_rat_labels:
        print(f"  [Warning] Not all rationality columns found in {file_path}. "
              f"Found: {available_rat_cols}. Auxiliary labels will be zero.")

    input_ids_list = []
    attention_masks_list = []
    pos_tags_list = []
    dep_parsing_list = []
    emoticon_list = []
    labels_list = []
    rat_labels_list = []

    for _, row in data.iterrows():
        claim = str(row["claim"])
        # Clamp check_worthy_label to [0, 1] and handle NaN
        cw_val = row["check_worthy_label"]
        cw_val = 0.0 if pd.isna(cw_val) else float(max(0.0, min(1.0, cw_val)))
        label = torch.tensor(cw_val, dtype=torch.float32)

        # ── BERT Tokenization (paper: BERT-base-uncased, max_length=128) ──
        encoding = tokenizer(
            claim,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)        # (128,) LongTensor
        attention_mask = encoding["attention_mask"].squeeze(0)  # (128,) LongTensor

        # ── POS Tags via spaCy ──
        doc = nlp(claim)
        pos_list = [token.pos % 10000 for token in doc]
        while len(pos_list) < MAX_LING_LEN:
            pos_list.append(0)
        pos_vector = torch.tensor(pos_list[:MAX_LING_LEN], dtype=torch.float32)

        # ── Dependency Parse via spaCy ──
        dep_list = [token.dep % 10000 for token in doc]
        while len(dep_list) < MAX_LING_LEN:
            dep_list.append(0)
        dep_vector = torch.tensor(dep_list[:MAX_LING_LEN], dtype=torch.float32)

        # ── Emoticon Feature (paper: LiNet includes emoticon features) ──
        emoticon_vector = torch.tensor([count_emoticons(claim)], dtype=torch.float32)

        # ── Rationality Labels L1–L6 (NaN → 0.0, clamp to [0.0, 1.0]) ──
        if has_rat_labels:
            rat = []
            for col in RAT_COLS:
                val = row[col]
                val = 0.0 if pd.isna(val) else float(max(0.0, min(1.0, val)))
                rat.append(val)
        else:
            rat = [0.0] * 6
        rat_tensor = torch.tensor(rat, dtype=torch.float32)

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        pos_tags_list.append(pos_vector)
        dep_parsing_list.append(dep_vector)
        emoticon_list.append(emoticon_vector)
        labels_list.append(label)
        rat_labels_list.append(rat_tensor)

    return (
        input_ids_list,
        attention_masks_list,
        pos_tags_list,
        dep_parsing_list,
        emoticon_list,
        labels_list,
        rat_labels_list,
    )


class ClaimDataset(Dataset):
    def __init__(self, input_ids, attention_masks, pos_tags,
                 dep_parsing, emoticons, labels, rat_labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.pos_tags = pos_tags
        self.dep_parsing = dep_parsing
        self.emoticons = emoticons
        self.labels = labels
        self.rat_labels = rat_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.pos_tags[idx],
            self.dep_parsing[idx],
            self.emoticons[idx],
            self.labels[idx],
            self.rat_labels[idx],
        )


def load_datasets(train_file, test_file, dev_file):
    print("Processing train set...")
    train_data = preprocess_data(train_file)
    print("Processing test set...")
    test_data = preprocess_data(test_file)
    print("Processing dev set...")
    dev_data = preprocess_data(dev_file)

    train_dataset = ClaimDataset(*train_data)
    test_dataset = ClaimDataset(*test_data)
    dev_dataset = ClaimDataset(*dev_data)

    return train_dataset, test_dataset, dev_dataset
