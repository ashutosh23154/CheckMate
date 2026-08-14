import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from checkmate import CheckMate
from preprocess import load_datasets


# ════════════════════════════════════════════════════════════
# Supervised Contrastive Loss  (NEW — Idea A: CRL)
# ════════════════════════════════════════════════════════════
class SupervisedContrastiveLoss(nn.Module):
    """
    Contrastive Rationality Learning (CRL).

    Pulls together L2-normalised embeddings of claims that SHARE at least
    one rationality label, and pushes apart claims with NO shared labels.

    This forces the model to cluster claims by their *intent* (harmful,
    verifiable, etc.) before making the final check-worthiness decision,
    improving generalisation on unseen test data.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, rat_labels):
        """
        Args:
            projections : (batch, 128)  — L2-normalised contrastive embeddings
            rat_labels  : (batch, 6)    — binary rationality label vectors
        Returns:
            scalar contrastive loss
        """
        batch_size = projections.size(0)
        device     = projections.device

        # ─ Pairwise cosine-similarity matrix scaled by temperature ─
        # (already L2-normalised, so dot product = cosine similarity)
        sim = torch.matmul(projections, projections.T) / self.temperature  # (B, B)

        # Numerical stability: subtract row-wise max before exp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # ─ Positive-pair mask ─
        # Two claims are "positives" if their rationality vectors share ≥1 label
        label_overlap = torch.matmul(rat_labels, rat_labels.T)   # (B, B)
        pos_mask = (label_overlap > 0).float()                   # 1 = same rationale family

        # Remove self-pairs from the mask and from the denominator
        diag     = torch.eye(batch_size, device=device)
        pos_mask = pos_mask * (1 - diag)                         # no self-positives

        # ─ Log-denominator: sum over all non-self pairs ─
        exp_sim   = torch.exp(sim) * (1 - diag)                  # zero out diagonal
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # (B, 1)

        # ─ Log-probability for every potential positive pair ─
        log_prob = sim - log_denom                               # (B, B) broadcast

        # ─ Average loss over positive pairs per anchor ─
        num_pos = pos_mask.sum(dim=1).clamp(min=1)              # (B,)
        loss_per_sample = -(pos_mask * log_prob).sum(dim=1) / num_pos  # (B,)

        # Only include anchors that actually have a positive pair in this batch
        has_pos = (pos_mask.sum(dim=1) > 0).float()
        loss = (loss_per_sample * has_pos).sum() / has_pos.sum().clamp(min=1)
        return loss

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters (exact values from paper) ──
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5       # AdamW fine-tuning lr for BERT
WEIGHT_DECAY = 0.01        # As specified in paper
EPSILON = 1e-8             # Adam epsilon
BETAS = (0.9, 0.999)       # Adam betas
NUM_RAT_LABELS = 6         # Rationality labels L1–L6

# Contrastive loss weight λ (NEW)
# Scales how strongly the contrastive signal influences training.
# 0.1 = contrastive loss contributes ~10% of total gradient signal.
CONTRASTIVE_WEIGHT = 0.1

# File paths
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
DEV_FILE = "dev.csv"

# ── Load datasets ──
print("Loading datasets...")
train_dataset, test_dataset, dev_dataset = load_datasets(TRAIN_FILE, TEST_FILE, DEV_FILE)

# ── DataLoaders ──
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ── Initialize CheckMate (paper parameters) ──
print("Initializing model...")
model = CheckMate(
    embed_size=768,          # BERT-base-uncased hidden size
    num_heads=6,             # 6 attention heads per MHA block
    num_rat_labels=6,        # One MHA block per rationality label L1–L6
    ling_input_dim=101,      # 50 POS + 50 DEP + 1 emoticon
    ling_output_dim=64,
).to(device)

# ── Loss functions ──
bce_loss  = nn.BCELoss()
con_loss  = SupervisedContrastiveLoss(temperature=0.07)  # NEW

# ── Optimizer: AdamW (paper: β1=0.9, β2=0.999, ε=1e-8, wd=0.01) ──
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=BETAS,
    eps=EPSILON,
    weight_decay=WEIGHT_DECAY,
)

# ── Training loop ──
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, pos_tags, dep_parsing, \
            emoticons, labels, rat_labels = batch

        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pos_tags       = pos_tags.to(device)
        dep_parsing    = dep_parsing.to(device)
        emoticons      = emoticons.to(device)
        labels         = labels.to(device)          # (batch,) float binary
        rat_labels     = rat_labels.to(device)      # (batch, 6) float binary

        # Forward pass — now returns 3 outputs
        cw_probs, rat_probs, proj = model(
            input_ids, attention_mask, pos_tags, dep_parsing, emoticons
        )

        # Primary loss: BCE on check-worthiness (paper)
        loss_cw = bce_loss(cw_probs, labels)

        # Auxiliary losses: BCE on each of the 6 rationality labels (paper)
        loss_rat = sum(
            bce_loss(rat_probs[:, i], rat_labels[:, i])
            for i in range(NUM_RAT_LABELS)
        )

        # Contrastive loss: pulls same-rationale claims together (NEW)
        loss_cl = con_loss(proj, rat_labels)

        # Total loss = paper loss + weighted contrastive term
        loss = loss_cw + loss_rat + CONTRASTIVE_WEIGHT * loss_cl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")


# ── Evaluation ──
def evaluate(model, data_loader):
    """Evaluate model accuracy, precision, recall, and F1 score."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, pos_tags, dep_parsing, \
                emoticons, labels, rat_labels = batch

            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pos_tags       = pos_tags.to(device)
            dep_parsing    = dep_parsing.to(device)
            emoticons      = emoticons.to(device)

            cw_probs, _, _ = model(    # discard rat_probs and proj
                input_ids, attention_mask, pos_tags, dep_parsing, emoticons
            )

            predicted = (cw_probs >= 0.5).long().cpu().numpy()
            true_labels = labels.long().cpu().numpy()

            all_preds.extend(predicted)
            all_labels.extend(true_labels)

    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    return acc, prec, rec, f1


print("Evaluating on dev set...")
dev_acc, dev_prec, dev_rec, dev_f1 = evaluate(model, dev_loader)
print(f"Validation -> Acc: {dev_acc:.2f}% | Prec: {dev_prec:.2f}% | Rec: {dev_rec:.2f}% | F1: {dev_f1:.2f}%")

print("Evaluating on test set...")
test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader)
print(f"Test       -> Acc: {test_acc:.2f}% | Prec: {test_prec:.2f}% | Rec: {test_rec:.2f}% | F1: {test_f1:.2f}%")


# ── Explainability Demo ──
print("\n--- EXPLAINABILITY DEMO ---")
model.eval()
with torch.no_grad():
    sample_batch = next(iter(test_loader))
    input_ids, attention_mask, pos_tags, dep_parsing, \
        emoticons, labels, rat_labels = sample_batch

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pos_tags       = pos_tags.to(device)
    dep_parsing    = dep_parsing.to(device)
    emoticons      = emoticons.to(device)

    cw_probs, rat_probs, _ = model(    # discard proj during demo
        input_ids, attention_mask, pos_tags, dep_parsing, emoticons
    )
    predicted_label = (cw_probs[0] >= 0.5).long().item()

    labels_map = ["Non Check-Worthy", "Check-Worthy"]
    causes_names = [
        "L1 - Verifiable Factual Claim",
        "L2 - Contains False Information",
        "L3 - General Public Interest",
        "L4 - Harmful to Society",
        "L5 - Needs Fact-Checker Verification",
        "L6 - Needs Govt Attention",
    ]

    print(f"Model Prediction: {labels_map[predicted_label]}")
    print(f"Check-Worthiness Score: {cw_probs[0].item() * 100:.2f}%")
    print("Underlying Rationality (Explainability):")
    causes_scores = rat_probs[0].cpu().numpy()
    for name, score in zip(causes_names, causes_scores):
        print(f"  - {name}: {score * 100:.2f}%")