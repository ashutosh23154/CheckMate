import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from co_net import CoNet
from li_net import LiNet


class CheckMate(nn.Module):
    """
    CheckMate — Joint model for explainable claim check-worthiness.
    Exactly as described in the paper:
    "Leveraging Rationality Labels for Explainable Claim Check-Worthiness"

    Architecture:
      1. BERT-base-uncased  →  contextual sequence embeddings (768-dim)
      2. CoNet              →  6 MHA blocks + self-attention aggregation
                               Output: (batch, 6, 768)
      3. LiNet              →  POS + DEP + emoticon features
                               Output: (batch, ling_output_dim)
      4. Rationality head   →  6 sigmoid neurons (one per label L1–L6)
      5. Check-Worthiness   →  Two MLP layers → 1 sigmoid neuron

    Outputs:
      - cw_prob   : (batch,)    — check-worthiness probability
      - rat_probs : (batch, 6)  — rationality label probabilities (L1–L6)
      - proj      : (batch, 128)— L2-normalised contrastive projection (training only)
    """

    def __init__(
        self,
        embed_size: int = 768,
        num_heads: int = 6,
        num_rat_labels: int = 6,
        ling_input_dim: int = 101,   # 50 POS + 50 DEP + 1 emoticon
        ling_output_dim: int = 64,
    ):
        super(CheckMate, self).__init__()

        # ── 1. BERT-base-uncased backbone ──
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # ── 2. CoNet: 6 MHA blocks + self-attention ──
        self.co_net = CoNet(embed_size, num_heads, num_rat_labels)

        # ── 3. LiNet: linguistic features feed-forward ──
        self.li_net = LiNet(ling_input_dim, ling_output_dim)

        # ── Dropout (paper: 0.1) ──
        self.dropout = nn.Dropout(0.1)

        # ── 4. Rationality head: 6 sigmoid neurons ──
        # Applied per-block on the aggregated (batch, 6, embed_size) tensor.
        # Linear(embed_size → 1) broadcast across the 6 label positions.
        self.rat_predictor = nn.Linear(embed_size, 1)

        # ── 5. Check-Worthiness head: Two MLPs → 1 sigmoid ──
        combined_dim = embed_size * num_rat_labels + ling_output_dim  # 6*768 + 64 = 4672
        hidden_dim = 256
        self.mlp1 = nn.Linear(combined_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        # ── 6. Contrastive Projection Head (NEW) ──
        # A 2-layer MLP that maps the 4608-dim CoNet output into a compact
        # 128-dim space where Supervised Contrastive Loss is applied.
        # This is only used during training — discarded at inference.
        self.proj_head = nn.Sequential(
            nn.Linear(embed_size * num_rat_labels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, input_ids, attention_mask, pos_tags, dep_parsing, emoticons):
        """
        Args:
            input_ids      : (batch, 128)  — BERT token IDs
            attention_mask : (batch, 128)  — BERT attention mask
            pos_tags       : (batch, 50)   — POS tag features
            dep_parsing    : (batch, 50)   — dependency parse features
            emoticons      : (batch, 1)    — emoticon count feature

        Returns:
            cw_prob   : (batch,)    — check-worthiness probability in [0, 1]
            rat_probs : (batch, 6)  — rationality probabilities in [0, 1]
            proj      : (batch, 128)— L2-normalised contrastive projection
        """

        # ── Step 1: BERT encoding ──
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state  # (batch, 128, 768)
        sequence_output = self.dropout(sequence_output)

        # ── Step 2: CoNet (6 MHA blocks + self-attention) ──
        # aggregated: (batch, 6, 768)
        aggregated = self.co_net(sequence_output)

        # ── Step 3: Rationality predictions (6 sigmoid neurons) ──
        # rat_predictor maps each (batch, embed_size) → scalar per label
        rat_logits = self.rat_predictor(aggregated).squeeze(-1)  # (batch, 6)
        rat_probs = torch.sigmoid(rat_logits)                    # (batch, 6)

        # ── Step 4: Flatten CoNet output for check-worthiness head ──
        co_flat = aggregated.reshape(aggregated.size(0), -1)  # (batch, 6*768 = 4608)
        co_flat = self.dropout(co_flat)

        # ── Step 5: LiNet linguistic features ──
        li_out = self.li_net(pos_tags, dep_parsing, emoticons)  # (batch, 64)
        li_out = self.dropout(li_out)

        # ── Step 6: Two MLP layers for check-worthiness ──
        combined = torch.cat([co_flat, li_out], dim=1)  # (batch, 4672)
        h = self.relu(self.mlp1(combined))               # (batch, 256)
        h = self.dropout(h)
        cw_logit = self.mlp2(h).squeeze(-1)             # (batch,)
        cw_prob = torch.sigmoid(cw_logit)               # (batch,)

        # ── Step 7: Contrastive projection (NEW) ──
        # Project the CoNet representation into a 128-dim space and L2-normalise.
        # Similar rationality profiles will be pulled together in this space.
        proj = self.proj_head(co_flat)          # (batch, 128)
        proj = F.normalize(proj, dim=1)         # L2-normalise → unit sphere

        return cw_prob, rat_probs, proj