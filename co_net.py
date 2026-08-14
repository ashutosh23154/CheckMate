import torch
import torch.nn as nn


class CoNet(nn.Module):
    """
    Contextual Network (CoNet) — as described in the paper.

    Architecture:
      - 6 Multi-Head Attention (MHA) blocks, each dedicated to one
        rationality label (L1–L6), processing the BERT sequence output.
      - A Self-Attention layer aggregates the 6 block-level representations.

    Returns:
      aggregated: (batch, num_rat_labels, embed_size)
    """

    def __init__(self, embed_size: int = 768, num_heads: int = 6,
                 num_rat_labels: int = 6):
        super(CoNet, self).__init__()
        self.num_rat_labels = num_rat_labels

        # 6 MHA blocks — one per rationality label (L1–L6)
        self.mha_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_size,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_rat_labels)
        ])

        # Self-Attention layer to aggregate the 6 block outputs
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_size) — BERT last hidden states

        Returns:
            aggregated: (batch, num_rat_labels, embed_size)
        """
        # Each MHA block performs self-attention over the token sequence,
        # then we mean-pool across the sequence to get one vector per block.
        block_outputs = []
        for mha in self.mha_blocks:
            out, _ = mha(x, x, x)           # (batch, seq_len, embed_size)
            pooled = out.mean(dim=1)         # (batch, embed_size)
            block_outputs.append(pooled)

        # Stack along a new "label" dim → (batch, num_rat_labels, embed_size)
        stacked = torch.stack(block_outputs, dim=1)

        # Self-attention across the 6 rationality-label representations
        aggregated, _ = self.self_attn(stacked, stacked, stacked)
        # aggregated: (batch, num_rat_labels, embed_size)

        return aggregated
