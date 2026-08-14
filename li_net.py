import torch
import torch.nn as nn


class LiNet(nn.Module):
    """
    Linguistic Network (LiNet) — as described in the paper.

    Processes three linguistic feature streams extracted by spaCy:
      - POS tags        (50-dim)
      - Dependency parse (50-dim)
      - Emoticon count   (1-dim)
    Total input: 101-dim → feed-forward linear → output_dim
    """

    def __init__(self, input_dim: int = 101, output_dim: int = 64):
        super(LiNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, pos_tags, dep_parsing, emoticons):
        """
        Args:
            pos_tags:    (batch, 50)
            dep_parsing: (batch, 50)
            emoticons:   (batch, 1)

        Returns:
            out: (batch, output_dim)
        """
        # Concatenate POS (50) + DEP (50) + emoticon (1) = 101
        linguistic_features = torch.cat([pos_tags, dep_parsing, emoticons], dim=1)
        out = self.linear(linguistic_features)
        return out
