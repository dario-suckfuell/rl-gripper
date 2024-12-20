import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadProjection2L(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadProjection2L, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        # Parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for query, key, and value
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Final linear layer
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = value.size(0)

        # Transform query, key, and value using linear layers
        V = self.v_linear(value)

        # Apply final linear layer
        output = self.out_linear(V)
        return output, output



class MultiHeadProjection(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadProjection, self).__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        # Parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for each head
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, self.head_dim) for _ in range(num_heads)
        ])

        # Initialize each head with unique random weights
        self._initialize_heads()

        # Final linear layer
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def _initialize_heads(self):
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, query, key, value, mask=None):
        batch_size = value.size(0)

        # Apply per-head transformations
        head_outputs = []
        for head_layer in self.heads:
            head_outputs.append(head_layer(value))  # Each head transforms the input

        # Concatenate all head outputs
        V_concat = torch.cat(head_outputs, dim=-1)  # Combine all head outputs

        # Apply final linear layer
        output = self.out_linear(V_concat)
        return output, output

