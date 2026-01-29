
"""PyTorch implementation of progress estimation head with attention pooling and binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention pooling layer to extract global representation from sequence with dimension reduction."""
    
    def __init__(self, input_dim: int, pool_dim: int):
        """
        Args:
            input_dim: Input feature dimension
            pool_dim: Pooled dimension (for query and output)
        """
        super().__init__()
        self.pool_dim = pool_dim
        
        # Learnable query vector for pooling
        self.query = nn.Parameter(torch.randn(1, pool_dim) * 0.02)
        
        # Key and value projections (compress from input_dim to pool_dim)
        # bias=False because JAX checkpoint has None bias values
        self.key_proj = nn.Linear(input_dim, pool_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, pool_dim, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim] input features
            mask: [batch, seq_len] attention mask
        Returns:
            pooled: [batch, embed_dim] pooled representation
        """
        batch_size = x.shape[0]
        
        # Expand query to batch
        query = self.query.expand(batch_size, -1, -1)  # [b, 1, d]
        
        # Project keys and values
        keys = self.key_proj(x)  # [b, s, d]
        values = self.value_proj(x)  # [b, s, d]
        
        # Compute attention scores: query @ keys^T
        scores = torch.matmul(query, keys.transpose(1, 2))  # [b, 1, s]
        scores = scores / torch.sqrt(torch.tensor(self.pool_dim, dtype=scores.dtype, device=scores.device))
        
        # Apply mask
        mask_expanded = mask.unsqueeze(1)  # [b, 1, s]
        scores = torch.where(mask_expanded, scores, torch.tensor(-1e9, dtype=scores.dtype, device=scores.device))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [b, 1, s]
        
        # Weighted sum over values
        pooled = torch.matmul(attn_weights, values)  # [b, 1, d]
        
        return pooled[:, 0, :]  # [b, d]


class MLPBlock(nn.Module):
    """MLP block with LayerNorm and residual connection."""
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] input features
        Returns:
            output: [batch, hidden_dim] with residual connection
        """
        residual = x
        x = self.fc(x)
        x = self.norm(x)
        x = F.gelu(x)
        return x + residual  # Residual connection


class ProgressHead(nn.Module):
    """Binary classification head for task completion prediction."""
    
    def __init__(
        self, 
        input_dim: int, 
        num_bins: int = 2, 
        hidden_dim: int = 512, 
        num_layers: int = 3,
        pool_dim: int = 2048
    ):
        """
        Args:
            input_dim: VLM feature dimension (2048 for PaliGemma)
            num_bins: Number of classes (2 for binary: 0=incomplete, 1=complete)
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP blocks
            pool_dim: Dimension after attention pooling
        """
        super().__init__()
        self.num_bins = num_bins
        
        # 1. Attention pooling to extract global representation (with compression)
        self.attention_pool = AttentionPooling(input_dim, pool_dim)
        
        # 2. Input projection to hidden dimension
        self.input_proj = nn.Linear(pool_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # 3. MLP blocks with residual connections
        self.mlp_blocks = nn.ModuleList([MLPBlock(hidden_dim) for _ in range(num_layers)])
        
        # 4. Output projection to bins (2 classes for binary classification)
        self.output_proj = nn.Linear(hidden_dim, num_bins)
    
    def forward(
        self, 
        features: torch.Tensor, 
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, seq_len, input_dim] VLM features
            mask: [batch, seq_len] attention mask
        Returns:
            progress: [batch] probability of task completion (class 1), rounded to 0 or 1
            logits: [batch, num_bins] class logits (for computing loss)
        """
        # 1. Attention pooling
        x = self.attention_pool(features, mask)  # [b, pool_dim]
        
        # 2. Input projection
        x = self.input_proj(x)  # [b, hidden_dim]
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # 3. MLP blocks with residual connections
        for block in self.mlp_blocks:
            x = block(x)  # [b, hidden_dim]
        
        # 4. Binary classification logits
        logits = self.output_proj(x)  # [b, 2]
        
        # 5. Return prediction: -1 (incomplete) or 1 (complete)
        probs = F.softmax(logits, dim=-1)  # [b, 2]
        progress = (probs[:, 1] > 0.5).float() * 2 - 1  # [b] -1 or 1
        
        return progress, logits


