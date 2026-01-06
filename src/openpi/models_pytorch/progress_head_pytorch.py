
"""PyTorch implementation of progress estimation head with attention pooling and binning."""

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
    """Improved progress estimation head with binning and soft-argmax."""
    
    def __init__(
        self, 
        input_dim: int, 
        num_bins: int = 101, 
        hidden_dim: int = 512, 
        num_layers: int = 3,
        pool_dim: int = 256
    ):
        """
        Args:
            input_dim: VLM feature dimension (2048 for PaliGemma)
            num_bins: Number of progress bins (101 for 0%, 1%, ..., 100%)
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP blocks
            pool_dim: Dimension after attention pooling (256 for compression)
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
        
        # 4. Output projection to bins
        self.output_proj = nn.Linear(hidden_dim, num_bins)
        
        # Bin centers for soft-argmax: [0.0, 0.01, 0.02, ..., 1.0]
        self.register_buffer('bin_centers', torch.linspace(0.0, 1.0, num_bins))
    
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
            progress: [batch] progress values in [0, 1]
            logits: [batch, num_bins] bin logits (for computing loss)
        """
        # 1. Attention pooling
        x = self.attention_pool(features, mask)  # [b, input_dim]
        
        # 2. Input projection
        x = self.input_proj(x)  # [b, hidden_dim]
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # 3. MLP blocks with residual connections
        for block in self.mlp_blocks:
            x = block(x)  # [b, hidden_dim]
        
        # 4. Bin logits
        logits = self.output_proj(x)  # [b, num_bins]
        
        # 5. Soft-argmax: compute expected value (differentiable)
        probs = F.softmax(logits, dim=-1)  # [b, num_bins]
        progress = torch.sum(probs * self.bin_centers.unsqueeze(0), dim=-1)  # [b]
        
        return progress, logits

