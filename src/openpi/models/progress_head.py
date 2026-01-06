"""Improved progress estimation head with attention pooling and binning."""

import flax.nnx as nnx
import jax.numpy as jnp
from openpi.shared import array_typing as at


@at.typecheck
class AttentionPooling(nnx.Module):
    """Attention pooling layer to extract global representation from sequence with dimension reduction."""
    
    def __init__(self, input_dim: int, pool_dim: int, rngs: nnx.Rngs = None):
        """
        Args:
            input_dim: Input feature dimension
            pool_dim: Pooled dimension (for query and output)
            rngs: Random number generator
        """
        # Store dims as Python int (safe for tracing)
        self.pool_dim = int(pool_dim)
        
        # Learnable query vector for pooling
        self.query = nnx.Param(nnx.initializers.normal(stddev=0.02)(rngs.params(), (1, pool_dim)))
        
        # Key and value projections (compress from input_dim to pool_dim)
        # use_bias=False because checkpoint has None bias values
        self.key_proj = nnx.Linear(input_dim, pool_dim, rngs=rngs, use_bias=False)
        self.value_proj = nnx.Linear(input_dim, pool_dim, rngs=rngs, use_bias=False)
    
    def __call__(self, x: at.Float[at.Array, "b s d"], mask: at.Bool[at.Array, "b s"]) -> at.Float[at.Array, "b d"]:
        """
        Args:
            x: [batch, seq_len, embed_dim] input features
            mask: [batch, seq_len] attention mask
        Returns:
            pooled: [batch, embed_dim] pooled representation
        """
        batch_size = x.shape[0]
        
        # Expand query to batch
        query = jnp.tile(self.query, (batch_size, 1, 1))  # [b, 1, d]
        
        # Project keys and values
        keys = self.key_proj(x)  # [b, s, d]
        values = self.value_proj(x)  # [b, s, d]
        
        # Compute attention scores: query @ keys^T
        scores = jnp.matmul(query, keys.transpose(0, 2, 1))  # [b, 1, s]
        scores = scores / jnp.sqrt(self.pool_dim)  # Scale by sqrt(pool_dim)
        
        # Apply mask
        mask_expanded = mask[:, None, :]  # [b, 1, s]
        scores = jnp.where(mask_expanded, scores, -1e9)
        
        # Attention weights
        attn_weights = nnx.softmax(scores, axis=-1)  # [b, 1, s]
        
        # Weighted sum over values
        pooled = jnp.matmul(attn_weights, values)  # [b, 1, d]
        
        return pooled[:, 0, :]  # [b, d]


@at.typecheck
class MLPBlock(nnx.Module):
    """MLP block with LayerNorm and residual connection."""
    
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs = None):
        """
        Args:
            hidden_dim: Hidden dimension
            rngs: Random number generator
        """
        self.fc = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
    
    def __call__(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        """
        Args:
            x: [batch, hidden_dim] input features
        Returns:
            output: [batch, hidden_dim] with residual connection
        """
        residual = x
        x = self.fc(x)
        x = self.norm(x)
        x = nnx.gelu(x)
        return x + residual  # Residual connection


@at.typecheck
class ProgressHead(nnx.Module):
    """Improved progress estimation head with binning and soft-argmax."""
    
    def __init__(
        self, 
        input_dim: int, 
        num_bins: int = 101, 
        hidden_dim: int = 512, 
        num_layers: int = 3,
        pool_dim: int = 256,
        rngs: nnx.Rngs = None
    ):
        """
        Args:
            input_dim: VLM feature dimension (2048 for PaliGemma)
            num_bins: Number of progress bins (101 for 0%, 1%, ..., 100%)
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP blocks
            pool_dim: Dimension after attention pooling (256 for compression)
            rngs: Random number generator
        """
        self.num_bins = num_bins
        
        # 1. Attention pooling to extract global representation (with compression)
        self.attention_pool = AttentionPooling(input_dim, pool_dim, rngs=rngs)
        
        # 2. Input projection to hidden dimension
        self.input_proj = nnx.Linear(pool_dim, hidden_dim, rngs=rngs)
        self.input_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        
        # 3. MLP blocks with residual connections
        # Use dict instead of list to avoid integer keys in flatten_dict
        self.mlp_blocks = {
            f"block_{i}": MLPBlock(hidden_dim, rngs=rngs) 
            for i in range(num_layers)
        }
        
        # 4. Output projection to bins
        self.output_proj = nnx.Linear(hidden_dim, num_bins, rngs=rngs)
        
        # Bin centers for soft-argmax: [0.0, 0.01, 0.02, ..., 1.0]
        # Wrap as Variable (non-trainable constant) to avoid tracing issues
        self.bin_centers = nnx.Variable(jnp.linspace(0.0, 1.0, num_bins))
    
    def __call__(
        self, 
        features: at.Float[at.Array, "b s d"], 
        mask: at.Bool[at.Array, "b s"]
    ) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, "b {self.num_bins}"]]:
        """
        Args:
            features: [batch, seq_len, input_dim] VLM features
            mask: [batch, seq_len] attention mask
        Returns:
            progress: [batch] progress values in [0, 1]
            logits: [batch, num_bins] bin logits (for computing loss)
        """
        # 1. Attention pooling (compress to pool_dim)
        x = self.attention_pool(features, mask)  # [b, pool_dim]
        
        # 2. Input projection
        x = self.input_proj(x)  # [b, hidden_dim]
        x = self.input_norm(x)
        x = nnx.gelu(x)
        
        # 3. MLP blocks with residual connections
        # Iterate in order: block_0, block_1, block_2, ...
        for key in sorted(self.mlp_blocks.keys()):
            x = self.mlp_blocks[key](x)  # [b, hidden_dim]
        
        # 4. Bin logits
        logits = self.output_proj(x)  # [b, num_bins]
        
        # 5. Soft-argmax: compute expected value (differentiable)
        probs = nnx.softmax(logits, axis=-1)  # [b, num_bins]
        progress = jnp.sum(probs * self.bin_centers.value[None, :], axis=-1)  # [b]
        
        return progress, logits

