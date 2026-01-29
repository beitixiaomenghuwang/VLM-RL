#!/usr/bin/env python3
"""
Convert JAX model with continuous progress head (101 classes) to PyTorch format.

Usage:
    python examples/convert_jax_model_to_pytorch_continuous.py \
        --checkpoint_dir checkpoints/pi05RL_pick_marker_continuous \
        --output_path checkpoints/pi05RL_pick_marker_continuous_torch \
        --config-name pi05_teleavatar
"""

import json
import os
import pathlib
import shutil
from typing import Literal

from flax.nnx import traversals
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config


def slice_progress_head_state_dict_continuous(state_dict):
    """Convert ProgressHead JAX parameters to PyTorch format (101-class version)."""
    if state_dict is None or len(state_dict) == 0:
        return {}
    
    pytorch_state_dict = {}
    state_dict = dict(state_dict)
    
    # 1. Attention pooling
    if "attention_pool/query" in state_dict:
        query = state_dict.pop("attention_pool/query")
        # Keep shape as [1, embed_dim] to match PyTorch model
        pytorch_state_dict["progress_head.attention_pool.query"] = torch.from_numpy(np.array(query))
    
    if "attention_pool/key_proj/kernel" in state_dict:
        pytorch_state_dict["progress_head.attention_pool.key_proj.weight"] = torch.from_numpy(
            np.array(state_dict.pop("attention_pool/key_proj/kernel")).T
        )
    # Note: key_proj and value_proj do not have bias (use_bias=False)
    
    if "attention_pool/value_proj/kernel" in state_dict:
        pytorch_state_dict["progress_head.attention_pool.value_proj.weight"] = torch.from_numpy(
            np.array(state_dict.pop("attention_pool/value_proj/kernel")).T
        )
    
    # 2. Input projection
    if "input_proj/kernel" in state_dict:
        pytorch_state_dict["progress_head.input_proj.weight"] = torch.from_numpy(
            np.array(state_dict.pop("input_proj/kernel")).T
        )
    if "input_proj/bias" in state_dict:
        pytorch_state_dict["progress_head.input_proj.bias"] = torch.from_numpy(np.array(state_dict.pop("input_proj/bias")))
    
    # 3. Input LayerNorm
    if "input_norm/scale" in state_dict:
        pytorch_state_dict["progress_head.input_norm.weight"] = torch.from_numpy(np.array(state_dict.pop("input_norm/scale")))
    if "input_norm/bias" in state_dict:
        pytorch_state_dict["progress_head.input_norm.bias"] = torch.from_numpy(np.array(state_dict.pop("input_norm/bias")))
    
    # 4. MLP blocks
    block_keys = set()
    for key in list(state_dict.keys()):
        if key.startswith("mlp_blocks/"):
            block_keys.add(key.split("/")[1])
    
    # Convert block names: "block_0" -> "0", "block_1" -> "1", etc.
    for block_name in sorted(block_keys):
        # Extract block index from "block_X" format
        block_idx = block_name.replace("block_", "")
        
        if f"mlp_blocks/{block_name}/fc/kernel" in state_dict:
            pytorch_state_dict[f"progress_head.mlp_blocks.{block_idx}.fc.weight"] = torch.from_numpy(
                np.array(state_dict.pop(f"mlp_blocks/{block_name}/fc/kernel")).T
            )
        if f"mlp_blocks/{block_name}/fc/bias" in state_dict:
            pytorch_state_dict[f"progress_head.mlp_blocks.{block_idx}.fc.bias"] = torch.from_numpy(
                np.array(state_dict.pop(f"mlp_blocks/{block_name}/fc/bias"))
            )
        if f"mlp_blocks/{block_name}/norm/scale" in state_dict:
            pytorch_state_dict[f"progress_head.mlp_blocks.{block_idx}.norm.weight"] = torch.from_numpy(
                np.array(state_dict.pop(f"mlp_blocks/{block_name}/norm/scale"))
            )
        if f"mlp_blocks/{block_name}/norm/bias" in state_dict:
            pytorch_state_dict[f"progress_head.mlp_blocks.{block_idx}.norm.bias"] = torch.from_numpy(
                np.array(state_dict.pop(f"mlp_blocks/{block_name}/norm/bias"))
            )
    
    # 5. Output projection (101D for continuous classification)
    if "output_proj/kernel" in state_dict:
        kernel = np.array(state_dict.pop("output_proj/kernel"))
        pytorch_state_dict["progress_head.output_proj.weight"] = torch.from_numpy(kernel.T)
    if "output_proj/bias" in state_dict:
        bias = np.array(state_dict.pop("output_proj/bias"))
        pytorch_state_dict["progress_head.output_proj.bias"] = torch.from_numpy(bias)
    
    return pytorch_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """Load and process params by restoring via JAX model loader first."""
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )
    print(f"Loaded params from {checkpoint_dir}/params/")
    return params


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """Load JAX model and print parameter keys."""
    params = slice_initial_orbax_checkpoint(checkpoint_dir)
    
    def print_tree(d, prefix=""):
        if isinstance(d, dict):
            for key in sorted(d.keys()):
                print(f"{prefix}{key}")
                print_tree(d[key], prefix + "  ")
        elif hasattr(d, "shape"):
            print(f"{prefix}  -> shape: {d.shape}, dtype: {d.dtype}")
    
    print("\nParameter tree:")
    print_tree(params)


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    """Convert PI0 JAX checkpoint to PyTorch format."""
    print(f"\nConverting checkpoint from {checkpoint_dir} to {output_path}")
    
    # Load JAX checkpoint
    params = slice_initial_orbax_checkpoint(checkpoint_dir, restore_precision=precision)
    
    # Extract progress_head params
    progress_head_params = traversals.flatten_dict(params.get("progress_head", {}), sep="/")
    
    # Convert progress head (101-class version)
    pytorch_state_dict = slice_progress_head_state_dict_continuous(progress_head_params)
    
    print(f"\nConverted {len(pytorch_state_dict)} progress_head parameters")
    
    # Save to safetensors
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    safetensors.torch.save_file(pytorch_state_dict, output_path / "model.safetensors")
    
    # Save config
    config_dict = {
        "num_bins": 101,
        "hidden_dim": 512,
        "num_layers": 3,
        "pool_dim": 2048,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✅ Saved PyTorch checkpoint to {output_path}")
    print(f"   - model.safetensors ({len(pytorch_state_dict)} parameters)")
    print(f"   - config.json")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    """
    Convert JAX checkpoint with continuous progress head to PyTorch.
    
    Args:
        checkpoint_dir: Path to JAX checkpoint directory
        config_name: Config name (e.g., 'pi05_teleavatar')
        output_path: Output path for PyTorch checkpoint
        precision: Precision for conversion
        inspect_only: Only print keys without conversion
    """
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
        return
    
    if output_path is None:
        raise ValueError("output_path is required for conversion")
    
    # Load model config
    model_config = _config.resolve_model_config(config_name)
    
    # Convert checkpoint
    convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)
