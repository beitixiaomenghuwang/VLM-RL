"""Pi0 model with continuous progress head (101 classes for 0-100% prediction)."""

from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

import logging
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import nnx_bridge
from openpi.models import nnx_utils
from openpi.models import siglip as _siglip

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0Continuous(_model.BaseModel):
    """Pi0 model with continuous progress head (101 classes)."""
    
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Continuous progress estimation head with 101-class classification
        from openpi.models import progress_head_continuous as _progress_head
        self.progress_head = _progress_head.ProgressHead(
            input_dim=paligemma_config.width,  # 2048 for PaliGemma
            num_bins=101,  # Continuous classification: 0-100%
            hidden_dim=512,  # Match checkpoint configuration
            num_layers=3,
            pool_dim=2048,  # No dimension reduction in attention pooling
            rngs=rngs,
        )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image = obs.images[name]
            img_tokens = self.PaliGemma.img(image, train=not self.deterministic)
            tokens.append(img_tokens)
            input_mask.append(jnp.ones(img_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * img_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            llm_tokens = self.PaliGemma.llm.module.embed_module(obs.tokenized_prompt)
            tokens.append(llm_tokens)
            input_mask.append(obs.tokenized_prompt != 0)
            ar_mask += [True] * llm_tokens.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_tokens = self.state_proj(obs.state)
            tokens.append(state_tokens[:, None])
            input_mask.append(jnp.ones(state_tokens.shape[0], dtype=jnp.bool_)[:, None])
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_out(nnx.gelu(self.time_mlp_in(time_emb)))
            action_expert_tokens = action_tokens + time_emb[:, None, :]
            adarms_cond = time_emb
        else:
            state_time_emb = jnp.concatenate([state_tokens, time_emb], axis=-1)
            state_time_emb = self.action_time_mlp_out(nnx.gelu(self.action_time_mlp_in(state_time_emb)))
            action_expert_tokens = action_tokens + state_time_emb[:, None, :]
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    def compute_loss_with_progress(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        progress_target: at.Float[at.Array, " b"] | None = None,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, " b"]]:
        """Compute both action loss and progress loss.
        
        Args:
            rng: Random key
            observation: Observation including optional progress field
            actions: Ground truth actions
            train: Whether in training mode
            progress_target: Optional progress targets (if not in observation.progress)
            
        Returns:
            Tuple of (action_loss, progress_loss)
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once for action loss
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        
        # Compute progress loss
        progress_loss = jnp.zeros(batch_shape)
        if progress_target is not None:
            pred_progress = self.estimate_progress(observation)
            # Use smooth L1 loss (Huber loss) for robustness
            progress_loss = jnp.square(pred_progress - progress_target)
        elif observation.progress is not None:
            pred_progress = self.estimate_progress(observation)
            progress_loss = jnp.square(pred_progress - observation.progress)
        
        return action_loss, progress_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(rng, observation, train=False)
        batch_size = next(iter(observation.images.values())).shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        x = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim)) if noise is None else noise
        ts = jnp.linspace(0.0, 1.0, num_steps + 1)
        for i, t in enumerate(ts[1:]):
            time = jnp.full((batch_size,), t)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x, time)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            x = x - v_t * (ts[i + 1] - ts[i])
        return x

    @at.typecheck
    def estimate_progress(self, obs: _model.Observation) -> at.Float[at.Array, " b"]:
        prefix_tokens, prefix_mask, _ = self.embed_prefix(obs)
        progress, _ = self.progress_head(prefix_tokens, prefix_mask)  # [b]
        return progress
    
    @at.typecheck
    def estimate_progress_with_logits(
        self, obs: _model.Observation, stop_gradient_backbone: bool = False
    ) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, "b {self.progress_head.num_bins}"]]:
        """Estimate progress and return both the weighted average and raw logits.
        
        Args:
            obs: Observation containing images
            stop_gradient_backbone: If True, stop gradient through PaliGemma backbone
            
        Returns:
            progress: [batch] weighted average progress (0-1)
            logits: [batch, 101] class logits for computing loss
        """
        prefix_tokens, prefix_mask, _ = self.embed_prefix(obs)
        if stop_gradient_backbone:
            prefix_tokens = jax.lax.stop_gradient(prefix_tokens)
        return self.progress_head(prefix_tokens, prefix_mask)
