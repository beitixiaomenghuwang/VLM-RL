#!/usr/bin/env python3
"""
测试进度估计功能的脚本。

用法：
    python test_progress_estimation.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config


def test_model_has_progress_head():
    """测试模型是否有进度估计头"""
    print("=" * 60)
    print("测试 1: 检查模型是否有进度估计头")
    print("=" * 60)
    
    config = _config.get_config("pi05_teleavatar")
    model = config.model.create(jax.random.key(0))
    
    has_progress_head = hasattr(model, 'progress_head')
    has_estimate_method = hasattr(model, 'estimate_progress')
    has_compute_loss_with_progress = hasattr(model, 'compute_loss_with_progress')
    
    print(f"  ✓ 模型有 progress_head: {has_progress_head}")
    print(f"  ✓ 模型有 estimate_progress 方法: {has_estimate_method}")
    print(f"  ✓ 模型有 compute_loss_with_progress 方法: {has_compute_loss_with_progress}")
    
    assert has_progress_head, "模型缺少 progress_head"
    assert has_estimate_method, "模型缺少 estimate_progress 方法"
    assert has_compute_loss_with_progress, "模型缺少 compute_loss_with_progress 方法"
    
    print("\n✅ 测试通过：模型具有进度估计功能\n")
    return model, config


def test_progress_estimation(model, config):
    """测试进度估计功能"""
    print("=" * 60)
    print("测试 2: 测试进度估计功能")
    print("=" * 60)
    
    # 创建假观测
    fake_obs = config.model.fake_obs(batch_size=4)
    
    # 估计进度
    progress = model.estimate_progress(fake_obs)
    
    print(f"  输入批次大小: {fake_obs.state.shape[0]}")
    print(f"  输出进度形状: {progress.shape}")
    print(f"  进度值: {progress}")
    
    # 验证
    assert progress.shape == (4,), f"进度形状应为 (4,)，实际为 {progress.shape}"
    assert jnp.all((progress >= 0.0) & (progress <= 1.0)), "进度值应在 [0, 1] 范围内"
    
    print(f"  ✓ 所有进度值都在 [0, 1] 范围内")
    print("\n✅ 测试通过：进度估计功能正常\n")


def test_progress_loss(model, config):
    """测试进度损失计算"""
    print("=" * 60)
    print("测试 3: 测试进度损失计算")
    print("=" * 60)
    
    # 创建假数据
    rng = jax.random.key(42)
    fake_obs = config.model.fake_obs(batch_size=2)
    fake_actions = config.model.fake_act(batch_size=2)
    
    # 添加进度标签
    fake_progress = jnp.array([0.3, 0.7])
    fake_obs = _model.Observation(
        images=fake_obs.images,
        image_masks=fake_obs.image_masks,
        state=fake_obs.state,
        tokenized_prompt=fake_obs.tokenized_prompt,
        tokenized_prompt_mask=fake_obs.tokenized_prompt_mask,
        progress=fake_progress,
    )
    
    # 计算损失
    action_loss, progress_loss = model.compute_loss_with_progress(
        rng, fake_obs, fake_actions, train=False
    )
    
    print(f"  Action loss shape: {action_loss.shape}")
    print(f"  Progress loss shape: {progress_loss.shape}")
    print(f"  Action loss: {jnp.mean(action_loss):.4f}")
    print(f"  Progress loss: {jnp.mean(progress_loss):.4f}")
    
    # 验证
    assert action_loss.shape[0] == 2, "Action loss 批次大小应为 2"
    assert progress_loss.shape == (2,), "Progress loss 形状应为 (2,)"
    assert jnp.all(jnp.isfinite(action_loss)), "Action loss 应该是有限值"
    assert jnp.all(jnp.isfinite(progress_loss)), "Progress loss 应该是有限值"
    
    print(f"  ✓ 损失计算成功")
    print(f"  ✓ 所有损失值都是有限的")
    print("\n✅ 测试通过：进度损失计算正常\n")


def test_observation_with_progress():
    """测试 Observation 是否支持 progress 字段"""
    print("=" * 60)
    print("测试 4: 测试 Observation 支持 progress 字段")
    print("=" * 60)
    
    # 创建包含 progress 的数据字典
    data = {
        "image": {
            "base_0_rgb": jnp.ones((2, 224, 224, 3), dtype=jnp.float32),
        },
        "image_mask": {
            "base_0_rgb": jnp.ones((2,), dtype=bool),
        },
        "state": jnp.ones((2, 16), dtype=jnp.float32),
        "progress": jnp.array([0.2, 0.8], dtype=jnp.float32),
    }
    
    # 从字典创建 Observation
    obs = _model.Observation.from_dict(data)
    
    print(f"  Progress 字段: {obs.progress}")
    print(f"  Progress 形状: {obs.progress.shape if obs.progress is not None else None}")
    
    # 验证
    assert obs.progress is not None, "Progress 字段应该存在"
    assert obs.progress.shape == (2,), f"Progress 形状应为 (2,)，实际为 {obs.progress.shape}"
    assert jnp.allclose(obs.progress, jnp.array([0.2, 0.8])), "Progress 值不匹配"
    
    print(f"  ✓ Observation 正确支持 progress 字段")
    print("\n✅ 测试通过：Observation 数据结构正常\n")


def test_backward_compatibility():
    """测试向后兼容性（没有 progress 字段）"""
    print("=" * 60)
    print("测试 5: 测试向后兼容性")
    print("=" * 60)
    
    config = _config.get_config("pi05_teleavatar")
    model = config.model.create(jax.random.key(0))
    
    # 创建不包含 progress 的观测
    fake_obs = config.model.fake_obs(batch_size=2)
    assert fake_obs.progress is None, "默认 fake_obs 不应有 progress"
    
    # 测试 compute_loss（不使用 progress）
    rng = jax.random.key(42)
    fake_actions = config.model.fake_act(batch_size=2)
    
    # 原始 compute_loss 方法应该仍然工作
    action_loss = model.compute_loss(rng, fake_obs, fake_actions, train=False)
    print(f"  Action loss (without progress): {jnp.mean(action_loss):.4f}")
    
    # compute_loss_with_progress 也应该工作（progress_target=None）
    action_loss2, progress_loss = model.compute_loss_with_progress(
        rng, fake_obs, fake_actions, train=False
    )
    print(f"  Action loss (with progress API): {jnp.mean(action_loss2):.4f}")
    print(f"  Progress loss (no target): {jnp.mean(progress_loss):.4f}")
    
    # 验证
    assert jnp.allclose(action_loss, action_loss2, rtol=1e-5), "两种方法的 action loss 应该相同"
    assert jnp.all(progress_loss == 0.0), "没有目标时 progress loss 应为 0"
    
    print(f"  ✓ 向后兼容性正常")
    print("\n✅ 测试通过：向后兼容性良好\n")


def main():
    print("\n" + "=" * 60)
    print("Pi0.5 进度估计功能测试套件")
    print("=" * 60 + "\n")
    
    try:
        # 测试 1: 检查模型结构
        model, config = test_model_has_progress_head()
        
        # 测试 2: 进度估计
        test_progress_estimation(model, config)
        
        # 测试 3: 进度损失
        test_progress_loss(model, config)
        
        # 测试 4: Observation 数据结构
        test_observation_with_progress()
        
        # 测试 5: 向后兼容性
        test_backward_compatibility()
        
        print("=" * 60)
        print("🎉 所有测试通过！进度估计功能已正确实现。")
        print("=" * 60)
        print("\n下一步：")
        print("  1. 使用 scripts/add_progress_labels.py 为数据集添加进度标签")
        print("  2. 训练模型验证实际效果")
        print("  3. 在推理时测试进度输出")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

