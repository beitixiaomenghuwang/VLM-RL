# Pi0.5 进度估计功能快速开始指南

## 快速开始步骤

### 步骤 1: 生成带进度标签的数据集（5-10分钟）

```bash
# 激活 lerobot 环境
conda activate lerobot

# 生成带进度标签的数据集
python scripts/add_progress_labels.py \
    --input_dataset /media/caslx/0E73-05CF/Data/cubestack2025_1126_merge \
    --output_dataset /media/caslx/0E73-05CF/Data/cubestack2025_1126_merge_with_progress \
    --overwrite
```

**预期输出**：
```
加载了 X 个 episodes
处理 episodes: 100%|██████████| X/X
✅ 成功创建带进度标签的数据集
   总帧数: XXXX
   Episodes: X
```

### 步骤 2: 测试模型功能（1分钟）

```bash
# 运行测试脚本
python test_progress_estimation.py
```

**预期输出**：
```
🎉 所有测试通过！进度估计功能已正确实现。
```

### 步骤 3: 训练模型（数小时，取决于数据量）

```bash
# 训练 pi0.5 模型
uv run scripts/train.py \
    --config=pi05_teleavatar \
    --exp_name=cubestack_with_progress \
    --data.repo_id=/media/caslx/0E73-05CF/Data/cubestack2025_1126_merge_with_progress \
    --num_train_steps=20000 \
    --batch_size=64
```

**预期日志**（每100步）：
```
Step 100: loss=0.1234, action_loss=0.1200, progress_loss=0.0034, grad_norm=1.23, param_norm=45.67
Step 200: loss=0.1100, action_loss=0.1070, progress_loss=0.0030, grad_norm=1.15, param_norm=45.67
...
```

**监控指标**（WandB）：
- `loss`: 总损失（应持续下降）
- `action_loss`: 动作预测损失（主要任务，应下降）
- `progress_loss`: 进度估计损失（辅助任务，应下降）

### 步骤 4: 启动推理服务器

```bash
# 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_teleavatar \
    --policy.dir=checkpoints/pi05_teleavatar/cubestack_with_progress/19999
```

**预期输出**：
```
INFO: Policy server started on 0.0.0.0:8000
INFO: Model supports progress estimation
```

### 步骤 5: 运行客户端测试推理

```bash
# 在另一个终端运行
python examples/teleavatar/main.py --remote-host 127.0.0.1
```

**预期输出**（服务器端）：
```
INFO: Connection from ('127.0.0.1', 12345) opened
INFO: Task progress: 0.00%
INFO: Task progress: 5.23%
INFO: Task progress: 12.45%
...
INFO: Task progress: 98.76%
INFO: Task progress: 100.00%
```

## 验证清单

- [ ] 数据集生成成功，包含 `observation.progress` 字段
- [ ] 测试脚本全部通过
- [ ] 训练日志显示 `action_loss` 和 `progress_loss` 都在下降
- [ ] 推理服务器启动成功
- [ ] 客户端可以接收到 `progress` 字段
- [ ] 进度值在 [0, 1] 范围内且合理变化

## 故障排除

### 问题 1: 数据集生成失败

**错误**: `ModuleNotFoundError: No module named 'lerobot'`

**解决**:
```bash
conda activate lerobot
# 确保在正确的环境中
```

### 问题 2: 训练时没有 progress_loss

**检查**:
```bash
# 确认数据集包含 progress 字段
conda run -n lerobot python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('/path/to/dataset_with_progress')
print('Keys:', list(ds[0].keys()))
"
# 应该看到 'observation.progress' 在列表中
```

### 问题 3: 推理时 progress 为 None

**原因**: 可能加载了旧模型（没有 progress_head）

**解决**: 确保使用新训练的模型：
```bash
ls checkpoints/pi05_teleavatar/cubestack_with_progress/19999/params/
# 应该看到包含 progress_head 的参数文件
```

## 高级配置

### 调整进度损失权重

编辑 `scripts/train.py`，修改第 231 行附近：

```python
# 默认权重 0.1
total_loss = jnp.mean(action_loss) + 0.1 * jnp.mean(progress_loss)

# 增加进度估计重要性
total_loss = jnp.mean(action_loss) + 0.2 * jnp.mean(progress_loss)

# 减少进度估计重要性
total_loss = jnp.mean(action_loss) + 0.05 * jnp.mean(progress_loss)
```

### 使用不同的进度标签策略

当前使用线性进度。如果需要基于里程碑的进度，修改 `scripts/add_progress_labels.py`：

```python
# 线性进度（默认）
progress = frame_idx / (ep_length - 1)

# 基于里程碑的进度（需要手动定义里程碑）
milestones = [0, 100, 200, ep_length-1]  # 帧索引
milestone_progress = [0.0, 0.3, 0.7, 1.0]  # 对应进度
progress = np.interp(frame_idx, milestones, milestone_progress)
```

## 下一步

完成上述步骤后，你可以：

1. **分析进度估计质量**: 检查预测进度与实际任务进展是否一致
2. **优化权重**: 根据训练效果调整 progress_loss 权重
3. **集成到应用**: 在机器人控制应用中使用进度信息进行任务监控
4. **扩展功能**: 添加子任务识别、剩余时间估计等功能

## 联系与支持

如遇到问题，请检查：
- 日志文件中的错误信息
- WandB 中的训练曲线
- 测试脚本的输出

祝训练顺利！🚀

