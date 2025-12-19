# Teleavatar数据集推理功能 - 开发总结

## 概述

为OpenPI的Teleavatar机器人添加了从LeRobot格式数据集读取observation进行离线推理的功能，替代实时ROS2接口。

## 创建的文件

### 1. `dataset_interface.py` - 数据集接口类

**功能**：
- 从LeRobot格式数据集加载observation数据
- 提供与`TeleavatarROS2Interface`相同的API接口
- 支持episode和frame级别的导航

**主要类**：
- `TeleavatarDatasetInterface`: 主接口类

**关键方法**：
- `__init__(dataset_path, episode_index, start_frame)`: 初始化数据集
- `get_observation()`: 获取当前帧的observation
- `wait_for_initial_data()`: 兼容性方法（数据集总是就绪）
- `reset(episode_index, start_frame)`: 重置到指定episode/frame
- `get_episode_info()`: 获取当前episode信息
- `publish_action(actions)`: 兼容性方法（数据集模式不执行动作）

**数据格式**：
```python
{
    'images': {
        'left_color': np.ndarray (480, 848, 3),    # 左手腕相机
        'right_color': np.ndarray (480, 848, 3),   # 右手腕相机
        'head_camera': np.ndarray (1080, 1920, 3), # 头部相机
    },
    'state': np.ndarray (48,)  # [positions(16), velocities(16), efforts(16)]
}
```

### 2. `env_dataset.py` - 数据集环境包装器

**功能**：
- 实现`openpi_client.runtime.Environment`接口
- 使用`TeleavatarDatasetInterface`作为数据源
- 处理episode完成逻辑

**主要类**：
- `TeleavatarDatasetEnvironment`: 环境类

**关键方法**：
- `__init__(prompt, dataset_path, episode_index, start_frame)`: 初始化环境
- `get_observation()`: 获取observation（调用dataset_interface）
- `apply_action(action)`: 应用动作（数据集模式为空操作）
- `is_episode_complete()`: 检查episode是否完成
- `reset()`: 重置环境
- `reset_to_episode(episode_index, start_frame)`: 重置到特定episode
- `set_prompt(prompt)`: 更新语言指令
- `get_episode_info()`: 获取episode元数据

**返回的observation格式**：
```python
{
    'observation/state': np.ndarray (48,),
    'observation/images/left_color': np.ndarray (480, 848, 3),
    'observation/images/right_color': np.ndarray (480, 848, 3),
    'observation/images/head_camera': np.ndarray (1080, 1920, 3),
    'prompt': str,
}
```

### 3. `main_dataset.py` - 数据集推理主程序

**功能**：
- 使用数据集环境进行策略推理
- 连接远程策略服务器
- 支持命令行参数配置

**命令行参数**：
- `--remote-host`: 策略服务器IP（默认：127.0.0.1）
- `--remote-port`: 策略服务器端口（默认：8000）
- `--dataset-path`: 数据集路径
- `--episode-index`: Episode索引（默认：0）
- `--start-frame`: 起始帧（默认：0）
- `--control-frequency`: 控制频率（默认：20.0 Hz）
- `--action-horizon`: 动作horizon（默认：10）
- `--open-loop-horizon`: 开环horizon（默认：8）
- `--prompt`: 语言指令
- `--num-episodes`: Episode数量（默认：1）
- `--max-episode-steps`: 最大步数（0=使用数据集长度）

### 4. `test_dataset_interface.py` - 测试脚本

**功能**：
- 测试dataset_interface的基本功能
- 无需策略服务器即可运行
- 验证数据加载和导航功能

**测试内容**：
1. 初始化数据集接口
2. 加载初始数据
3. 显示episode信息
4. 读取前5帧
5. 测试reset功能
6. 测试切换episode

### 5. `README_DATASET.md` - 使用文档

**内容**：
- 详细的使用说明
- 参数解释
- 数据集格式要求
- 接口映射关系
- 与ROS2接口的对比
- 故障排除指南
- 使用示例

### 6. `DATASET_INFERENCE_SUMMARY.md` - 本文档

总结开发的所有组件和使用方法。

## 使用流程

### 快速开始

1. **测试数据集接口**（无需策略服务器）：
```bash
python examples/teleavatar/test_dataset_interface.py
```

2. **启动策略服务器**：
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_low_mem_finetune \
    --policy.dir=pi0_teleavatar_low_mem_finetune_new_data/pi0_lora_with_joint_positions_and_gripper_efforts_new_data/29999
```

3. **运行数据集推理**：
```bash
python examples/teleavatar/main_dataset.py \
    --remote-host 127.0.0.1 \
    --dataset-path datasets \
    --episode-index 0 \
    --prompt "Pick up the toy and drop it in the basket on the left"
```

## 架构设计

### 层次结构

```
main_dataset.py
    ↓ 使用
env_dataset.py (TeleavatarDatasetEnvironment)
    ↓ 使用
dataset_interface.py (TeleavatarDatasetInterface)
    ↓ 加载
LeRobot Dataset (datasets/)
```

### 与原有架构的对比

**原有（ROS2模式）**：
```
main.py
    ↓
env.py (TeleavatarEnvironment)
    ↓
ros2_interface.py (TeleavatarROS2Interface)
    ↓
ROS2 Topics (实时数据)
```

**新增（数据集模式）**：
```
main_dataset.py
    ↓
env_dataset.py (TeleavatarDatasetEnvironment)
    ↓
dataset_interface.py (TeleavatarDatasetInterface)
    ↓
LeRobot Dataset (离线数据)
```

### API一致性

两种模式提供相同的接口，确保可以无缝切换：

| 接口方法 | ROS2模式 | 数据集模式 |
|---------|---------|-----------|
| `get_observation()` | ✓ 从ROS2读取 | ✓ 从数据集读取 |
| `wait_for_initial_data()` | ✓ 等待ROS2数据 | ✓ 立即返回True |
| `publish_action()` | ✓ 发布到ROS2 | ✓ 空操作 |
| `is_episode_complete()` | ✗ 永不完成 | ✓ 检查数据集结束 |
| `reset()` | ✓ 无操作 | ✓ 重置到起点 |

## 数据映射

### 数据集结构
```
datasets/
├── data/chunk-000/
│   └── episode_*.parquet  # 状态和元数据
├── videos/chunk-000/
│   ├── observation.images.left_wrist/
│   ├── observation.images.right_wrist/
│   └── observation.images.head/
└── meta/
    ├── info.json           # 数据集元信息
    ├── episodes.jsonl      # Episode元数据
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

### 字段映射

| 数据集字段 | 环境字段 | 维度/分辨率 |
|-----------|---------|------------|
| `observation.state[:16]` | `observation/state[:16]` | 16 (positions) |
| `observation.state[16:32]` | `observation/state[16:32]` | 16 (velocities) |
| `observation.state[32:48]` | `observation/state[32:48]` | 16 (efforts) |
| `observation.images.left_wrist` | `observation/images/left_color` | 480×848×3 |
| `observation.images.right_wrist` | `observation/images/right_color` | 480×848×3 |
| `observation.images.head` | `observation/images/head_camera` | 1080×1920×3 |

注意：数据集的`observation.state`是62维，包含end-effector poses（索引48-61），但推理时只使用前48维。

## 关键特性

### 1. 完全兼容的API
- 数据集接口提供与ROS2接口相同的方法
- 环境类实现相同的`Environment`接口
- 可以轻松切换数据源而不修改其他代码

### 2. Episode管理
- 支持选择任意episode进行推理
- 可以从episode的任意帧开始
- 自动检测episode结束

### 3. 灵活的配置
- 通过命令行参数配置所有选项
- 支持自定义控制频率
- 可调整action horizon和open-loop horizon

### 4. 离线测试
- 无需真实机器人即可测试策略
- 可重复的测试环境
- 便于调试和分析

## 使用场景

### 1. 策略评估
在采集的演示数据上评估训练后的策略：
```bash
python examples/teleavatar/main_dataset.py \
    --episode-index 0 \
    --dataset-path datasets
```

### 2. 批量测试
测试多个episode：
```bash
for i in {0..9}; do
    python examples/teleavatar/main_dataset.py \
        --episode-index $i \
        --dataset-path datasets
done
```

### 3. 调试策略
使用已知数据序列调试：
```bash
python examples/teleavatar/test_dataset_interface.py
```

### 4. 可视化
生成策略预测用于分析（可配合其他工具）。

## 技术细节

### 状态向量处理
- 数据集：62维（positions + velocities + efforts + ee_poses）
- 推理：48维（只使用positions + velocities + efforts）
- 提取：`state_array[:48]`

### 图像处理
- 保持原始分辨率（480×848 或 1080×1920）
- 确保uint8格式
- 保持(H, W, C)格式
- 策略的`_parse_image`会处理进一步的转换

### Episode边界
- 从`episodes.jsonl`读取每个episode的长度
- 计算累积帧索引以定位全局帧
- 当到达episode末尾时返回None

### 线程安全
- 数据集接口不需要线程同步（单线程顺序读取）
- 与ROS2接口的多线程设计形成对比

## 依赖项

- `lerobot`: LeRobot数据集加载
- `numpy`: 数组操作
- `openpi_client`: OpenPI客户端库
- `tyro`: 命令行参数解析
- 标准库：`logging`, `pathlib`, `json`

## 后续改进建议

1. **性能优化**
   - 预加载多帧数据
   - 缓存解码的图像
   - 并行加载视频

2. **功能扩展**
   - 支持多episode顺序播放
   - 添加随机采样模式
   - 记录预测动作与真实动作的对比

3. **可视化**
   - 实时显示观测和预测
   - 生成对比视频
   - 绘制状态轨迹

4. **评估指标**
   - 计算预测准确度
   - 统计episode成功率
   - 分析动作分布

## 总结

该实现提供了一个完整的数据集推理解决方案，可以：
- ✅ 从LeRobot数据集读取observation
- ✅ 保持与ROS2接口的API一致性
- ✅ 支持灵活的episode和frame导航
- ✅ 集成到现有的OpenPI推理框架
- ✅ 易于测试和调试
- ✅ 完整的文档和示例

通过导入新的接口，用户可以轻松地在数据集模式和ROS2模式之间切换，实现离线和在线推理的无缝过渡。




