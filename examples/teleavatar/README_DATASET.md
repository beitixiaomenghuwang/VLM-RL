# Teleavatar 数据集推理模式

本模块提供了使用LeRobot格式数据集进行离线推理测试的功能，可以替代真实机器人的ROS2接口进行策略评估。

## 文件说明

- `dataset_interface.py`: 从LeRobot数据集读取observation的接口类，提供与`ros2_interface.py`相同的API
- `env_dataset.py`: 使用数据集接口的环境包装器
- `main_dataset.py`: 数据集推理的主入口脚本

## 使用方法

### 1. 启动策略服务器

首先在一个终端启动策略服务器：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_low_mem_finetune \
    --policy.dir=pi0_teleavatar_low_mem_finetune_new_data/pi0_lora_with_joint_positions_and_gripper_efforts_new_data/29999
```

### 2. 运行数据集推理

#### 模式A：纯数据集推理（不控制机器人，默认模式）

```bash
python examples/teleavatar/main_dataset.py \
    --remote-host 127.0.0.1 \
    --dataset-path /home/caslx/Robotics/openpi/datasets \
    --episode-index 0 \
    --prompt "Pick up the toy and drop it in the basket on the left"
```

这个模式只读取数据集进行推理，不会发布动作到ROS2，适合离线测试。

#### 模式B：数据集推理 + ROS2动作发布（控制真实机器人）

```bash
python examples/teleavatar/main_dataset.py \
    --remote-host 127.0.0.1 \
    --dataset-path /home/caslx/Robotics/openpi/datasets \
    --episode-index 0 \
    --enable-ros2-publishing  # 启用真实机器人控制
```

**⚠️ 警告**：使用`--enable-ros2-publishing`时机器人会真正移动！详见`README_DATASET_WITH_ROS2.md`

**重要提示**：`--dataset-path`参数会自动转换为绝对路径，但建议直接使用绝对路径以避免混淆。

### 参数说明

- `--remote-host`: 策略服务器的IP地址（默认：127.0.0.1）
- `--remote-port`: 策略服务器端口（默认：8000）
- `--dataset-path`: LeRobot数据集路径（默认：/home/caslx/Robotics/openpi/datasets）
- `--episode-index`: 要播放的episode索引（默认：0）
- `--start-frame`: episode中的起始帧（默认：0）
- `--control-frequency`: 控制频率Hz（默认：20.0）
- `--action-horizon`: 策略返回的动作数量（默认：10）
- `--open-loop-horizon`: 执行多少个动作后再次查询策略（默认：8）
- `--prompt`: 语言指令（默认："Pick up the toy and drop it in the basket on the left"）
- `--enable-ros2-publishing`: 启用ROS2动作发布，控制真实机器人（默认：False）⚠️
- `--num-episodes`: 运行的episode数量（默认：1）
- `--max-episode-steps`: 每个episode的最大步数（0=使用数据集长度）

## 数据集格式要求

数据集必须是LeRobot v2.1格式，包含以下结构：

```
datasets/
├── data/
│   └── chunk-000/
│       └── episode_*.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.left_wrist/
│       ├── observation.images.right_wrist/
│       └── observation.images.head/
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

### 必需的特征字段

- **observation.state**: 62维状态向量（前48维将被使用）
    - 索引 0-15: positions（左臂7个关节 + 左夹爪 + 右臂7个关节 + 右夹爪）
    - 索引 16-31: velocities
    - 索引 32-47: efforts
    - 索引 48-61: end effector poses（将被忽略）

- **observation.images.left_wrist**: 左手腕相机图像（480×848×3）
- **observation.images.right_wrist**: 右手腕相机图像（480×848×3）
- **observation.images.head**: 头部相机图像（1080×1920×3）

## 接口映射

数据集键名到环境观测键名的映射：

| 数据集键名 | 环境观测键名 | 说明 |
|-----------|-------------|------|
| `observation.images.left_wrist` | `observation/images/left_color` | 左手腕相机 |
| `observation.images.right_wrist` | `observation/images/right_color` | 右手腕相机 |
| `observation.images.head` | `observation/images/head_camera` | 头部相机 |
| `observation.state[:48]` | `observation/state` | 48维状态向量 |

## 测试数据集接口

可以直接运行`dataset_interface.py`来测试数据加载：

```bash
python examples/teleavatar/dataset_interface.py
```

这将：

1. 加载数据集
2. 读取第0个episode的前5帧
3. 打印状态和图像信息
4. 显示episode元数据

## 与ROS2接口的区别

### 相同点

- 提供相同的`get_observation()`方法
- 返回相同格式的observation字典
- 支持`wait_for_initial_data()`方法
- 提供`publish_action()`方法（数据集模式下为空操作）

### 不同点

| 特性 | ROS2接口 | 数据集接口 |
|-----|---------|-----------|
| 数据源 | 实时ROS2话题 | 本地数据集文件 |
| Episode完成 | 永不完成（需手动停止） | 到达episode末尾时完成 |
| 动作发布 | 发布到ROS2话题 | 不执行（空操作） |
| 时序 | 实时传感器数据 | 按控制频率播放录制数据 |

## 应用场景

1. **策略评估**: 在不需要真实机器人的情况下测试训练好的策略
2. **调试**: 使用已知的数据序列调试策略推理流程
3. **可视化**: 生成策略预测的动作序列用于分析
4. **基准测试**: 比较不同checkpoint的性能

## 注意事项

1. **控制频率**: 建议将`--control-frequency`设置为与数据集FPS相同（默认30Hz）以获得真实的时序
2. **图像格式**: 图像保持原始分辨率，与训练数据一致
3. **状态维度**: 数据集的62维状态中只使用前48维（positions + velocities + efforts）
4. **Episode长度**: 每个episode的长度由数据集决定，可以通过`max_episode_steps`限制
5. **无动作执行**: 在数据集模式下，`apply_action()`不会执行任何操作

## 示例：多episode推理

```bash
# 推理前3个episode
for i in 0 1 2; do
    echo "Running episode $i..."
    python examples/teleavatar/main_dataset.py \
        --episode-index $i \
        --dataset-path datasets \
        --remote-host 127.0.0.1
done
```

## 故障排除

### 问题：找不到数据集

**错误信息**:

```
FileNotFoundError: [Errno 2] No such file or directory: '.../datasets/meta/info.json'
```

**解决方案**:

- 确认`--dataset-path`指向正确的数据集目录
- 检查数据集是否包含`meta/info.json`文件
- 使用绝对路径而不是相对路径

### 问题：图像加载失败

**错误信息**:

```
Error loading frame X: ...
```

**解决方案**:

- 确认`videos/`目录包含所有必需的视频文件
- 检查视频文件是否损坏
- 确认LeRobot库版本兼容（推荐使用v2.1+）

### 问题：状态维度不匹配

**错误信息**:

```
IndexError: index 48 is out of bounds for axis 0 with size X
```

**解决方案**:

- 确认数据集的`observation.state`维度至少为48
- 检查数据集的`meta/info.json`中features定义
- 确认使用的是teleavatar格式的数据集

## 相关文件

- `main.py`: 真实机器人ROS2推理脚本
- `ros2_interface.py`: ROS2接口实现
- `env.py`: ROS2环境包装器
- `convert_teleavatar_data_to_lerobot.py`: 数据集转换脚本
