#!/usr/bin/env python3
"""
为 LeRobot 数据集添加任务进度标签（高效版本 - 只修改元数据，不重新处理视频）。

用法：
    conda activate lerobot
    python scripts/add_progress_labels.py \
        --input_dataset /media/caslx/1635-A2D7/Data/putplates_20251117 \
        --output_dataset /media/caslx/1635-A2D7/Data/putplates_20251117_with_progress
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_episode_metadata(dataset_path: Path):
    """加载 episode 元数据"""
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    logger.info(f"加载了 {len(episodes)} 个 episodes")
    return episodes


def add_progress_labels_fast(input_path: str, output_path: str, overwrite: bool = False):
    """
    为数据集的每一帧添加线性进度标签（高效版本：只修改元数据）。
    
    Args:
        input_path: 输入数据集路径
        output_path: 输出数据集路径
        overwrite: 是否覆盖已存在的输出数据集
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 检查输入数据集
    if not input_path.exists():
        raise FileNotFoundError(f"输入数据集不存在: {input_path}")
    
    # 处理输出数据集
    if output_path.exists():
        if overwrite:
            logger.warning(f"删除已存在的输出数据集: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"输出数据集已存在: {output_path}。使用 --overwrite 覆盖。")
    
    # 创建输出目录结构
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    (output_path / "data").mkdir(exist_ok=True)
    
    logger.info(f"复制数据集从 {input_path} 到 {output_path}")
    
    # 1. 复制视频文件（如果存在）
    if (input_path / "videos").exists():
        logger.info("复制视频文件...")
        shutil.copytree(input_path / "videos", output_path / "videos")
    
    # 2. 加载 episode 元数据
    episodes_metadata = load_episode_metadata(input_path)
    
    # 3. 读取并更新 info.json
    info_file = input_path / "meta" / "info.json"
    with open(info_file, 'r') as f:
        dataset_info = json.load(f)
    
    # 添加 progress 特征
    dataset_info["features"]["observation.progress"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["progress"]
    }
    
    # 保存更新后的 info.json
    with open(output_path / "meta" / "info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 4. 复制所有元数据文件
    for meta_file in ["tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]:
        src_file = input_path / "meta" / meta_file
        if src_file.exists():
            shutil.copy(src_file, output_path / "meta" / meta_file)
        else:
            logger.warning(f"元数据文件不存在，跳过: {meta_file}")
    
    # 5. 处理数据文件，添加 progress 列
    logger.info("处理数据文件...")
    
    # 查找数据文件（支持两种结构）
    # 结构1: data/chunk-000.parquet (平铺)
    # 结构2: data/chunk-000/episode_000000.parquet (分层)
    data_files = []
    data_dir = input_path / "data"
    
    # 先尝试查找 chunk-*.parquet (平铺结构)
    flat_files = sorted(data_dir.glob("chunk-*.parquet"))
    if flat_files:
        data_files = flat_files
        logger.info(f"发现平铺结构数据文件: {len(data_files)} 个")
    else:
        # 尝试查找 chunk-*/episode_*.parquet (分层结构)
        chunk_dirs = sorted(data_dir.glob("chunk-*"))
        for chunk_dir in chunk_dirs:
            if chunk_dir.is_dir():
                episode_files = sorted(chunk_dir.glob("episode_*.parquet"))
                data_files.extend(episode_files)
        logger.info(f"发现分层结构数据文件: {len(data_files)} 个")
    
    if not data_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件")
    
    # 计算全局帧到 episode 的映射（离散化为 0-100 的整数百分比）
    frame_to_episode = {}
    global_idx = 0
    for ep_idx, episode in enumerate(episodes_metadata):
        ep_length = episode["length"]
        for frame_idx in range(ep_length):
            # 连续进度值
            continuous_progress = frame_idx / max(1, ep_length - 1)
            # 离散化为 0-100 的整数百分比，然后归一化到 [0, 1]
            discrete_percent = int(round(continuous_progress * 100))
            discrete_percent = min(100, max(0, discrete_percent))  # 确保在 [0, 100]
            discrete_progress = discrete_percent / 100.0  # 转回 [0, 1] 范围
            frame_to_episode[global_idx] = discrete_progress
            global_idx += 1
    
    # 处理每个数据文件
    for data_file in tqdm(data_files, desc="处理数据文件"):
        # 读取原始 parquet 文件
        table = pq.read_table(data_file)
        
        # 获取帧索引
        frame_indices = table.column('index').to_pylist()
        
        # 为这些帧创建 progress 值
        progress_values = [frame_to_episode.get(idx, 0.0) for idx in frame_indices]
        
        # 创建 progress 列（作为 (N, 1) 的 FixedSizeList）
        progress_array = pa.array([np.array([p], dtype=np.float32) for p in progress_values], 
                                   type=pa.list_(pa.float32(), 1))
        
        # 添加新列到表
        new_table = table.append_column('observation.progress', progress_array)
        
        # 保存到输出路径（保持相同的目录结构）
        rel_path = data_file.relative_to(input_path / "data")
        output_file = output_path / "data" / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, output_file)
    
    logger.info(f"✅ 成功创建带进度标签的数据集: {output_path}")
    logger.info(f"   总帧数: {global_idx}")
    logger.info(f"   Episodes: {len(episodes_metadata)}")
    
    # 验证几个样本（直接读取 parquet 文件）
    logger.info("\n验证样本数据:")
    
    # 查找第一个数据文件进行验证
    first_file = None
    if (output_path / "data" / "chunk-000.parquet").exists():
        first_file = output_path / "data" / "chunk-000.parquet"
    elif (output_path / "data" / "chunk-000").exists():
        episode_files = sorted((output_path / "data" / "chunk-000").glob("episode_*.parquet"))
        if episode_files:
            first_file = episode_files[0]
    
    if first_file and first_file.exists():
        table = pq.read_table(first_file)
        if 'observation.progress' in table.column_names:
            progress_col = table.column('observation.progress')
            index_col = table.column('index')
            # 取前3个样本验证
            for i in range(min(3, len(progress_col))):
                progress_val = progress_col[i].as_py()[0]  # 获取第一个元素
                frame_idx = index_col[i].as_py()
                logger.info(f"  帧 {frame_idx}: progress = {progress_val:.4f}")
            logger.info(f"  ✓ progress 字段已成功添加到 {len(data_files)} 个文件")
        else:
            logger.error("  ✗ progress 字段未找到！")
    else:
        logger.warning("  无法验证：未找到数据文件")


def main():
    parser = argparse.ArgumentParser(description="为 LeRobot 数据集添加任务进度标签")
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="输入数据集路径"
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        required=True,
        help="输出数据集路径"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出数据集"
    )
    
    args = parser.parse_args()
    
    add_progress_labels_fast(
        input_path=args.input_dataset,
        output_path=args.output_dataset,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()

