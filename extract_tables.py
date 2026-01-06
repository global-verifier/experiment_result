#!/usr/bin/env python3
"""
从实验结果目录提取分数，生成表格格式的CSV。

表格结构:
- 行: no-memory, vanilla, vanilla-glove, memorybank, memorybank-glove, voyager, voyager-glove, generative, generative-glove
- 列: webshop(env0,env1,env2), frozen lake(env0,env1,env2), mountain car(env0,env1,env2)
"""

import csv
import re
from pathlib import Path
from collections import defaultdict


# 已知的memory类型
MEMORY_TYPES = {"vanilla", "memorybank", "voyager", "generative"}


def parse_log_dir_name(dir_name: str) -> dict:
    """
    智能解析日志目录名，提取配置信息。
    
    支持的格式:
    - log_{env}_{model}_{memory}_{use_memory}_{use_glove}
    - log_{env}_{model}_{memory}_{use_memory}
    - log_{env}_{model}_{memory}__{use_memory}_{use_glove}  (双下划线)
    - log_hidden_{env}_{model}_{memory}_{use_memory}_{use_glove}
    - log_hidden_{env}_{memory}_{model}_{use_memory}_{use_glove} (frozenlake implicit格式)
    """
    # 移除 log_ 或 log_hidden_ 前缀
    is_implicit = dir_name.startswith("log_hidden_")
    
    if is_implicit:
        remainder = dir_name[len("log_hidden_"):]
    else:
        remainder = dir_name[len("log_"):]
    
    # 分割剩余部分，处理双下划线
    parts = remainder.replace("__", "_").split("_")
    
    # 提取环境名（第一部分）
    env = parts[0]
    
    # 在剩余部分中找memory类型
    memory = None
    memory_idx = -1
    for i, part in enumerate(parts[1:], 1):
        if part in MEMORY_TYPES:
            memory = part
            memory_idx = i
            break
    
    if memory is None:
        return None
    
    # model 在 memory 之前或之后
    # 通常格式是 env_model_memory_... 或 env_memory_model_...
    model = None
    for i, part in enumerate(parts[1:memory_idx], 1):
        # model 通常包含 gpt 或 llama
        if "gpt" in part.lower() or "llama" in part.lower():
            model = part
            break
    
    # 如果在memory之前没找到model，在memory之后找
    if model is None:
        for part in parts[memory_idx + 1:]:
            if "gpt" in part.lower() or "llama" in part.lower():
                model = part
                break
    
    # 提取 use_memory 和 use_glove
    # 查找 True/False 值
    bool_values = []
    for part in parts[memory_idx + 1:]:
        if part in ("True", "False"):
            bool_values.append(part)
    
    use_memory = bool_values[0] if len(bool_values) > 0 else "True"
    use_glove = bool_values[1] if len(bool_values) > 1 else "False"
    
    return {
        "env": env,
        "model": model,
        "memory": memory,
        "use_memory": use_memory,
        "use_glove": use_glove,
        "is_implicit": is_implicit,
    }


def get_row_name(memory: str, use_memory: str, use_glove: str) -> str:
    """根据配置生成行名。"""
    if use_memory == "False":
        return "no-memory"
    
    base = memory  # vanilla, memorybank, voyager, generative
    if use_glove == "True":
        return f"{base}-glove"
    return base


def calculate_env_averages(values: list, items_per_env: int = 20) -> list:
    """计算每个环境的平均值。"""
    averages = []
    for i in range(0, len(values), items_per_env):
        chunk = values[i:i + items_per_env]
        if chunk:
            numeric_values = [float(v) for v in chunk]
            avg = sum(numeric_values) / len(numeric_values)
            averages.append(avg)
    return averages


def extract_scores_from_csv(csv_path: Path) -> list:
    """从CSV文件提取最后一列的分数。"""
    values = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过标题行
            for row in reader:
                if row:
                    values.append(row[-1])
    except Exception as e:
        print(f"读取 {csv_path} 失败: {e}")
    return values


def match_model_name(subdir_name: str, model_name: str) -> bool:
    """检查目录名是否匹配模型名。"""
    name_lower = subdir_name.lower()
    
    if model_name == "gpt-4o":
        # 匹配 gpt4o, gpt40, gpt-4o 等
        return "gpt4o" in name_lower or "gpt40" in name_lower or "gpt-4o" in name_lower
    elif model_name == "llama3.1-8b":
        # 匹配 llama3.1_8b, llama3.1-8b, llama31-8b 等
        return ("llama3.1" in name_lower or "llama31" in name_lower) and "8b" in name_lower
    else:
        return model_name.lower() in name_lower


def match_env_name(subdir_name: str) -> str:
    """从目录名中提取环境名。"""
    name_lower = subdir_name.lower().replace("-", "").replace("_", "")
    
    if "webshop" in name_lower:
        return "webshop"
    elif "frozenlake" in name_lower:
        return "frozenlake"
    elif "mountaincar" in name_lower:
        return "mountaincar"
    return None


def should_skip_dir(subdir_name: str) -> bool:
    """判断是否应该跳过该目录（如 old- 前缀的目录）。"""
    return subdir_name.startswith("old")


def determine_experiment_type(subdir_name: str, log_dirs: list) -> str:
    """
    判断实验类型 (explicit/implicit)。
    
    1. 如果目录名包含 explicit/implicit，直接使用
    2. 否则，检查子目录是否有 log_hidden_ 前缀来判断
    """
    name_lower = subdir_name.lower()
    
    if "explicit" in name_lower:
        return "explicit"
    elif "implicit" in name_lower:
        return "implicit"
    
    # 检查子目录
    for log_dir in log_dirs:
        if log_dir.name.startswith("log_hidden_"):
            return "implicit"
    
    # 默认为 explicit（没有 hidden 前缀）
    return "explicit"


def generate_table(base_dir: str, model_name: str, output_file: str = "table.csv"):
    """生成表格CSV文件。"""
    base_path = Path(base_dir)
    
    # 定义行顺序
    row_order = [
        "no-memory",
        "vanilla", "vanilla-glove",
        "memorybank", "memorybank-glove",
        "voyager", "voyager-glove",
        "generative", "generative-glove",
    ]
    
    # 定义环境顺序
    envs = ["webshop", "frozenlake", "mountaincar"]
    
    # 数据结构: data[explicit/implicit][row_name][env] = [env0_avg, env1_avg, env2_avg]
    data = {
        "explicit": defaultdict(lambda: defaultdict(list)),
        "implicit": defaultdict(lambda: defaultdict(list)),
    }
    
    # 遍历所有目录查找CSV文件
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        subdir_name = subdir.name
        
        # 跳过 old- 前缀的目录
        if should_skip_dir(subdir_name):
            print(f"跳过目录: {subdir_name}")
            continue
        
        # 根据目录名确定模型
        if not match_model_name(subdir_name, model_name):
            continue
        
        # 确定环境
        env = match_env_name(subdir_name)
        if not env:
            continue
        
        # 获取子目录列表
        log_dirs = [d for d in subdir.iterdir() if d.is_dir()]
        
        # 确定是 explicit 还是 implicit
        experiment_type = determine_experiment_type(subdir_name, log_dirs)
        
        print(f"\n扫描目录: {subdir_name} ({experiment_type}, {env})")
        
        # 遍历子目录查找日志
        for log_dir in sorted(log_dirs):
            csv_path = log_dir / "log" / "explorer_summary.csv"
            if not csv_path.exists():
                continue
            
            config = parse_log_dir_name(log_dir.name)
            if config is None:
                print(f"  跳过无法解析: {log_dir.name}")
                continue
            
            row_name = get_row_name(config["memory"], config["use_memory"], config["use_glove"])
            
            values = extract_scores_from_csv(csv_path)
            averages = calculate_env_averages(values, 20)
            
            data[experiment_type][row_name][env] = averages
            print(f"  -> {row_name}: {averages}")
    
    # 生成CSV
    output_path = base_path / output_file
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 标题行1: 模型名
        writer.writerow([model_name] + [""] * 10)
        
        # 标题行2: 环境大类
        writer.writerow(["", "", "webshop", "", "", "frozen lake", "", "", "mountain car", "", ""])
        
        # 标题行3: env0, env1, env2
        writer.writerow(["", "", "env0", "env1", "env2", "env0", "env1", "env2", "env0", "env1", "env2"])
        
        # explicit 部分
        first_explicit = True
        for row_name in row_order:
            row_data = []
            if first_explicit:
                row_data.append("explicit")
                first_explicit = False
            else:
                row_data.append("")
            
            row_data.append(row_name)
            
            for env in envs:
                averages = data["explicit"].get(row_name, {}).get(env, [])
                for i in range(3):
                    if i < len(averages):
                        row_data.append(f"{averages[i]:.4f}")  # 0 也要显示
                    else:
                        row_data.append("")
            
            writer.writerow(row_data)
        
        # 空行 + 重复标题
        writer.writerow(["", "", "env0", "env1", "", "env0", "env1", "", "", "", ""])
        
        # implicit 部分
        first_implicit = True
        for row_name in row_order:
            row_data = []
            if first_implicit:
                row_data.append("implicit")
                first_implicit = False
            else:
                row_data.append("")
            
            row_data.append(row_name)
            
            for env in envs:
                averages = data["implicit"].get(row_name, {}).get(env, [])
                for i in range(3):
                    if i < len(averages):
                        row_data.append(f"{averages[i]:.4f}")  # 0 也要显示
                    else:
                        row_data.append("")
            
            writer.writerow(row_data)
    
    print(f"\n表格已生成: {output_path}")


if __name__ == "__main__":
    # 生成 gpt-4o 的表格
    generate_table("/home/xingkun/experiment_result", "gpt-4o", "table_gpt4o.csv")
    
    # 生成 llama3.1-8b 的表格
    generate_table("/home/xingkun/experiment_result", "llama3.1-8b", "table_llama3.1-8b.csv")
