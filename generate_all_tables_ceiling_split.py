#!/usr/bin/env python3
"""
为所有模型生成表格CSV文件（ceiling版本 + explicit/implicit分离版本）。

参考 check_integrity.py 的模型定义和 extract_tables.py 的表格生成逻辑。
如果某个cell的数据不存在或数据点少于20个，则该cell留空。

与 generate_all_tables.py 的区别：
- 在统计数据时，首先对最后一列做 ceiling 处理：
  - 0 保持为 0
  - 所有非 0 的数字变成 1
- 然后再计算平均值（即计算非零率/成功率）
- explicit 和 implicit 分开成两个独立的表格文件
"""

import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 配置
BASE_DIR = Path("/data/xingkun/experiment_result")

# 模型列表（按指定顺序，与 check_integrity.py 一致）
# model_folder: (env_folder_prefix, [possible_method_folder_model_names], display_name)
MODELS = {
    "llama3.1_8b": ("llama3.1_8b", ["llama3.1_8b", "llama3.1-8b"], "Llama3.1-8B"),
    "llama-3.3-70b-instruct": ("llama-3.3-70b-instruct", ["llama-3.3-70b-instruct"], "Llama3.3-70B"),
    "qwen2.5-7b": ("qwen2.5-7b-instruct", ["qwen2.5-7b"], "Qwen2.5-7B"),
    "qwen3-30b": ("qwen3-30b-instruct", ["qwen3-30b"], "Qwen3-30B"),
    "gpt4o": ("gpt4o", ["gpt4o", "gpt-4o"], "GPT-4o"),
    "grok-3": ("grok-3", ["grok-3"], "Grok-3"),
    "deepseek-r1": ("deepseek-r1", ["deepseek-r1"], "DeepSeek-R1"),
}

# 环境列表
ENVIRONMENTS = [
    ("webshop-explicit", "webshop", "explicit"),
    ("webshop-implicit", "webshop", "implicit"),
    ("frozenlake-explicit", "frozenlake", "explicit"),
    ("frozenlake-implicit", "frozenlake", "implicit"),
    ("mountaincar", "mountaincar", "explicit"),  # mountaincar 只有 explicit
]

# 方法映射: (memory_type, use_memory, use_glove) -> row_name
METHOD_TO_ROW = {
    ("vanilla", "False", "False"): "no-memory",
    ("vanilla", "True", "False"): "vanilla",
    ("vanilla", "True", "True"): "vanilla-glove",
    ("memorybank", "True", "False"): "memorybank",
    ("memorybank", "True", "True"): "memorybank-glove",
    ("voyager", "True", "False"): "voyager",
    ("voyager", "True", "True"): "voyager-glove",
    ("generative", "True", "False"): "generative",
    ("generative", "True", "True"): "generative-glove",
}

# 方法列表（与 check_integrity.py 一致）
METHODS = [
    "generative_True_False",
    "generative_True_True",
    "memorybank_True_False",
    "memorybank_True_True",
    "vanilla_False_False",
    "vanilla_True_False",
    "vanilla_True_True",
    "voyager_True_False",
    "voyager_True_True",
]

# 行顺序
ROW_ORDER = [
    "no-memory",
    "vanilla", "vanilla-glove",
    "memorybank", "memorybank-glove",
    "voyager", "voyager-glove",
    "generative", "generative-glove",
]

# 最小数据点要求
MIN_DATA_POINTS = 20


def find_env_folder(model_dir: Path, prefix: str, env: str) -> Path | None:
    """查找环境文件夹"""
    possible_names = [f"{prefix}-{env}", f"{prefix}_{env}"]
    
    for name in possible_names:
        folder = model_dir / name
        if folder.exists():
            return folder
    
    # 如果都不存在，尝试搜索
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir() and env in item.name:
                return item
    
    return None


def get_env_short_name(env: str) -> str:
    """获取环境的简短名称"""
    if env.startswith("frozenlake"):
        return "frozenlake"
    elif env.startswith("webshop"):
        return "webshop"
    else:
        return env


def get_log_folder_name(env: str, model_prefix: str, method: str) -> str:
    """生成日志文件夹名称"""
    env_short = get_env_short_name(env)
    if "implicit" in env:
        return f"log_hidden_{env_short}_{model_prefix}_{method}"
    else:
        return f"log_{env_short}_{model_prefix}_{method}"


def find_method_folder(env_folder: Path, model_prefix: str, env: str, method: str) -> Path | None:
    """查找方法文件夹"""
    expected_name = get_log_folder_name(env, model_prefix, method)
    folder = env_folder / expected_name
    if folder.exists():
        return folder
    
    # 尝试搜索包含方法名的文件夹
    for item in env_folder.iterdir():
        if item.is_dir() and method in item.name:
            return item
    
    return None


def parse_method(method: str) -> tuple:
    """解析方法字符串为 (memory_type, use_memory, use_glove)"""
    parts = method.split("_")
    return (parts[0], parts[1], parts[2])


def ceiling_value(value: float) -> float:
    """
    对值进行 ceiling 处理：
    - 0 保持为 0
    - 所有非 0 的数字变成 1
    """
    if value == 0:
        return 0.0
    else:
        return 1.0


def extract_scores_from_csv(csv_path: Path) -> list:
    """
    从CSV文件提取最后一列的分数，并进行 ceiling 处理。
    - 0 保持为 0
    - 所有非 0 的数字变成 1
    """
    values = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过标题行
            for row in reader:
                if row:
                    try:
                        original_value = float(row[-1])
                        # 应用 ceiling 处理
                        ceiling_val = ceiling_value(original_value)
                        values.append(ceiling_val)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"读取 {csv_path} 失败: {e}")
    return values


def calculate_env_averages(values: list, items_per_env: int = 20) -> list:
    """
    计算每个环境的平均值。
    如果某个环境的数据点少于 MIN_DATA_POINTS，返回 None 表示该环境数据不全。
    """
    averages = []
    for i in range(0, len(values), items_per_env):
        chunk = values[i:i + items_per_env]
        if len(chunk) >= MIN_DATA_POINTS:
            avg = sum(chunk) / len(chunk)
            averages.append(avg)
        else:
            averages.append(None)  # 数据不全，留空
    return averages


def generate_model_table(model_folder: str, model_prefix: str, model_variants: list, display_name: str):
    """为单个模型生成表格数据"""
    model_dir = BASE_DIR / model_folder
    
    if not model_dir.exists():
        print(f"  模型目录不存在: {model_dir}")
        return None
    
    # 数据结构: data[explicit/implicit][row_name][env] = [env0_avg, env1_avg, env2_avg]
    data = {
        "explicit": defaultdict(lambda: defaultdict(list)),
        "implicit": defaultdict(lambda: defaultdict(list)),
    }
    
    for env_name, env_short, exp_type in ENVIRONMENTS:
        env_folder = find_env_folder(model_dir, model_prefix, env_name)
        if not env_folder:
            continue
        
        for method in METHODS:
            method_folder = find_method_folder(env_folder, model_prefix, env_name, method)
            if not method_folder:
                continue
            
            csv_path = method_folder / "log" / "explorer_summary.csv"
            if not csv_path.exists():
                continue
            
            # 解析方法获取行名
            memory_type, use_memory, use_glove = parse_method(method)
            row_name = METHOD_TO_ROW.get((memory_type, use_memory, use_glove))
            if not row_name:
                continue
            
            # 提取分数（已经过 ceiling 处理）并计算平均值
            values = extract_scores_from_csv(csv_path)
            averages = calculate_env_averages(values, 20)
            
            data[exp_type][row_name][env_short] = averages
    
    return data


def write_explicit_csv(model_data: dict, display_name: str, output_file: Path):
    """为单个模型写入 explicit CSV文件"""
    envs = ["webshop", "frozenlake", "mountaincar"]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 标题行1: 模型名
        writer.writerow([display_name] + [""] * 9)
        
        # 标题行2: 环境大类
        writer.writerow(["", "webshop", "", "", "frozen lake", "", "", "mountain car", "", ""])
        
        # 标题行3: env0, env1, env2
        writer.writerow(["", "env0", "env1", "env2", "env0", "env1", "env2", "env0", "env1", "env2"])
        
        # explicit 数据行
        for row_name in ROW_ORDER:
            row_data = [row_name]
            
            for env in envs:
                averages = model_data["explicit"].get(row_name, {}).get(env, []) if model_data else []
                
                for i in range(3):
                    if i < len(averages) and averages[i] is not None:
                        row_data.append(f"{averages[i]:.4f}")
                    else:
                        row_data.append("")
            
            writer.writerow(row_data)


def write_implicit_csv(model_data: dict, display_name: str, output_file: Path):
    """为单个模型写入 implicit CSV文件"""
    envs = ["webshop", "frozenlake"]  # implicit 只有 webshop 和 frozenlake
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 标题行1: 模型名
        writer.writerow([display_name] + [""] * 4)
        
        # 标题行2: 环境大类
        writer.writerow(["", "webshop", "", "frozen lake", ""])
        
        # 标题行3: env0, env1 (implicit 只有 2 个环境)
        writer.writerow(["", "env0", "env1", "env0", "env1"])
        
        # implicit 数据行
        for row_name in ROW_ORDER:
            row_data = [row_name]
            
            for env in envs:
                averages = model_data["implicit"].get(row_name, {}).get(env, []) if model_data else []
                
                for i in range(2):  # implicit 只有 env0, env1
                    if i < len(averages) and averages[i] is not None:
                        row_data.append(f"{averages[i]:.4f}")
                    else:
                        row_data.append("")
            
            writer.writerow(row_data)


def main():
    print("🔍 开始为所有模型生成表格（ceiling版本 + explicit/implicit 分离版本）...")
    
    all_data = {}
    
    for model_folder, (model_prefix, model_variants, display_name) in MODELS.items():
        print(f"\n处理模型: {display_name} ({model_folder})")
        model_data = generate_model_table(model_folder, model_prefix, model_variants, display_name)
        all_data[model_folder] = model_data
    
    # 为每个模型生成 explicit 和 implicit 两个独立的表格
    for model_folder, (model_prefix, model_variants, display_name) in MODELS.items():
        model_data = all_data.get(model_folder)
        if model_data:
            # explicit 表格
            explicit_output = BASE_DIR / f"table_ceiling_{model_folder}_explicit.csv"
            write_explicit_csv(model_data, display_name, explicit_output)
            print(f"  生成: {explicit_output}")
            
            # implicit 表格
            implicit_output = BASE_DIR / f"table_ceiling_{model_folder}_implicit.csv"
            write_implicit_csv(model_data, display_name, implicit_output)
            print(f"  生成: {implicit_output}")


if __name__ == "__main__":
    main()


