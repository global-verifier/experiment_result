#!/usr/bin/env python3
"""
为 frozenlak_explicit 文件夹中的每个版本（v0-v4）单独生成表格。

参考 generate_all_tables_split.py 的逻辑。
每个版本生成一个独立的 CSV 表格文件。
数据按 20 个一组分成 env0, env1, env2。
"""

import csv
from pathlib import Path
from collections import defaultdict

# 配置
BASE_DIR = Path("/data/xingkun/experiment_result/frozenlak_explicit")
OUTPUT_DIR = Path("/data/xingkun/experiment_result")

# 版本列表
VERSIONS = ["v0", "v1", "v2", "v3", "v4"]

# 模型映射: 文件夹名称模式 -> (log文件夹中的模型名, 显示名称)
# 不同版本的文件夹命名可能不同，需要灵活匹配
MODEL_PATTERNS = {
    "llama3.1_8b": (["llama3.1-8b", "llama3.1_8b"], "Llama3.1-8B"),
    "llama-3.3-70b": (["llama-3.3-70b-instruct", "llama-3.3-70b"], "Llama3.3-70B"),
    "qwen2.5-7b": (["qwen2.5-7b", "qwen2.5-7b-instruct"], "Qwen2.5-7B"),
    "qwen3-30b": (["qwen3-30b", "qwen3-30b-instruct"], "Qwen3-30B"),
    "gpt4o": (["gpt-4o", "gpt4o"], "GPT-4o"),
    "grok-3": (["grok-3"], "Grok-3"),
    "deepseek-r1": (["deepseek-r1"], "DeepSeek-R1"),
    "deepseek-v3.2": (["deepseek-v3.2"], "DeepSeek-V3.2"),
}

# 方法列表
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

# 行顺序
ROW_ORDER = [
    "no-memory",
    "vanilla", "vanilla-glove",
    "memorybank", "memorybank-glove",
    "voyager", "voyager-glove",
    "generative", "generative-glove",
]

# 模型显示顺序
MODEL_ORDER = [
    "Llama3.1-8B",
    "Llama3.3-70B",
    "Qwen2.5-7B",
    "Qwen3-30B",
    "GPT-4o",
    "Grok-3",
    "DeepSeek-R1",
    "DeepSeek-V3.2",
]

# 最小数据点要求
MIN_DATA_POINTS = 20
ITEMS_PER_ENV = 20


def parse_method(method: str) -> tuple:
    """解析方法字符串为 (memory_type, use_memory, use_glove)"""
    parts = method.split("_")
    return (parts[0], parts[1], parts[2])


def extract_scores_from_csv(csv_path: Path) -> list:
    """从CSV文件提取最后一列的分数"""
    values = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过标题行
            for row in reader:
                if row:
                    try:
                        values.append(float(row[-1]))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"  读取 {csv_path} 失败: {e}")
    return values


def calculate_env_averages(values: list, items_per_env: int = ITEMS_PER_ENV) -> list:
    """
    计算每个环境的平均值。
    将数据按 items_per_env 分组，返回 [env0_avg, env1_avg, env2_avg]。
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


def find_model_folder(version_dir: Path, model_key: str) -> Path | None:
    """查找模型文件夹"""
    if not version_dir.exists():
        return None
    
    for item in version_dir.iterdir():
        if not item.is_dir():
            continue
        item_name = item.name.lower()
        
        # 匹配模型关键字
        if model_key.lower() in item_name:
            return item
    
    return None


def find_log_folder(model_folder: Path, model_variants: list, method: str) -> Path | None:
    """查找日志文件夹"""
    for variant in model_variants:
        log_name = f"log_frozenlake_{variant}_{method}"
        log_folder = model_folder / log_name
        if log_folder.exists():
            return log_folder
    
    # 尝试搜索包含方法名的文件夹
    for item in model_folder.iterdir():
        if item.is_dir() and method in item.name:
            return item
    
    return None


def process_version(version: str) -> dict:
    """处理单个版本，返回数据字典"""
    version_dir = BASE_DIR / version
    print(f"\n处理版本: {version}")
    
    if not version_dir.exists():
        print(f"  版本目录不存在: {version_dir}")
        return {}
    
    # 数据结构: data[display_name][row_name] = [env0_avg, env1_avg, env2_avg]
    data = defaultdict(dict)
    
    for model_key, (model_variants, display_name) in MODEL_PATTERNS.items():
        model_folder = find_model_folder(version_dir, model_key)
        if not model_folder:
            print(f"  未找到模型: {model_key}")
            continue
        
        print(f"  处理模型: {display_name} ({model_folder.name})")
        
        for method in METHODS:
            log_folder = find_log_folder(model_folder, model_variants, method)
            if not log_folder:
                continue
            
            csv_path = log_folder / "log" / "explorer_summary.csv"
            if not csv_path.exists():
                continue
            
            # 解析方法获取行名
            memory_type, use_memory, use_glove = parse_method(method)
            row_name = METHOD_TO_ROW.get((memory_type, use_memory, use_glove))
            if not row_name:
                continue
            
            # 提取分数并计算每个环境的平均值
            values = extract_scores_from_csv(csv_path)
            env_averages = calculate_env_averages(values, ITEMS_PER_ENV)
            
            data[display_name][row_name] = env_averages
            
            # 打印调试信息
            avg_strs = [f"{a:.4f}" if a is not None else "N/A" for a in env_averages]
            print(f"    {row_name}: {avg_strs} ({len(values)} 数据点)")
    
    return data


def write_version_csv(version: str, data: dict, output_file: Path):
    """为单个版本写入 CSV 文件"""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 获取该版本中存在的模型（按顺序）
        available_models = [m for m in MODEL_ORDER if m in data]
        
        if not available_models:
            print(f"  {version} 没有可用数据，跳过生成表格")
            return
        
        # 标题行1: 版本信息
        header1 = [f"FrozenLake Explicit - {version}"]
        for model in available_models:
            header1.extend([model, "", ""])
        writer.writerow(header1)
        
        # 标题行2: env0, env1, env2 (每个模型重复)
        header2 = ["Method"]
        for _ in available_models:
            header2.extend(["env0", "env1", "env2"])
        writer.writerow(header2)
        
        # 数据行
        for row_name in ROW_ORDER:
            row_data = [row_name]
            for model in available_models:
                env_averages = data.get(model, {}).get(row_name, [])
                for i in range(3):
                    if i < len(env_averages) and env_averages[i] is not None:
                        row_data.append(f"{env_averages[i]:.4f}")
                    else:
                        row_data.append("")
            writer.writerow(row_data)
    
    print(f"  生成: {output_file}")


def write_summary_csv(all_data: dict, output_file: Path):
    """生成汇总表格，包含所有版本"""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 标题行1: 版本
        header1 = [""]
        for version in VERSIONS:
            # 每个版本有 7 个模型 × 3 个 env = 21 列
            header1.extend([version] + [""] * (len(MODEL_ORDER) * 3 - 1))
        writer.writerow(header1)
        
        # 标题行2: 模型名称（每个版本重复）
        header2 = [""]
        for version in VERSIONS:
            for model in MODEL_ORDER:
                header2.extend([model, "", ""])
        writer.writerow(header2)
        
        # 标题行3: env0, env1, env2
        header3 = ["Method"]
        for version in VERSIONS:
            for _ in MODEL_ORDER:
                header3.extend(["env0", "env1", "env2"])
        writer.writerow(header3)
        
        # 数据行
        for row_name in ROW_ORDER:
            row_data = [row_name]
            for version in VERSIONS:
                version_data = all_data.get(version, {})
                for model in MODEL_ORDER:
                    env_averages = version_data.get(model, {}).get(row_name, [])
                    for i in range(3):
                        if i < len(env_averages) and env_averages[i] is not None:
                            row_data.append(f"{env_averages[i]:.4f}")
                        else:
                            row_data.append("")
            writer.writerow(row_data)
    
    print(f"\n生成汇总表格: {output_file}")


def main():
    print("🔍 开始为 frozenlak_explicit 每个版本生成表格（分 env0/env1/env2）...")
    
    all_data = {}
    
    # 处理每个版本
    for version in VERSIONS:
        data = process_version(version)
        all_data[version] = data
        
        # 为每个版本生成单独的表格
        if data:
            output_file = OUTPUT_DIR / f"table_frozenlake_explicit_{version}.csv"
            write_version_csv(version, data, output_file)
    
    # 生成汇总表格
    summary_file = OUTPUT_DIR / "table_frozenlake_explicit_summary.csv"
    write_summary_csv(all_data, summary_file)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
