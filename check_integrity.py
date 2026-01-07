#!/usr/bin/env python3
"""
实验结果完整性检查脚本
检查 /data/xingkun/experiment_result 目录下的实验结果是否完整
"""

import os
from pathlib import Path
from datetime import datetime

# 配置
BASE_DIR = Path("/data/xingkun/experiment_result")
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = BASE_DIR / f"integrity_report_{TIMESTAMP}.md"

# 模型列表及其前缀映射（按指定顺序）
# model_name: (env_folder_prefix, [possible_method_folder_model_names])
MODELS = {
    "llama3.1_8b": ("llama3.1_8b", ["llama3.1_8b", "llama3.1-8b"]),          # Llama3.1-8B
    "llama-3.3-70b-instruct": ("llama-3.3-70b-instruct", ["llama-3.3-70b-instruct"]),  # Llama3.3-70B
    "qwen2.5-7b": ("qwen2.5-7b-instruct", ["qwen2.5-7b"]),                   # Qwen2.5-7B
    "qwen3-30b": ("qwen3-30b-instruct", ["qwen3-30b"]),                      # Qwen3-30B
    "gpt4o": ("gpt4o", ["gpt4o", "gpt-4o"]),                                 # GPT-4o
    "grok-3": ("grok-3", ["grok-3"]),                                        # Grok-3
    "deepseek-r1": ("deepseek-r1", ["deepseek-r1"]),                         # DeepSeek-R1
}

# 环境列表
ENVIRONMENTS = [
    "frozenlake-explicit",
    "frozenlake-implicit",
    "mountaincar",
    "webshop-explicit",
    "webshop-implicit",
]

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


def find_env_folder(model_dir: Path, prefix: str, env: str) -> Path | None:
    """查找环境文件夹，尝试不同的命名格式"""
    # 可能的命名格式
    possible_names = [
        f"{prefix}-{env}",
        f"{prefix}_{env}",
    ]
    
    for name in possible_names:
        folder = model_dir / name
        if folder.exists():
            return folder
    
    # 如果都不存在，尝试搜索
    for item in model_dir.iterdir():
        if item.is_dir() and env in item.name:
            return item
    
    return None


def get_env_short_name(env: str) -> str:
    """获取环境的简短名称（用于日志文件夹）"""
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


def check_csv_lines(csv_path: Path) -> int:
    """检查CSV文件的行数"""
    if not csv_path.exists():
        return -1
    
    with open(csv_path, 'r') as f:
        return sum(1 for _ in f)


def is_implicit_env(env: str) -> bool:
    """判断是否是implicit环境"""
    return "implicit" in env


def check_env_folder_consistency(folder_name: str, model_name: str, model_variants: list, env: str) -> list:
    """
    检查环境文件夹名称的一致性
    返回不一致的问题列表
    """
    issues = []
    folder_lower = folder_name.lower()
    
    # 1. 检查模型一致性：文件夹名必须包含模型名的某个变体
    model_found = any(variant.lower() in folder_lower for variant in model_variants)
    if not model_found:
        issues.append(f"模型不匹配: 期望包含 {model_variants} 之一")
    
    # 2. 检查环境一致性
    env_short = get_env_short_name(env)
    if env_short.lower() not in folder_lower:
        issues.append(f"环境不匹配: 期望包含 '{env_short}'")
    
    return issues


def check_method_folder_consistency(folder_name: str, model_name: str, model_variants: list, env: str) -> list:
    """
    检查方法文件夹名称的一致性
    返回不一致的问题列表
    """
    issues = []
    folder_lower = folder_name.lower()
    
    # 1. 检查模型一致性：文件夹名必须包含模型名的某个变体
    model_found = any(variant.lower() in folder_lower for variant in model_variants)
    if not model_found:
        issues.append(f"模型不匹配: 期望包含 {model_variants} 之一")
    
    # 2. 检查环境一致性
    env_short = get_env_short_name(env)
    if env_short.lower() not in folder_lower:
        issues.append(f"环境不匹配: 期望包含 '{env_short}'")
    
    # 3. 检查 implicit/explicit 一致性 (只对方法文件夹检查 hidden)
    has_hidden = "hidden" in folder_lower
    if is_implicit_env(env):
        if not has_hidden:
            issues.append("implicit环境但文件夹名不包含'hidden'")
    else:
        if has_hidden:
            issues.append("非implicit环境但文件夹名包含'hidden'")
    
    return issues


def check_model(model_name: str, model_prefix: str, model_variants: list) -> dict:
    """检查单个模型的完整性"""
    model_dir = BASE_DIR / model_name
    result = {
        "exists": model_dir.exists(),
        "environments": {},
        "consistency_issues": [],  # 一致性问题
    }
    
    if not result["exists"]:
        return result
    
    for env in ENVIRONMENTS:
        env_folder = find_env_folder(model_dir, model_prefix, env)
        env_result = {
            "exists": env_folder is not None,
            "folder_name": env_folder.name if env_folder else None,
            "methods": {},
            "method_count": 0,
            "consistency_issues": [],  # 环境级别的一致性问题
        }
        
        if env_folder:
            # 检查环境文件夹名的一致性
            env_consistency = check_env_folder_consistency(env_folder.name, model_name, model_variants, env)
            env_result["consistency_issues"] = env_consistency
            
            for method in METHODS:
                method_folder = find_method_folder(env_folder, model_prefix, env, method)
                csv_path = method_folder / "log" / "explorer_summary.csv" if method_folder else None
                
                expected_lines = 41 if is_implicit_env(env) else 61  # 用户指定的行数
                actual_lines = check_csv_lines(csv_path) if csv_path else -1
                
                # 检查方法文件夹名的一致性
                method_consistency = []
                if method_folder:
                    method_consistency = check_method_folder_consistency(method_folder.name, model_name, model_variants, env)
                
                method_result = {
                    "exists": method_folder is not None,
                    "folder_name": method_folder.name if method_folder else None,
                    "csv_exists": csv_path.exists() if csv_path else False,
                    "csv_lines": actual_lines,
                    "expected_lines": expected_lines,
                    "csv_ok": actual_lines == expected_lines,
                    "consistency_issues": method_consistency,
                }
                
                env_result["methods"][method] = method_result
                if method_folder:
                    env_result["method_count"] += 1
        
        result["environments"][env] = env_result
    
    return result


def generate_report(results: dict) -> str:
    """生成完整性报告"""
    lines = []
    lines.append("# 实验结果完整性检查报告")
    lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**检查目录**: `{BASE_DIR}`")
    
    # 总览
    lines.append("\n## 📊 总览")
    lines.append("\n| 模型 | 环境数 | 完整性 | 一致性 |")
    lines.append("|------|--------|--------|--------|")
    
    total_issues = 0
    total_consistency_issues = 0
    
    for model_name in MODELS:
        result = results[model_name]
        if not result["exists"]:
            lines.append(f"| {model_name} | 0/5 | ❌ 文件夹不存在 | - |")
            total_issues += 1
            continue
        
        env_count = sum(1 for env in result["environments"].values() if env["exists"])
        model_issues = []
        consistency_count = 0
        
        if env_count < 5:
            model_issues.append(f"缺少{5-env_count}个环境")
        
        for env_name, env_result in result["environments"].items():
            if env_result["exists"]:
                if env_result["method_count"] < 9:
                    model_issues.append(f"{env_name}缺少方法")
                for method_name, method_result in env_result["methods"].items():
                    if method_result["exists"] and not method_result["csv_ok"]:
                        model_issues.append(f"CSV行数错误")
                        break
                # 统计一致性问题
                consistency_count += len(env_result.get("consistency_issues", []))
                for method_result in env_result["methods"].values():
                    consistency_count += len(method_result.get("consistency_issues", []))
        
        if model_issues:
            total_issues += 1
            status = "⚠️ " + ", ".join(set(model_issues))
        else:
            status = "✅ 完整"
        
        if consistency_count > 0:
            total_consistency_issues += consistency_count
            consistency_status = f"⚠️ {consistency_count}个问题"
        else:
            consistency_status = "✅ 一致"
        
        lines.append(f"| {model_name} | {env_count}/5 | {status} | {consistency_status} |")
    
    # 一致性问题报告（放在详细报告之前）
    lines.append("\n## 🔍 一致性检查")
    
    has_consistency_issues = False
    for model_name in MODELS:
        result = results[model_name]
        if not result["exists"]:
            continue
        
        model_consistency_lines = []
        for env_name in ENVIRONMENTS:
            env_result = result["environments"][env_name]
            if not env_result["exists"]:
                continue
            
            env_issues = env_result.get("consistency_issues", [])
            method_issues = []
            
            for method_name, method_result in env_result["methods"].items():
                if method_result["exists"] and method_result.get("consistency_issues"):
                    method_issues.append((method_name, method_result["folder_name"], method_result["consistency_issues"]))
            
            if env_issues or method_issues:
                model_consistency_lines.append(f"\n#### {env_name}")
                if env_issues:
                    model_consistency_lines.append(f"- 环境文件夹 `{env_result['folder_name']}`:")
                    for issue in env_issues:
                        model_consistency_lines.append(f"  - ❌ {issue}")
                if method_issues:
                    for method_name, folder_name, issues in method_issues:
                        model_consistency_lines.append(f"- 方法 `{method_name}` (`{folder_name}`):")
                        for issue in issues:
                            model_consistency_lines.append(f"  - ❌ {issue}")
        
        if model_consistency_lines:
            has_consistency_issues = True
            lines.append(f"\n### 🔹 {model_name}")
            lines.extend(model_consistency_lines)
    
    if not has_consistency_issues:
        lines.append("\n✅ **所有文件夹命名一致，无问题！**")
    
    # 详细报告
    lines.append("\n## 📋 详细报告")
    
    for model_name in MODELS:
        lines.append(f"\n### 🔹 {model_name}")
        result = results[model_name]
        
        if not result["exists"]:
            lines.append("\n❌ **模型文件夹不存在**")
            continue
        
        for env_name in ENVIRONMENTS:
            env_result = result["environments"][env_name]
            
            if not env_result["exists"]:
                lines.append(f"\n#### ❌ {env_name}")
                lines.append("- 环境文件夹不存在")
                continue
            
            # 检查该环境是否有问题
            env_issues = []
            if env_result["method_count"] < 9:
                env_issues.append(f"方法数: {env_result['method_count']}/9")
            
            csv_issues = []
            for method_name, method_result in env_result["methods"].items():
                if method_result["exists"]:
                    if not method_result["csv_exists"]:
                        csv_issues.append(f"{method_name}: CSV不存在")
                    elif not method_result["csv_ok"]:
                        csv_issues.append(f"{method_name}: {method_result['csv_lines']}/{method_result['expected_lines']}行")
            
            if env_issues or csv_issues:
                lines.append(f"\n#### ⚠️ {env_name}")
                lines.append(f"- 文件夹: `{env_result['folder_name']}`")
                if env_issues:
                    for issue in env_issues:
                        lines.append(f"- {issue}")
                if csv_issues:
                    lines.append(f"- CSV行数问题:")
                    for issue in csv_issues:
                        lines.append(f"  - {issue}")
                
                # 列出缺失的方法
                missing_methods = [m for m, r in env_result["methods"].items() if not r["exists"]]
                if missing_methods:
                    lines.append(f"- 缺失方法: {', '.join(missing_methods)}")
            else:
                lines.append(f"\n#### ✅ {env_name}")
                lines.append(f"- 文件夹: `{env_result['folder_name']}`")
                lines.append(f"- 方法数: {env_result['method_count']}/9 ✓")
                lines.append(f"- CSV行数: 全部正确 ✓")
    
    # 统计摘要
    lines.append("\n## 📈 统计摘要")
    
    total_envs = 0
    total_methods = 0
    total_csvs_ok = 0
    total_csvs = 0
    
    for model_name in MODELS:
        result = results[model_name]
        if result["exists"]:
            for env_result in result["environments"].values():
                if env_result["exists"]:
                    total_envs += 1
                    for method_result in env_result["methods"].values():
                        if method_result["exists"]:
                            total_methods += 1
                            total_csvs += 1
                            if method_result["csv_ok"]:
                                total_csvs_ok += 1
    
    lines.append(f"\n- **模型数**: {sum(1 for r in results.values() if r['exists'])}/7")
    lines.append(f"- **环境数**: {total_envs}/{7*5}")
    lines.append(f"- **方法数**: {total_methods}/{7*5*9}")
    lines.append(f"- **CSV正确率**: {total_csvs_ok}/{total_csvs} ({100*total_csvs_ok/total_csvs:.1f}%)" if total_csvs > 0 else "- **CSV正确率**: N/A")
    lines.append(f"- **一致性问题**: {total_consistency_issues}个")
    
    return "\n".join(lines)


def main():
    print("🔍 开始检查实验结果完整性...")
    
    results = {}
    for model_name, (model_prefix, model_variants) in MODELS.items():
        print(f"  检查 {model_name}...")
        results[model_name] = check_model(model_name, model_prefix, model_variants)
    
    report = generate_report(results)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 报告已生成: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

