#!/usr/bin/env python3
"""
检查 frozenlake explicit 实验中 glove 与非 glove 的性能对比。
找出 env1 和 env2 场景下 glove 比非 glove 差的情况。
"""

import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 配置
BASE_DIR = Path("/data/xingkun/experiment_result")
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_FILE = BASE_DIR / f"glove_performance_report_{TIMESTAMP}.md"

# 版本列表
VERSIONS = ["v0", "v1", "v2", "v3", "v4"]

# 方法对：非glove -> glove
METHOD_PAIRS = [
    ("vanilla", "vanilla-glove"),
    ("memorybank", "memorybank-glove"),
    ("voyager", "voyager-glove"),
    ("generative", "generative-glove"),
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


def parse_csv(csv_path: Path) -> dict:
    """
    解析 CSV 文件，返回数据字典
    返回: {model: {method: {env0: val, env1: val, env2: val}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    if not csv_path.exists():
        return data
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < 3:
        return data
    
    # 第一行是模型名（每3列一个模型）
    # 第二行是 env0, env1, env2 重复
    # 第三行开始是数据
    
    header1 = rows[0]
    header2 = rows[1]
    
    # 解析模型位置
    models = []
    col = 1
    while col < len(header1):
        model_name = header1[col].strip()
        if model_name:
            models.append((model_name, col))
        col += 3
    
    # 解析数据行
    for row in rows[2:]:
        if not row or not row[0]:
            continue
        method = row[0].strip()
        
        for model_name, start_col in models:
            try:
                env0_val = row[start_col].strip() if start_col < len(row) else ""
                env1_val = row[start_col + 1].strip() if start_col + 1 < len(row) else ""
                env2_val = row[start_col + 2].strip() if start_col + 2 < len(row) else ""
                
                data[model_name][method] = {
                    "env0": float(env0_val) if env0_val else None,
                    "env1": float(env1_val) if env1_val else None,
                    "env2": float(env2_val) if env2_val else None,
                }
            except (ValueError, IndexError):
                pass
    
    return data


def compare_glove_performance(data: dict) -> list:
    """
    比较 glove 与非 glove 的性能
    返回 glove 比非 glove 差的情况列表
    """
    issues = []
    
    for model in MODEL_ORDER:
        if model not in data:
            continue
        
        model_data = data[model]
        
        for base_method, glove_method in METHOD_PAIRS:
            if base_method not in model_data or glove_method not in model_data:
                continue
            
            base = model_data[base_method]
            glove = model_data[glove_method]
            
            # 检查 env1
            if base["env1"] is not None and glove["env1"] is not None:
                if glove["env1"] < base["env1"]:
                    diff = base["env1"] - glove["env1"]
                    issues.append({
                        "model": model,
                        "method": base_method,
                        "env": "env1",
                        "base_score": base["env1"],
                        "glove_score": glove["env1"],
                        "diff": diff,
                    })
            
            # 检查 env2
            if base["env2"] is not None and glove["env2"] is not None:
                if glove["env2"] < base["env2"]:
                    diff = base["env2"] - glove["env2"]
                    issues.append({
                        "model": model,
                        "method": base_method,
                        "env": "env2",
                        "base_score": base["env2"],
                        "glove_score": glove["env2"],
                        "diff": diff,
                    })
    
    return issues


def analyze_all_versions() -> dict:
    """分析所有版本"""
    results = {}
    
    for version in VERSIONS:
        csv_path = BASE_DIR / f"table_frozenlake_explicit_{version}.csv"
        data = parse_csv(csv_path)
        issues = compare_glove_performance(data)
        results[version] = {
            "data": data,
            "issues": issues,
        }
    
    return results


def generate_report(results: dict) -> str:
    """生成报告"""
    lines = []
    lines.append("# FrozenLake Explicit: Glove 性能对比报告")
    lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n**分析内容**: 在 env1 和 env2 场景下，glove 比非 glove 表现**更差**的情况")
    
    # 总览
    lines.append("\n## 📊 总览")
    lines.append("\n| 版本 | 问题数量 | 涉及模型 | 涉及方法 |")
    lines.append("|------|----------|----------|----------|")
    
    total_issues = 0
    all_issues = []
    
    for version in VERSIONS:
        issues = results[version]["issues"]
        total_issues += len(issues)
        all_issues.extend([(version, issue) for issue in issues])
        
        if issues:
            models = set(i["model"] for i in issues)
            methods = set(i["method"] for i in issues)
            lines.append(f"| {version} | {len(issues)} | {', '.join(sorted(models))} | {', '.join(sorted(methods))} |")
        else:
            lines.append(f"| {version} | 0 | - | - |")
    
    # 按严重程度排序的问题汇总
    lines.append("\n## 🔥 问题汇总（按差异排序）")
    
    if all_issues:
        # 按差异从大到小排序
        all_issues_sorted = sorted(all_issues, key=lambda x: x[1]["diff"], reverse=True)
        
        lines.append("\n| 版本 | 模型 | 方法 | 环境 | 非Glove | Glove | 差异 |")
        lines.append("|------|------|------|------|---------|-------|------|")
        
        for version, issue in all_issues_sorted:
            lines.append(
                f"| {version} | {issue['model']} | {issue['method']} | {issue['env']} | "
                f"{issue['base_score']:.4f} | {issue['glove_score']:.4f} | "
                f"**-{issue['diff']:.4f}** |"
            )
    else:
        lines.append("\n✅ **没有发现 glove 比非 glove 差的情况！**")
    
    # 按版本详细报告
    lines.append("\n## 📋 按版本详细报告")
    
    for version in VERSIONS:
        lines.append(f"\n### 🔹 {version}")
        issues = results[version]["issues"]
        
        if not issues:
            lines.append("\n✅ 该版本没有问题")
            continue
        
        # 按模型分组
        by_model = defaultdict(list)
        for issue in issues:
            by_model[issue["model"]].append(issue)
        
        for model in MODEL_ORDER:
            if model not in by_model:
                continue
            
            model_issues = by_model[model]
            lines.append(f"\n#### {model}")
            
            for issue in model_issues:
                lines.append(
                    f"- **{issue['method']}** @ {issue['env']}: "
                    f"非Glove={issue['base_score']:.4f}, Glove={issue['glove_score']:.4f} "
                    f"(差异: **-{issue['diff']:.4f}**)"
                )
    
    # 按方法统计
    lines.append("\n## 📈 按方法统计")
    
    method_stats = defaultdict(lambda: {"env1": 0, "env2": 0, "total_diff": 0.0})
    for version, issue in all_issues:
        method = issue["method"]
        env = issue["env"]
        method_stats[method][env] += 1
        method_stats[method]["total_diff"] += issue["diff"]
    
    if method_stats:
        lines.append("\n| 方法 | env1问题数 | env2问题数 | 总差异累计 |")
        lines.append("|------|------------|------------|------------|")
        
        for base_method, _ in METHOD_PAIRS:
            stats = method_stats.get(base_method, {"env1": 0, "env2": 0, "total_diff": 0.0})
            lines.append(
                f"| {base_method} | {stats['env1']} | {stats['env2']} | {stats['total_diff']:.4f} |"
            )
    
    # 按模型统计
    lines.append("\n## 📈 按模型统计")
    
    model_stats = defaultdict(lambda: {"count": 0, "total_diff": 0.0})
    for version, issue in all_issues:
        model = issue["model"]
        model_stats[model]["count"] += 1
        model_stats[model]["total_diff"] += issue["diff"]
    
    if model_stats:
        lines.append("\n| 模型 | 问题数 | 总差异累计 |")
        lines.append("|------|--------|------------|")
        
        for model in MODEL_ORDER:
            stats = model_stats.get(model, {"count": 0, "total_diff": 0.0})
            if stats["count"] > 0:
                lines.append(f"| {model} | {stats['count']} | {stats['total_diff']:.4f} |")
    
    # 按环境统计
    lines.append("\n## 📈 按环境统计")
    
    env_stats = {"env1": 0, "env2": 0}
    for version, issue in all_issues:
        env_stats[issue["env"]] += 1
    
    lines.append(f"\n- **env1 问题数**: {env_stats['env1']}")
    lines.append(f"- **env2 问题数**: {env_stats['env2']}")
    lines.append(f"- **总问题数**: {total_issues}")
    
    # 结论
    lines.append("\n## 📝 结论")
    
    if total_issues == 0:
        lines.append("\n✅ 在所有版本的 env1 和 env2 场景下，glove 方法的表现均**不差于**非 glove 方法。")
    else:
        lines.append(f"\n⚠️ 共发现 **{total_issues}** 处 glove 表现不如非 glove 的情况。")
        
        # 找出最严重的问题
        if all_issues:
            worst = max(all_issues, key=lambda x: x[1]["diff"])
            lines.append(
                f"\n最严重的问题出现在 **{worst[0]}** 版本，"
                f"**{worst[1]['model']}** 模型的 **{worst[1]['method']}** 方法，"
                f"在 **{worst[1]['env']}** 场景下，"
                f"glove 比非 glove 低 **{worst[1]['diff']:.4f}**。"
            )
    
    return "\n".join(lines)


def main():
    print("🔍 开始分析 glove 性能对比...")
    
    results = analyze_all_versions()
    
    report = generate_report(results)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 报告已生成: {OUTPUT_FILE}")
    
    # 同时输出到控制台
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
