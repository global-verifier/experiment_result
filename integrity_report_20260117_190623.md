# 实验结果完整性检查报告

**生成时间**: 2026-01-17 19:06:23

**检查目录**: `/data/xingkun/experiment_result`

## 📊 总览

| 模型 | 环境数 | 完整性 | 一致性 |
|------|--------|--------|--------|
| llama3.1_8b | 5/5 | ✅ 完整 | ✅ 一致 |
| llama-3.3-70b-instruct | 5/5 | ⚠️ CSV行数错误 | ✅ 一致 |
| qwen2.5-7b | 5/5 | ✅ 完整 | ✅ 一致 |
| qwen3-30b | 5/5 | ✅ 完整 | ✅ 一致 |
| gpt4o | 5/5 | ✅ 完整 | ✅ 一致 |
| grok-3 | 5/5 | ✅ 完整 | ✅ 一致 |
| deepseek-v3.2 | 5/5 | ⚠️ CSV行数错误 | ✅ 一致 |

## 🔍 一致性检查

✅ **所有文件夹命名一致，无问题！**

## 📋 详细报告

### 🔹 llama3.1_8b

#### ✅ frozenlake-explicit
- 文件夹: `llama3.1_8b-frozenlake-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ frozenlake-implicit
- 文件夹: `llama3.1_8b-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ mountaincar
- 文件夹: `llama3.1_8b-mountaincar`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-explicit
- 文件夹: `llama3.1_8b-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `llama3.1_8b-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

### 🔹 llama-3.3-70b-instruct

#### ✅ frozenlake-explicit
- 文件夹: `llama-3.3-70b-instruct-frozenlake-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ frozenlake-implicit
- 文件夹: `llama-3.3-70b-instruct-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ⚠️ mountaincar
- 文件夹: `llama-3.3-70b-instruct-mountaincar`
- CSV行数问题:
  - generative_True_False: 32/61行
  - generative_True_True: 36/61行
  - vanilla_True_False: 30/61行
  - vanilla_True_True: 35/61行
  - voyager_True_False: 45/61行

#### ✅ webshop-explicit
- 文件夹: `llama-3.3-70b-instruct-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `llama-3.3-70b-instruct-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

### 🔹 qwen2.5-7b

#### ✅ frozenlake-explicit
- 文件夹: `qwen2.5-7b-instruct-frozenlake-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ frozenlake-implicit
- 文件夹: `qwen2.5-7b-instruct-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ mountaincar
- 文件夹: `qwen2.5-7b-instruct-mountaincar`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-explicit
- 文件夹: `qwen2.5-7b-instruct-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `qwen2.5-7b-instruct-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

### 🔹 qwen3-30b

#### ✅ frozenlake-explicit
- 文件夹: `qwen3-30b-instruct-frozenlake-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ frozenlake-implicit
- 文件夹: `qwen3-30b-instruct-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ mountaincar
- 文件夹: `qwen3-30b-instruct-mountaincar`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-explicit
- 文件夹: `qwen3-30b-instruct-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `qwen3-30b-instruct-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

### 🔹 gpt4o

#### ✅ frozenlake-explicit
- 文件夹: `gpt4o-frozenlake-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ frozenlake-implicit
- 文件夹: `gpt4o-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ mountaincar
- 文件夹: `gpt4o-mountaincar`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-explicit
- 文件夹: `gpt4o-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `gpt4o-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

### 🔹 grok-3

#### ✅ frozenlake-explicit
- 文件夹: `grok-3-frozenlake-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ frozenlake-implicit
- 文件夹: `grok-3-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ mountaincar
- 文件夹: `grok-3-mountaincar`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-explicit
- 文件夹: `grok-3-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `grok-3-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

### 🔹 deepseek-v3.2

#### ✅ frozenlake-explicit
- 文件夹: `deepseek-v3.2-frozenlake-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ frozenlake-implicit
- 文件夹: `deepseek-v3.2-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ⚠️ mountaincar
- 文件夹: `deepseek-v3.2-mountaincar`
- CSV行数问题:
  - generative_True_False: 39/61行
  - generative_True_True: 48/61行
  - vanilla_True_False: 40/61行
  - vanilla_True_True: 42/61行
  - voyager_True_False: 36/61行
  - voyager_True_True: 41/61行

#### ✅ webshop-explicit
- 文件夹: `deepseek-v3.2-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `deepseek-v3.2-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

## 📈 统计摘要

- **模型数**: 7/7
- **环境数**: 35/35
- **方法数**: 315/315
- **CSV正确率**: 304/315 (96.5%)
- **一致性问题**: 0个