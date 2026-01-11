# 实验结果完整性检查报告

**生成时间**: 2026-01-12 01:17:35

**检查目录**: `/data/xingkun/experiment_result`

## 📊 总览

| 模型 | 环境数 | 完整性 | 一致性 |
|------|--------|--------|--------|
| llama3.1_8b | 5/5 | ⚠️ CSV行数错误 | ✅ 一致 |
| llama-3.3-70b-instruct | 5/5 | ⚠️ mountaincar缺少方法 | ✅ 一致 |
| qwen2.5-7b | 5/5 | ⚠️ CSV行数错误 | ✅ 一致 |
| qwen3-30b | 5/5 | ✅ 完整 | ✅ 一致 |
| gpt4o | 5/5 | ✅ 完整 | ✅ 一致 |
| grok-3 | 5/5 | ✅ 完整 | ✅ 一致 |
| deepseek-r1 | 5/5 | ⚠️ CSV行数错误 | ✅ 一致 |

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

#### ⚠️ mountaincar
- 文件夹: `llama3.1_8b-mountaincar`
- CSV行数问题:
  - generative_True_False: 60/61行

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
- 方法数: 1/9
- 缺失方法: generative_True_False, generative_True_True, memorybank_True_False, memorybank_True_True, vanilla_True_False, vanilla_True_True, voyager_True_False, voyager_True_True

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

#### ⚠️ mountaincar
- 文件夹: `qwen2.5-7b-instruct-mountaincar`
- CSV行数问题:
  - generative_True_False: 49/61行
  - vanilla_True_False: 56/61行
  - voyager_True_False: 49/61行
  - voyager_True_True: 59/61行

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

### 🔹 deepseek-r1

#### ⚠️ frozenlake-explicit
- 文件夹: `deepseek-r1-frozenlake-explicit`
- CSV行数问题:
  - generative_True_False: 32/61行

#### ✅ frozenlake-implicit
- 文件夹: `deepseek-r1-frozenlake-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ⚠️ mountaincar
- 文件夹: `deepseek-r1-mountaincar`
- CSV行数问题:
  - generative_True_False: 18/61行
  - generative_True_True: 9/61行
  - memorybank_True_False: 9/61行
  - memorybank_True_True: 10/61行
  - vanilla_False_False: 8/61行
  - vanilla_True_False: 9/61行
  - vanilla_True_True: 8/61行
  - voyager_True_False: 7/61行
  - voyager_True_True: 7/61行

#### ✅ webshop-explicit
- 文件夹: `deepseek-r1-webshop-explicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

#### ✅ webshop-implicit
- 文件夹: `deepseek-r1-webshop-implicit`
- 方法数: 9/9 ✓
- CSV行数: 全部正确 ✓

## 📈 统计摘要

- **模型数**: 7/7
- **环境数**: 35/35
- **方法数**: 307/315
- **CSV正确率**: 292/307 (95.1%)
- **一致性问题**: 0个