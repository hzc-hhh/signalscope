# Codex for Open Source 申请表填写指南

> 本指南基于 SignalScope 项目，逐字段告诉你如何填写 OpenAI 的 Codex for OSS 申请表单。
> 
> 表单地址：https://openai.com/zh-Hans-CN/form/codex-for-oss/

---

## 表单字段逐一解析

### 1️⃣ 说明你的角色：你是主要维护者还是协助维护者？

| 字段 | 你需要选的 |
|------|-----------|
| 主要维护者 | ✅ **选这个** |
| 协助维护者 | ❌ |

**理由：** 你是 SignalScope 的创建者和主要维护者。GitHub 仓库的 commit 历史可以证明你是 contributor 排第一的人。

---

### 2️⃣ 为什么这个开源项目很重要？（最多 500 字符）

**你需要填的内容：** 把下面这段话复制进去（英文，因为我看到你的 PDF 表单界面是中英混合的；如果你进的是中文版，下面提供中英两版）

#### 👇 推荐填写的英文版（474 字符，在 500 字以内）

```
Biomedical sensor AI — especially contactless radar & multi-modal
physiological sensing — is a frontier for AI4Health. Unlike CV or NLP,
this field has no standardized preprocessing toolchain or benchmark.
SignalScope fills that gap: it provides a unified pipeline from raw
radar IQ signals to ML-ready features, a model zoo spanning classical DSP
to Transformers/SSL, reproducible benchmarks, and human-in-the-loop
research workflows. Built by researchers who face this gap daily,
SignalScope lowers the entry barrier for the global biosensor AI community.
```

#### 👇 如果表单是中文，可以用这个（约 480 字符）

```
生物医学传感器AI——特别是非接触雷达和多模态生理传感——是AI4Health
的前沿。与CV和NLP不同，这个领域缺乏标准化的信号预处理工具链和统一的
基准测试框架。SignalScope填补了这个空白：提供从原始雷达IQ信号到ML
特征的统一流水线、涵盖传统DSP到Transformer/SSL的模型库、可复现的
benchmark，以及人在回路中的科研工作流。由直面这一空白的研究者构建。
```

---

### 3️⃣ 你感兴趣的方向（多选）

两个选项：
- ☑ **Codex Security** 
- ☑ **项目的 API 额度**

**建议：两个都勾选。**

- **Codex Security**：可以帮你检查代码安全漏洞，SignalScope 处理的是医疗相关数据，安全性很重要
- **API 额度**：你提到有 Claude 和 DeepSeek API 但还没用——Codex 的 API 额度可以用来做代码自动补全、文档生成等功能开发

---

### 4️⃣ OpenAI 组织 ID

**留空。** 这是可选的。你没有 OpenAI 组织账户就不用填。

---

### 5️⃣ 你会如何使用 API 额度来做自己的项目？

这个问题问的是"你会用 API 做什么"。回答示例：

```
I will use the API credits to integrate Codex as an AI-assisted research
agent within SignalScope: (1) literature survey automation — Codex reads
papers and generates structured notes; (2) experiment design — Codex helps
formulate hypothesis tests from data patterns; (3) paper drafting — Codex
assists with method descriptions and result interpretation. The API will
also power SignalScope's documentation examples and tutorial generation.
```

中文版：

```
我将用API额度在SignalScope中集成Codex作为AI辅助科研代理：(1) 文献调研
自动化——Codex阅读论文生成结构化笔记；(2) 实验设计——Codex从数据模式中
帮助形成假设检验；(3) 论文撰写——Codex辅助方法描述和结果解释。API也将
用于SignalScope的文档示例和教程生成。
```

---

## ⚠️ 提交前的检查清单

| 检查项 | 状态 |
|--------|------|
| GitHub 仓库已公开（public）且 README 完整 | ☐ |
| README 使用英文（OpenAI 评审者主要看英文） | ☐ |
| 有 LICENSE（Apache 2.0） | ☐ |
| 有 CONTRIBUTING.md | ☐ |
| 有 CI/CD badge（绿色通过状态） | ☐ |
| 项目有明确的"Why this matters"段落 | ☐ |
| 仓库有一定数量的 commit（至少 10-20 个） | ☐ |
| Star 数（可以请朋友/同学 star，有一些即可） | ☐ |

---

## 🚀 下一步操作步骤

1. **推送到 GitHub**
   ```bash
   cd D:/project/signalscope
   git remote add origin https://github.com/<你的用户名>/signalscope.git
   git add -A
   git commit -m "feat: initial SignalScope release v0.1.0"
   git push -u origin main
   ```

2. **等待 CI 通过**（GitHub Actions 会自动跑测试和 lint）

3. **邀请同学 star 你的仓库**（有几个 star 比 0 好看很多）

4. **打开申请页面** → 填写表单 → 提交

5. **等待审核**（通常是几周内回复）

---

## 💡 提高通过率的技巧

- **在申请提交前 2-3 天内保持仓库活跃**（每天有 commit）
- **README 里明确写「本项目正在申请 OpenAI Codex for OSS 计划」** 这会让评审觉得你是认真的
- **发一条 Twitter/X 或 LinkedIn 宣传你的项目**，可以让项目看起来更有社区影响力
- **如果被拒，不要气馁**，继续维护项目、积累 star 和用户，3 个月后可以再次申请
