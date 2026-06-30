# 🧠 认知框架 / Cognitive Architecture

> 三维连续坐标 + 因果链导航的记忆系统设计。
> 让 AI 的**检索 = 思考**，而不是扁平的关键词匹配。
>
> 作者: Joe & Hermes, 2026-06
> 状态: 理论完整，待实现验证

---

## 是什么

一个为 AI Agent 设计的**认知管线架构**——不依赖深度学习、不依赖 embedding 模型、纯拓扑驱动。

核心命题：AI 怎么知道自己知道什么、不知道什么、以及**在知道和不知道之间怎么走**。

### 一句话

把知识组织成连续的三维空间（密度 × 领域 × 时间），用因果链导航，让检索本身就是推理过程。

---

## 核心理念

| 概念 | 含义 |
|------|------|
| **三维坐标** | 每个知识节点是一个点：密度(0~1) × 领域(0°~360°) × 时间(遗忘曲线) |
| **因果链** | 只有 4 种边：`depends-on` / `uses` / `avoids` / `related`。没有"相似"边 |
| **检索 ≡ 思考** | 找答案 = 在空间里走路径，路径选择 = 推理 |
| **三值真值** | `true` / `unknown` / `false`。`unknown` 吸收级联崩溃的冲击 |
| **自修正拓扑** | 不需要额外"检查模块"——图结构自身会表面"这条路不对劲" |
| **热权重** | 每条边有相关度（静态）× 热度（动态），热度分频次热 + 查询临时热 |
| **假设分支** | 查询含假设时，仅在分歧点分叉，不跑两趟完整搜索 |
| **NN 同构性** | 这个框架的拓扑结构 = 神经网络的拓扑结构。从完全不同起点收敛 |

详见 `README.md`（原 SKILL.md）第 3 节 "Core Architecture"。

---

## 目录结构

```
cognitive-architecture/
├── README.md                          ← 本文档
├── Cognitive-Architecture-Overview.md ← **GitHub 发布用总览文章**（v1 + 2026-06-30 深化整合）
├── SKILL.md                           ← ⚠️ 缺失（原位置已移除，v1 内容在 D:\llam\cognitive-framework-design.md）
└── references/
    ├── cognitive-framework-design.md  ← ⚠️ 空文件（v2 内容在 D:\llam\cognitive-framework-design.md）
    ├── cognitive-framework-design-session-update.md
    ├── cognitive-framework-deep-dive-20260630.md ← **最新** 因果双维度权重/压缩可追溯/认知总管/检索漏斗
    ├── framework-design-log.md        ← 设计推理全过程
    ├── embedding-routing-experiment-2026-06-07.md  ← 第一次实践验证
    ├── memory-research-foundations.md ← 神经科学/心理学/哲学基础（Bartlett/Bergson/预测加工）
    ├── neg-axis-layer-2026-06-07.md   ← 密度轴 < 0 的理论扩展
    ├── neural-network-isomorphism-2026-06-07.md    ← NN 拓扑同构发现
    ├── nn-topology-isomorphism-2026-06-07.md       ← 同构的深度映射
    └── state-vector-business-concept.md            ← 状态向量=氛围编码→产品方案
```

---

## 阅读顺序

| 目的 | 先读 |
|------|------|
|| 快速了解 | `D:\llam\cognitive-framework-design.md` 全文（v1 完整设计） |
|| 完整框架 + 最新深化 | `references/cognitive-framework-deep-dive-20260630.md` **← 2026-06-30 新增** |
|| v1 原始设计 | `D:\llam\cognitive-framework-design.md`（三轴坐标/因果链/热权重/三值真值） |
|| 为什么这么设计 | `references/framework-design-log.md` |
| **最新版本** | `references/cognitive-framework-design.md`（v2，含热权重/假设分支/三值真值） |
| 真的能用吗 | `references/embedding-routing-experiment-2026-06-07.md` |
| 科学依据 | `references/memory-research-foundations.md` |
| 神经网络揭示 | `references/neural-network-isomorphism-2026-06-07.md` |
| 变现方向 | `references/state-vector-business-concept.md` |

---

## 关键实验 & 实现

### Y 轴落地：state.db 事件线（2026-06-22）
47,210 条消息，856 个 session，5月17 ~ 今天。每条带 `timestamp`、`role`、`session_id`。微信 + 桌面端对话一字不落全在。每日 2AM 自动备份到 `D:\hermes对话数据\`。
### 蒸馏管线（设计中）

```
state.db (Y轴·事件)
    ↓ 自动提取: X领域 + Z密度 + because-of因果链
    ↓ 推送人/CC 把关
knowledge_nodes 表 (高置信度知识)
```

- ✅ 承载层就绪 — `session_backfill.py` + `learn()` + `knowledge_nodes` + `refresh_confidence.py`
- 🟡 缺口 — 自动因果提取、三轴标注、把关 pipeline

### 认知路由模型（2026-06-23 新设计）

认知系统最大的卡点是坐标标注靠手工。路由模型是解决这个的引擎——蒸馏一个小模型专门做标注。

**做什么：**
```
输入: 对话片段 / 文章 / 用户消息
  ↓
路由模型 (1-3B, 蒸馏)
  ↓
输出: {x: {ai, hardware, ...}, z: 0.01-1.0, causal_chain, routing}
```

**为什么蒸馏：** 这不是通用推理，是分类+结构化输出任务。1-3B 够用，不需要 35B MoE。

**数据管线：**
1. state.db 捞几千条对话 → DeepSeek 批量标注 X/Z/因果链
2. 标注数据 → QLoRA 蒸馏到 1-3B 模型
3. 训练用 2080Ti 22G（QLoRA 3B 完全够）
4. 推理：台式 CPU（:8084）或小米8 端侧（1B Q4, ~2GB）

**投入产出：**
-   episodic 表 44+ 条 x/z 坐标从 NULL → 实数据
-   高密度抽象(z>0.7)自动路由 DeepSeek/CC → 不浪费 token
-   低密度事实(z<0.3)小模型处理 → 省 token
-   认知系统的引擎——让检索=思考的框架第一次能自动运转

### 认知架构与嵌入层的关系（2026-06-23）

认知系统不是从头搭在一堆散点上的——它本来就有结构：

| 层 | 是什么 | 状态 |
|---|---|---|
| **图结构** | 节点间带标记的边、因果链四种关系（depends-on/uses/avoids/related） | ✅ 认知架构设计 |
| **三轴坐标** | X领域 / Y时间 / Z密度——可解释的语义维度 | ✅ 设计就绪，待自动标注 |
| **BGE 嵌入** | 节点的初始位置，384 维纠缠态——没有任何单个维度有独立含义 | ✅ :8082 跑着 |
| **路由模型** | 连接嵌入层和坐标层：384 维纠缠输入 → 三轴可解释输出 | 🟡 设计阶段 |

BGE 的 384 维不是 384 个有名字的轴。"AI"分散在所有 384 维上通过组合表达。三轴坐标是在这个纠缠空间之上**施加**的结构——解纠缠。路由模型就是做这个解纠缠的引擎。

### embedding 路由实验（2026-06-07）

| 场景 | 最佳方案 | 证据 |
|------|----------|------|
| 小图(3-5节点), 4B模型 | **纯模型路由**（prompt 逐级推理） | 11/14 vs 5/14, 10.7s avg |
| 大图(10+节点), 4B模型 | **纯模型路由**（embedding 不增益） | embed-only 5/14, hybrid 6/14 |
| 大图, 9B+模型 | embedding 可能赢（待测） | — |
| 强专用 embedding 模型 | embedding 可能赢（待测） | bge-m3 等 |

核心发现：**4B 模型的瓶颈是语义粒度，不是图大小**。它可以走 19 节点 × 2-3 跳，但分不清语义相近的类目（"是谁开发的" → identity-reply vs simple-fact）。这是模型能力边界，不是架构的问题。

---

## 产品方向

状态向量 = 氛围编码。两条路径：HTTP API（任何 agent 都能接状态感知）+ Hermes 插件（先自用再推广）。不做 Hindsight 竞品对标——做没人做的"感觉匹配"维度。

详见 `references/state-vector-business-concept.md`。

---

## 状态

- ✅ 理论自洽（跨 7 个问题的设计审查）
- ✅ 拓扑收敛验证（NN 同构性）
- ✅ 第一次实现验证完成（embedding 路由实验）
- ✅ **Y 轴（时间）已落地** — state.db 47K条消息，每条带 timestamp
- ✅ **承载层就绪** — memory-db (knowledge_nodes + episodic + memories) 6 月已上线
- ✅ **认知总管设计（2026-06-30）** — 小模型三合一（语义路由+实时监听+压缩写入）设计完成
- ✅ **因果双维度权重（2026-06-30）** — 因权重+果权重分离设计完成
- ✅ **压缩可追溯（2026-06-30）** — derived-from/compressed-to 引线设计完成
- ✅ **4K 宪法层分离（2026-06-30）** — 4K宪法层 vs memory-db图书馆层 vs 认知系统的三层架构确认
- 🟡 蒸馏管线设计中 — state.db → X领域+Z密度+因果链提取 → 人/CC把关 → knowledge_nodes
- 🟡 **认知路由模型**（2026-06-23 设计）— 蒸馏 1-3B 模型自动标注 X/Z/因果链+路由建议
- ❌ 未实现完整的因果链导航
- ❌ X 轴（领域）和 Z 轴（密度）手动填了一部分，自动标注待路由模型落地

**认识论状态：** 框架通过了自洽性检验、收敛性检验、可实现性检验。Y 轴已经实打实落库，承载层（memory-db）在生产环境中运行。2026-06-30 完成了边权重精细化（因/果分离）、压缩可追溯设计、认知总管架构设计。接下来把 X 和 Z 标上，因果链串起来。

---

## 相关项目

| 项目 | 关系 |
|------|------|
| `memory-db` (Hermes skill) | SQLite 分层记忆系统——已在生产环境承载本框架的知识层 |
| `embed`（嵌入项目） | 端侧 AI 电子宠物——氛围向量产品化方向。路由模型将在此项目训练 |
| `state.db` | 全部对话记录（47K条）——Y 轴已落地，路由模型训练数据源 |
| `D:\hermes对话数据\` | 每日自动备份——灾备副本 |
| `three-body-patterns` | 系统级设计模式（互补，不同抽象层） |
| `agent-thinking-patterns` | 操作级思考模式（本框架更深——知识导航层） |
| `harness-engineering` | AI 模型脚手架（本框架是其上的认知层） |
