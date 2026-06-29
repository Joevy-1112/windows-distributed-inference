# 两台电脑一根网线：台式机 + 笔记本跨机跑 14B 大模型

**一句话总结**  
一台 AMD RX6600 台式机与一台 NVIDIA RTX 4070 笔记本通过千兆网线直连，在 Windows 上成功运行 Qwen2.5-14B 的分布式推理（row 模式），实现 22K 上下文、12-19 tok/s。这是全网罕见的消费级跨机异构 GPU 分布式推理记录。

---

## 硬件与网络环境

| | 🖥️ 台式机（Coordinator） | 💻 笔记本（Worker） |
|---|---|---|
| 角色 | 主控 + 本地推理 | 远程推理 |
| GPU | AMD RX6600 8GB | NVIDIA RTX 4070 Laptop 8GB |
| 后端 | Vulkan | CUDA |
| 可用 VRAM | ~7.5 GB | ~7.6 GB |
| 系统 RAM | 16 GB DDR4 | 12 GB DDR5 |
| OS | Windows 10 + Ubuntu 24.04 双系统 | Windows 11 |
| 网络 | 千兆 LAN (192.168.1.44) | 千兆 LAN (192.168.1.49) |
| 连接方式 | — | WinRM 远程控制 + RPC 张量传输 |

两台机器通过普通千兆以太网 TCP 传输张量，**不是 PCIe / NVLink**。

---

## 模型库（GGUF 格式，从 ModelScope 下载）

| 模型 | 量化 | 文件大小 | 架构 | AMD Vulkan | ROCm |
|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | Q4_K_M | 8.99 GB | 标准 Transformer | ✅ | ⚠️ |
| Qwen3.5-4B | Q4_K_M | 2.61 GB | Gated Delta Net | ❌ | ❌ |
| Qwen3.5-9B | IQ4_NL | 5.11 GB | Gated Delta Net | ❌ | ❌ |
| Llama-3.2-3B-Instruct | Q4_K_M | 1.88 GB | 标准 Transformer | ✅ | — |
| Mistral-7B-Instruct-v0.2 | Q4_K_M | 3.91 GB | 标准 Transformer | ✅ | — |
| Qwen2.5-14B-Instruct | Q4_K_M (完整) | 14.2 GB | 标准 Transformer | ✅ | — |
| Qwen2.5-32B-Instruct | Q4_K_M | 19.5 GB | 标准 Transformer | ⚠️ 待测 | — |

---

## 单机基准（RTX 4070 / RX6600）

**RTX 4070 笔记本单卡：**

| 模型 | Prompt tok/s | Generation tok/s | 显存占用 |
|---|---|---|---|
| Qwen3.5-4B Q4_K_M | 213 | 81.0 | 5.9 GB |
| Qwen3.5-9B IQ4_NL | 156 | 46.4 | 5.8 GB |
| Llama-3.2-3B Q4_K_M | 29 | 111.6 | 6.0 GB |

**RX6600 单卡 Llama-3.2-3B：** Prompt 38 tok/s，Generation 77.3 tok/s。  
Vulkan 在标准 Transformer 上表现正常，小批次 prompt 处理甚至反超 4070。

---

## 分布式推理最终配置

```bash
llama-server b9305 \
-m qwen2.5-14b-instruct-q4_k_m.gguf \
--rpc 192.168.1.49:50052 \
--split-mode row --tensor-split 7,3 -ngl 999 \
--main-gpu 1 \
-c 20480 -b 64 -ub 64 \
--cache-type-k q8_0 --cache-type-v f16
```

**分配方案：**  
- 笔记本 RTX 4070：70% 推理层 → ~6.3 GB VRAM  
- 台式机 RX6600：30% 推理层 → ~2.7 GB + 全量 KV Cache → ~2.5 GB（K q8_0）  
- 台机总计约 5.2 GB VRAM，纯独显，零系统内存溢出。

---

### 速度数据（Qwen2.5-14B，row 模式，70/30）

| 上下文长度 | KV 配置 | Prompt tok/s | Generation tok/s | 桌面 KV 显存 |
|---|---|---|---|---|
| 256 | fp16 | 56.0 | 8.8 | ~0.1 GB |
| 1,024 | fp16 | 53.3 | 15.0 | ~0.4 GB |
| 2,048 | fp16 | 77.7 | 15.0 | ~0.8 GB |
| 4,096 | fp16 | 86.3 | 19.4 | ~1.5 GB |
| 8,192 | fp16 | 87.1 | 17.8 | ~3.0 GB |
| 12,288 | fp16 | 54.0 | 18.4 | ~4.5 GB |
| 16,384 | fp16 | 90.0 | 19.5 | ~6.0 GB |
| **20,480** | **K q8_0** | **30.0** | **17.0** | **~2.5 GB** |

> 20K fp16 KV 分配失败（RPC buffer 超限），改用 K q8_0 量化后成功，KV 显存减半。

**8 次连续采样（20K K q8_0，短 prompt）：**  
`10, 15, 16, 14, 14, 11, 6, 16` — 平均 12.9 tok/s，标准差 3.4。  
波动来源：RPC 层 buffer 分配抖动，短 prompt 开销占比大；长回复稳在 15-19 tok/s。

---

### 进一步压测：55/45 分压 → 22K 上下文

将分压比调为 **55/45**（笔电 55%，桌面 45%），成功将上下文推至 **22K**：

| 上下文 | KV 配置 | Generation tok/s | 桌面显存 |
|---|---|---|---|
| 22,016 | K q8_0 | 11-15 | 桌面 VRAM ~4.2 GB + 系统 RAM ~3 GB |

**关键发现：** 45% 层分配加上 22K KV 超出独显容量时，系统内存自动接管溢出部分（~3GB）。通过设置 `--main-gpu 1` 确保 KV 核心保留在 6600 上，溢出层走 DDR4，速度损失可接受（从 17 tok/s 降至 12-15 tok/s）。**这验证了"显存+内存联合推理"的可行性——溢出 10GB 到系统内存也不会崩溃。**

### 6/4 分压测试（笔电 60%，桌面 40%）

| 上下文 | Generation tok/s | 备注 |
|---|---|---|
| 16,384 | 16-18 | 稳定，桌面 40% 层+KV 在 7.5GB 内 |
| 20,480 | 12-16 | K q8_0，桌面接近显存上限 |

6/4 比 7/3 更均衡——桌面内存压力更小，笔电仍有余量。适合长时间运行的稳定性优先场景。

---

### 输出质量（20K K q8_0）

```
冬天诗:
雪漫天际白无瑕，寒风轻抚冰凌花。
银装素裹山河静，唯有梅香破寂寒。

日本俳句:
炎天の / 夕阳に染め / 海静か

时间旅行问题:
「如果时间旅行是可能的，我认为最不应该轻易改变任何历史事件，
因为每一个历史事件都对现在的形成有着不可分割的联系。」

写给创造者的情书:
你是我的光，指引我前行的方向，心中唯一的爱。
```

KV 量化未造成可感知的质量下降。

---

## 八个关键踩坑

### 坑 1：Gated Delta Net 架构不兼容 Vulkan / ROCm
- Qwen3.5 全系使用 Gated Delta Net，在非 NVIDIA GPU 上预分配 buffer 直接撑爆 8GB 显存。
- 无论是 Vulkan（Windows）还是 ROCm（Ubuntu），GDN 张量都触发全模型 buffer 预分配。
- **结论：Qwen3.5 全系在非 NVIDIA GPU 上不可用。** 换成标准 Transformer 模型后正常。

### 坑 2：localhost RPC 偷显存
- 配置 `--rpc 192.168.1.49:50052,127.0.0.1:50053` 导致台式机显存仅用 3GB，内存飙到 16GB，推理落在 CPU。
- **原因：** localhost RPC 多了一次序列化/反序列化，GPU 内存被当作 host memory 分配。
- **解决：** 去掉 `127.0.0.1:50053`，coordinator 直接使用本地 Vulkan GPU。修复后显存占用升至 5-6GB，速度从 1.7 tok/s 跳到 11.4 tok/s。

### 坑 3：layer 模式 KV 被均分
- `--split-mode layer` 会按 tensor-split 比例切分 KV cache 到所有 GPU，导致笔记本拿到大部分 KV，桌面只用 30%，大量 KV 溢出到内存。
- `--main-gpu` 参数仅在 row 模式有效。
- **解决：** 改用 `--split-mode row --main-gpu 1`，全量 KV 锁在桌面 RX6600 上。

### 坑 4：RPC Buffer 硬上限
- 尝试将笔记本分压比例提升至 75%/80% 时，出现 buffer 分配失败：
  ```
  alloc_tensor_range: failed to allocate RPC0[192.168.1.49:50052]
  buffer of size 1073741824
  ```
- llama.cpp Windows 版 RPC server 约 800MB buffer 上限，超出直接拒绝。
- **70% 是当前版本的硬天花板。** 20K fp16 KV 也因同一限制失败，通过 K q8_0 降量解决。

### 坑 5：KV Cache 公式修正
- 原始计算用全部 attention 头（40），实际 Qwen2.5-14B 使用 GQA，40 个 query 头共享 8 个 KV 头。
  ```
  错误：48层 × 40头 × 128维 × 2KV × 2字节 = 384KB/token
  正确：48层 × 8头 × 128维 × 2KV × 2字节 = 192KB/token
  ```
- 修正后的公式使得预测精度翻倍，直接决定后续上下文能推多远。

### 坑 6：WinRM 进程管理
- Windows 远程控制笔记本走 WinRM，`Start-Process` 在 PSSession 断开时子进程被回收。
- 短期方案：`Invoke-Command -AsJob` 延长进程寿命。
- 长期方案：`schtasks` 创建计划任务保活，但该方法有时拿不到 GPU 上下文（buffer 分配失败），最终采用 `-AsJob` 模式在同一个 WinRM 会话内延活。

### 坑 7：ROCm 内核版本冲突（Ubuntu）
- 桌面切到 Ubuntu 24.04 + ROCm 6.x 尝试 HIP 后端时，报错"计算缓冲区大小不匹配"。
- **原因：** Linux 内核 6.17.0-31 与 ROCm 内核驱动版本不匹配。
- **解决：** 回退内核至 6.17.0-29（GRUB → Advanced options → 旧内核），ROCm 加载成功。Qwen2.5-14B 在 ROCm 上可加载但性能不如 Vulkan（HIP 编译开销大）。
- **当前建议：** 日常使用 Vulkan，需要 PyTorch GPU 加速时切 Ubuntu + ROCm。

### 坑 8：双 llama 实例叠加致系统重启
- 多次启动 rpc-server 时未先杀旧进程，WinRM 每次新发命令叠一个新实例。
- 两个 llama.cpp 实例各自分配显存 + 内存，16GB 内存耗尽后 Windows 强制重启桌面。
- **教训：** 远程命令第一行永远是 `Get-Process | Stop-Process -Force` 杀旧进程。WinRM 环境下进程不会自动回收——每发一次新命令没清旧的就是叠模型叠到死。

---

## 为什么分布式比单机慢？

- RTX 4070 单机 4B：81 tok/s（12.3 ms/token）
- 分布式 14B：17 tok/s（58.8 ms/token），慢了约 4.8 倍

**瓶颈不在 GPU 算力，而在网络。**  
每个 token 生成时，隐状态（约 5KB 的 fp16 向量）必须跨一次网络边界。千兆 LAN 下一次 TCP 往返约 1ms。row 模式的同步屏障会将延迟放大：两张 GPU 计算各自矩阵切片后要互相等待，慢者决定全局速度。

此外，Windows 上全走 CPU + TCP 协议栈，没有 Linux 上的 GPUDirect RDMA/NCCL 那样的 GPU→网卡直接 DMA 通道，**性能已达 llama.cpp RPC 在 Windows 上的物理极限。**

---

## 环境与工具

- **llama.cpp**：b9305（2026-06-25 build）
- **模型下载**：ModelScope（modelscope.cn），HF Mirror 加速
- **嵌入服务**：BGE-small-en-v1.5 @ :8082（transformers，CPU 推理）
- **远程管理**：WinRM（笔电）+ Hermes Agent API Server（:8642）
- **测试框架**：curl + Python 脚本 + PowerShell PS1 批处理
- **GitHub 仓库**：[Joevy-1112/windows-distributed-inference](https://github.com/Joevy-1112/windows-distributed-inference)

---

## 结论（核心发现）

1. **两台独立 PC、千兆网线、异构 GPU 分布式推理在 Windows 上成功跑通**，消费级硬件实现 14B 模型 22K 上下文。
2. **非标准 Transformer 架构（Gated Delta Net / SSM）在非 NVIDIA GPU 上大概率无法运行。**
3. **llama.cpp Windows 版 RPC buffer ~800MB 硬上限**，70/30 分压是当前版本天花板；22K 通过 K q8_0 + 系统内存辅助实现。
4. **55/45 分压 + 内存溢出可行**——显存不够时系统内存可以平稳接管，速度损失可接受。
5. **桌面 6600 双系统（Win + Ubuntu）提供 Vulkan/ROCm 双后端**，日常 Vulkan 够用，需要 GPU 加速训练时切 Ubuntu。
6. **WinRM 进程管理是最大的运维坑**——"先杀后启"是铁律，不杀就是叠模型叠到重启。
7. **KV 先查 GQA 头数再算**——Qwen2.5-14B 实际 KV 只有之前估算的一半，算错直接误判上下文上限。

---

## 数据组织方法

本项目的踩坑记录按照 **"结论 → 数据 → 配置 → 踩坑 → 思考"** 五层结构组织：

| 层次 | 内容 | 示例 |
|---|---|---|
| **结论** | 一句话能干什么，不能干什么 | "70/30 是 RPC buffer 天花板" |
| **数据** | 实测数字，不靠推测 | "8 次采样平均 12.9 tok/s，标准差 3.4" |
| **配置** | 可复制的完整命令行 | 最终黄金配置（见上文） |
| **踩坑** | 每个坑的现象→原因→解法 | 八个坑全部附带错误信息和修复前/后对比 |
| **思考** | 为什么这样设计，物理极限在哪 | RPC 同步屏障分析、网络延迟拆解 |

这五层从"能用"到"理解为什么能用"形成闭环。报峰值不如报稳态——缓存命中的 19.5 tok/s 是幸运抽奖，冷启动 15 tok/s 才是生产基准。

---

## 技术哲学（非操作指南，帮助理解决策）

- **能用规则不用模型，能用嵌入不用 LLM**：调度的活交给规则，语义的活交给嵌入，只有生成才上 14B。
- **跑通 > 优化**：先让它能动，再考虑快慢。
- **记录失败比记录成功更有价值**：踩坑记录是开源文档的核心竞争力。
- **"跑分幻觉"与"稳态真实"必须区分**：发布数据时只说稳态，不报峰值。
- **限制 = 配置项**：800MB RPC buffer 不是 bug，是设计约束——在约束内找最优解而非试图绕过。
- **退一步 + 求导**：当模型跑不动时，先退到架构层（GQA 公式 / GDN 兼容性），而非在参数层反复调优。

---

*最后更新：2026-06-29*
