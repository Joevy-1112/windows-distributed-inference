# 两台电脑一根网线：台式机 + 笔记本跨机跑 14B 大模型

**一句话总结**  
一台 AMD RX6600 台式机与一台 NVIDIA RTX 4070 笔记本通过千兆网线直连，在 Windows 上成功运行 Qwen2.5-14B 的分布式推理（row 模式），实现 22K 上下文、12-19 tok/s。这是全网罕见的消费级跨机异构 GPU 分布式推理记录。

---

## 硬件与网络环境

| | 🖥️ 台式机（Coordinator） | 💻 笔记本（Worker） |
|---|---|---|
| 角色 | 主控 + 本地推理 | 远程推理 |
| GPU | AMD RX6600 8GB | NVIDIA RTX 4070 Laptop 8GB |
| 后端 | Vulkan (Windows) / ROCm (Ubuntu) | CUDA |
| 后端差异 | Vulkan 共享 GPU 内存可溢 RAM | — |
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
| Qwen3.5-4B IQ4_NL | 136.5 | 75.1 | — |
| Qwen3.5-9B IQ4_NL | 156 | 46.4 | 5.8 GB |
| Llama-3.2-3B Q4_K_M | 29 | 111.6 | 6.0 GB |

**RX6600 单卡：**

| 模型 | Prompt tok/s | Generation tok/s |
|---|---|---|
| Qwen3.5-4B IQ4_NL | 116.0 | 51.8 |
| Llama-3.2-3B Q4_K_M | 38 | 77.3 |

Vulkan 在标准 Transformer 上表现正常，4B 小批次 prompt 处理反超 4070。Qwen3.5-4B IQ4_NL 上 4070 领先 45%（gen）和 18%（prompt）。

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

### 速度数据（Qwen2.5-14B，row 模式）

**Windows Vulkan + CUDA（历史基准，b9305 预编版）：**

| 上下文长度 | KV 配置 | Prompt tok/s | Generation tok/s |
|---|---|---|---|
| 4,096 | fp16 | 86.3 | 19.4 |
| 8,192 | fp16 | 87.1 | 17.8 |
| 16,384 | fp16 | 90.0 | 19.5 |
| **20,480** | **K q8_0** | **30.0** | **17.0** |

**Ubuntu ROCm + CUDA（2026-06-29 重新编译版）：**

| 上下文 | 分压 | Prompt tok/s | Generation tok/s | 备注 |
|---|---|---|---|---|
| 4,096 | 7:3 | 20.4 | 23.3 | ROCm + CUDA RPC |
| 12,288 | 7:3 | 22.0 | 23.0 | mmap ✓ |
| 16,384 | 7:3 + `-b 64` | 18.9 | 21.9 | `-b 64 -ub 64` 缩小 RPC buffer |
| **20,480** | **55:45 + `-b 64`** | **18-48** | **19-21** | 55%层给笔电，KV 不超显存 |

> 重新编译版 prompt 速度显著低于预编版（86→20），推测与 ROCm 编译优化或 `ggml-rpc-server` 替代 `llama-server` 做 worker 有关。生成速度反而略优（19.4→23.3）。

**8 次连续采样（20K K q8_0，短 prompt）：**  
`10, 15, 16, 14, 14, 11, 6, 16` — 平均 12.9 tok/s，标准差 3.4。  
波动来源：RPC 层 buffer 分配抖动，短 prompt 开销占比大；长回复稳在 15-19 tok/s。

---

### 55:45 分压 → 20K 上下文

7:3 分压下 20K 失败——KV 按层分配，笔电扛 70% KV（~2.25GB）超出 4070 剩余显存。切到 **55:45** 后笔电 KV 降至 ~1.7GB，成功跑通。

**关键发现：`--main-gpu` 在 row 模式不生效。** 源码 `llama-kv-cache.cpp:229` 确认——KV 跟随 layer 分配，层在哪 GPU，KV 就在哪。`--main-gpu` 仅在 none/layer 模式有效。

| 上下文 | 分压 | Generation tok/s | 笔电 KV |
|---|---|---|---|
| 16,384 | 55:45 | 19-21 | ~1.4 GB |
| 20,480 | 55:45 + `-b 64` | 19-21 | ~1.7 GB |

**ROCm vs Vulkan 显存差异：** Windows Vulkan 有共享 GPU 内存——显存爆了自动溢到系统 RAM。Ubuntu ROCm 硬限 8GB 独显，超了直接分配失败。这就是上回 Windows 20K 7:3 成功、这回 ROCm 失败的根本原因。

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

### 坑 3：layer 模式 KV 被均分 / row 模式 main-gpu 不生效
- `--split-mode layer` 会按 tensor-split 比例切分 KV cache 到所有 GPU。
- **旧假设（错误）：** `--main-gpu 1` 在 row 模式可锁定 KV 到指定 GPU。
- **实测验证（2026-06-29）：** `--main-gpu` 在 row 模式根本不生效。源码 `llama-kv-cache.cpp:229` 确认——KV 跟随 layer 分配，层在哪 GPU，KV 就在哪。7:3 分压 = 笔电 70% KV。
- **解决：** 7:3 下笔电 KV ~2.25GB（20K q8_0）超显存 → 切 55:45 降笔电 KV 至 ~1.7GB 装得下。真正的 KV 控制靠 tensor-split 而非 main-gpu。

### 坑 4：RPC Buffer 与 `-b -ub` 批处理
- 默认批处理（2048+）会在 16K 上下文时产生 ~1.44GB 计算图缓冲，超出 RPC buffer 上限。
- **解决：** `-b 64 -ub 64` 将批处理批次从 2048 压到 64，计算图缓冲缩至几十分之一，16K 立即跑通。这是实测发现的——`-b` 不仅影响吞吐，还直接决定 RPC buffer 能否装下计算图。

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

### 坑 9：ROCm vs Vulkan 的显存溢出机制不同
- **Windows Vulkan：** 有共享 GPU 内存——独显爆了自动溢到系统 RAM。这也是上回 20K 7:3 成功的原因。
- **Ubuntu ROCm：** 硬限 8GB 独显，超出直接分配失败。同样 7:3 配置，20K 在 ROCm 下笔电 KV 2.25GB 超显存。
- **GPU 编号也反了：** Windows 下 GPU0=RPC笔电 GPU1=Vulkan桌面；Ubuntu 下 GPU0=ROCm桌面 GPU1=RPC笔电。`--main-gpu` 值在两平台含义不同。

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

## 模型扩展：32B & 20B 下载记录

2026-06-29 上午从 ModelScope 下载 6 个新模型到 `/media/joe/Data/llam/`（Ubuntu 数据盘），为更大规模分布式推理做准备：

| 模型 | 量化 | 大小 | 用途 |
|---|---|---|---|
| Qwen2.5-32B-Instruct | Q6_K | 26 GB | 32B 主力推理 |
| Qwen2.5-32B-Instruct | Q4_K_M | 19 GB | 32B 轻量版 |
| DeepSeek-R1-Distill-Qwen-32B | Q6_K | 26 GB | 思维链推理 |
| DeepSeek-R1-Distill-Qwen-32B | Q4_K_M | 19 GB | 32B 推理链轻量 |
| Qwen2.5-Coder-32B-Instruct | Q6_K | 26 GB | 32B 代码生成 |
| InternLM2.5-20B-Chat | Q6_K | 16 GB | 20B 中文对话 |

> 32B 模型 26GB × 2 卡 = 52GB，远超单卡 8GB。需要把模型拆分到两台机器上。当前下载到 Ubuntu 数据盘，待后续分布式测试。磁盘剩余 144 GB。

## 小结

---

*最后更新：2026-06-29*
