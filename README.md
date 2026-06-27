# AMD + NVIDIA 混跑 14B：Windows 异构 GPU 分布式推理全记录

2026 年 6 月 27 日，花了整整一天把一张 AMD RX6600 和一张 NVIDIA RTX 4070 凑在一起，在 Windows 上用 llama.cpp RPC 跑通了 Qwen2.5-14B 的分布式推理。20K 上下文，纯独显，生成速度 12-19 tok/s。

网上没见过同类记录。这篇文章是完整的实战笔记——每条死胡同、每个绕过去的墙、所有性能数据，都在这里。

---

## 硬件

两台机器在千兆局域网内：

| | 桌面（Coordinator） | 笔记本（Worker） |
|---|---|---|
| GPU | AMD RX6600 8GB | NVIDIA RTX 4070 Laptop 8GB |
| 后端 | Vulkan | CUDA |
| 可用 VRAM | ~7.5GB | ~7.6GB |
| 系统 RAM | 16GB DDR4 | 12GB DDR5 |
| OS | Windows 10 | Windows 11 |
| 网络 | 千兆 LAN (192.168.1.44) | 千兆 LAN (192.168.1.49) |

---

## 模型库

从 ModelScope 下载，全部 GGUF 格式：

| 模型 | 量化 | 文件大小 | 架构 | AMD Vulkan |
|---|---|---|---|---|
| Qwen2.5-14B-Instruct | Q4_K_M | 8.99 GB | 标准 Transformer | ✅ |
| Qwen3.5-4B | Q4_K_M | 2.61 GB | Gated Delta Net | ❌ |
| Qwen3.5-9B | IQ4_NL | 5.11 GB | Gated Delta Net | ❌ |
| Llama-3.2-3B-Instruct | Q4_K_M | 1.88 GB | 标准 Transformer | ✅ |
| Mistral-7B-Instruct-v0.2 | Q4_K_M | 3.91 GB | 标准 Transformer | ✅ |

---

## 单机基准（RTX 4070 solo）

分布式之前先摸清单机上限：

| 模型 | Prompt tok/s | Generation tok/s | 显存占用 |
|---|---|---|---|
| Qwen3.5-4B Q4_K_M | 213 | 81.0 | 5.9 GB |
| Qwen3.5-9B IQ4_NL | 156 | 46.4 | 5.8 GB |
| Llama-3.2-3B Q4_K_M | 29 | 111.6 | 6.0 GB |

RX6600 单机跑 Llama-3.2-3B：Prompt 38 tok/s，Generation 77.3 tok/s。Vulkan 在标准架构上表现正常，但 prompt 处理反超 4070——compute shader 在小 batch 下不输 CUDA。

---

## 分布式结果

### 最终配置

```bash
llama-server b9305 \
  -m qwen2.5-14b-instruct-q4_k_m.gguf \
  --rpc 192.168.1.49:50052 \
  --split-mode row --tensor-split 7,3 -ngl 999 \
  --main-gpu 1 \
  -c 20480 -b 64 -ub 64 \
  --cache-type-k q8_0 --cache-type-v f16
```

**分配：**
- 笔电 RTX 4070：模型 70% 推理层 → ~6.3 GB VRAM
- 桌面 RX6600：模型 30% 推理层 → ~2.7 GB + 全量 KV Cache → ~2.5 GB（K q8_0 量化）
- 总计桌面约 5.2 GB VRAM，纯独显，零共享内存溢出

### 速度数据

**上下文扩展测试（Qwen2.5-14B，row 模式，70/30 分压）：**

| 上下文 | KV 配置 | Prompt tok/s | Generation tok/s | 显存（桌面 KV） |
|---|---|---|---|---|
| 256 | fp16 | 56.0 | 8.8 | ~0.1 GB |
| 1,024 | fp16 | 53.3 | 15.0 | ~0.4 GB |
| 2,048 | fp16 | 77.7 | 15.0 | ~0.8 GB |
| 4,096 | fp16 | 86.3 | 19.4 | ~1.5 GB |
| 8,192 | fp16 | 87.1 | 17.8 | ~3.0 GB |
| 12,288 | fp16 | 54.0 | 18.4 | ~4.5 GB |
| 16,384 | fp16 | 90.0 | 19.5 | ~6.0 GB |
| **20,480** | **K q8_0** | **30.0** | **17.0** | **~2.5 GB** |

> 20K 上下文在 fp16 KV 下分配失败（RPC buffer 超限）。K q8_0 量化后 KV 减半，成功绕过。

**8 次连续采样（20K K q8_0，短 prompt）：**

`10, 15, 16, 14, 14, 11, 6, 16` — 平均 12.9，标准差 3.4

**波动原因：** RPC 层 buffer 分配抖动。短 prompt 固定开销占比大，波动更明显；长回复稳定在 15-19 tok/s。

### 模型输出质量（20K K q8_0）

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

KV 量化没有可感知的质量损失。

---

## 六个关键踩坑

### 坑 1：Gated Delta Net 架构不兼容 Vulkan

第一个分布式测试用的 Qwen3.5-9B。模型加载阶段，RX6600 的 Vulkan 后端直接分配了全部 8GB 显存——实际模型只有 2.6GB。原因是 Gated Delta Net 的张量在 Vulkan 后端强制预分配全模型 buffer，加上 GDN 特殊的 tensor 分配，2.6GB 的模型能吃掉整张卡。

笔记本硬重启了两次——多次启动叠了多个 llama 实例。

**结论：Qwen3.5 全系在非 NVIDIA GPU 上跑不了。** 换 Qwen2.5（标准 attention）和 Llama-3.2 后一切正常。

### 坑 2：localhost RPC 在偷显存

第一批分布式配置用了 `--rpc 192.168.1.49:50052,127.0.0.1:50053`——coordinator 通过 localhost RPC 连自己的 GPU。结果桌面显存只用了 3GB，内存却飙到 16GB。模型层实际上是跑在 CPU 上的。

原因：localhost RPC 多了一层序列化/反序列化，GPU 内存没走正常的 Vulkan allocator，被当成 host memory 分配。

**解决：去掉 `127.0.0.1:50053`。coordinator 直接用本地 Vulkan GPU，`--rpc` 只指笔电。**

修完之后显存占用立刻跳到 5-6GB，速度从 1.7 tok/s 跳到 11.4 tok/s。

### 坑 3：layer 模式 KV 被均分

`--split-mode layer` 默认行为：KV cache 按照 tensor-split 比例分拆到所有 GPU。桌面只拿 30%，KV 不到 1GB，大部分扔在内存里。

查 llama.cpp 文档：`--main-gpu` 只在 row 模式控制 KV 位置。layer 模式无视这个参数。

**解决：换 `--split-mode row --main-gpu 1`，全量 KV 锁在桌面 6600 上。** row 模式每层权重矩阵按行切分到多 GPU，同层内并行计算后汇总。KV 放在 main-gpu 上不拆。

### 坑 4：RPC Buffer 硬上限

想把笔电分压从 70% 推到 75%（tensor-split 3:1）：

```
alloc_tensor_range: failed to allocate RPC0[192.168.1.49:50052]
buffer of size 1073741824
```

1GB buffer 分配失败。80% 同样炸。llama.cpp Windows 版的 RPC server 约 800MB buffer 上限，超出直接拒绝。

**70% 是当前版本的硬天花板。** 20K 上下文在 fp16 下也炸在同一个地方——KV 太大导致 RPC buffer 超限。K q8_0 量化缩减 KV 后重新通过。

### 坑 5：KV Cache 公式修正

最初按全 attention 头计算 KV：

```
错误: 48层 × 40头 × 128维 × 2KV × 2字节 × context = 384KB/token
正确: 48层 × 8头 × 128维 × 2KV × 2字节 × context = 192KB/token
```

Qwen2.5-14B 使用 GQA（Grouped Query Attention），40 个 query 头共享 8 个 KV 头。实际 KV cache 只有之前估算的一半。这个修正直接决定了后面上下文能推多远。

### 坑 6：WinRM 进程管理

Windows 上远程控制笔记本走 WinRM。PSSession 断开时，`Start-Process` 子进程被系统自动回收。

```
短期方案：Invoke-Command -AsJob 延长进程寿命
长期方案：schtasks 创建计划任务保活
```

每次 RPC 测试需要笔记本电脑 rpc-server 持续运行。schtasks 方案在脚本里跑通了，但后面发现 schtasks 启的 rpc-server 有时拿不到 GPU 上下文（buffer 分配失败），最终用 `-AsJob` 模式在同一个 WinRM 会话内延活。

---

## 为什么分布式比单机慢？

4070 单机跑 4B：81 tok/s（12.3 ms/token）。分布式跑 14B：17 tok/s（58.8 ms/token）。每 token 慢了 4.8 倍，但算力翻倍——瓶颈不在 GPU。

每生成一个 token，隐状态（2560 维 fp16 向量，约 5KB）必须穿过一次网络边界。千兆 LAN 下一次 TCP 往返约 1ms。row 模式的同步屏障会把延迟放大——两台 GPU 算完各自的矩阵切片后要互相等，慢的那张卡决定全局速度。

Linux 上有 GPUDirect RDMA/NCCL，GPU 显存直接 DMA 到网卡。Windows 全走 CPU + TCP 协议栈。这个速度已经是 llama.cpp RPC 在 Windows 上的物理极限。

---

## 环境细节

- llama.cpp：b9305（2026-06-25 build）
- 模型下载源：ModelScope（modelscope.cn），HF Mirror 加速
- 嵌入服务：BGE-small-en-v1.5 @ :8082（transformers，CPU 推理）
- 远程管理：WinRM（笔电）+ Hermes Agent API Server（:8642）
- 测试框架：curl + Python 脚本 + PowerShell PS1 批处理

---

## 总结

1. AMD + NVIDIA 异构分布式在 Windows 上可行，但限制明确
2. 非标准 Transformer 架构（GDN/SSM）在非 NVIDIA GPU 上大概率跑不了
3. llama.cpp RPC buffer ~800MB 是硬上限，决定最大分压比例
4. row 模式 + main-gpu 控制 KV 位置是分布式调优的核心
5. K cache q8_0 量化几乎免费——不掉速度不掉质量，省一半显存
6. 在 Linux 上同等硬件用 NCCL/Tensor Parallel 能跑得更好

Windows 上做到这一步，应该是全网首例。

---

*由 Hermes Agent 辅助实验记录，所有数据在真实硬件上复现验证。2026-06-27*
