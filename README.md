# 在 Windows 上把 AMD 和 NVIDIA 显卡凑一起跑 14B 模型是什么体验

2026 年 6 月 27 日，我花了整整一天，在两台 Windows 机器上用 llama.cpp 的 RPC 功能把一张 AMD RX6600 和一张 NVIDIA RTX 4070 拼在一起，跑通了 Qwen2.5-14B 的分布式推理。20K 上下文，纯独显，生成速度 12-19 tok/s。

这篇文章记录整个过程——不是教程，是实战笔记。踩过的坑、走错的岔路、最后绕过去的墙，都在这里。

---

## 硬件环境

两台机器都在同一个千兆局域网里：

- **桌面（Coordinator）**：AMD RX6600 8GB，Vulkan 后端，16GB 内存
- **笔记本（Worker）**：NVIDIA RTX 4070 Laptop 8GB，CUDA 后端，12GB 内存

两个 GPU 都是消费级，加起来 16GB 显存，单张都塞不下 14B 的 Q4_K_M 量化模型（8.99GB）。所以必须拆开跑。

---

## 起点：先摸清单机的天花板

动手之前，先在 4070 上跑了几轮单机推理，建立基线：

| 模型 | 速度 |
|---|---|
| Qwen3.5-4B Q4_K_M | 81 tok/s |
| Qwen3.5-9B IQ4_NL | 46 tok/s |
| Llama-3.2-3B Q4_K_M | 112 tok/s |

这些数字是后面所有对比的参照系——分布式到底损失了多少速度，从这里开始算。

---

## 第一个坑：Gated Delta Net

兴冲冲拿 Qwen3.5-9B 做分布式测试——直接炸了。

RX6600 的 Vulkan 后端遇到 Qwen3.5 系列的 Gated Delta Net 架构，强制预分配全模型的 GPU buffer。2.6GB 的模型能在分配阶段吃掉全部 8GB 显存，然后溢出到内存，系统卡死。

这不是分片比例的问题。这是架构级不兼容。

**教训：Qwen3.5 系在非 NVIDIA GPU 上跑不了。换标准 Transformer 架构。**

于是下了 Qwen2.5-14B 和 Llama-3.2-3B，都是标准 attention，Vulkan 兼容。

---

## 第二个坑：localhost RPC 在偷吃显存

第一批分布式测试，桌面同时跑了 llama-cli 和 rpc-server，通过 127.0.0.1 连自己。结果桌面显存只用了 3GB，内存却飙到 16GB——模型层在 CPU 上跑。

原因：coordinator 通过 localhost RPC 层多了一次拷贝，GPU 内存没走正常的 allocator 路径，Vulkan 以为是普通 host memory。

**教训：去掉本地 RPC。coordinator 自己跑在桌面上，直接用本地 Vulkan GPU 处理自己那份层。`--rpc` 只指向笔电。**

修完之后，分布式第一次真正跑在 GPU 上——虽然速度只有 11.4 tok/s。

---

## 第三个坑：KV Cache 被均分了

layer 模式下，KV cache 按照 tensor-split 的比例被拆到两台 GPU 上。笔电 70%、桌面 30%，KV 也 7:3 分——桌面那 30% 的 KV 还不到 1GB，大部分扔在了内存里。

查了 llama.cpp 文档才搞清楚：`--main-gpu` 只在 row 模式生效。

**换 `--split-mode row` + `--main-gpu 1`，KV 全锁在桌面 6600 上。**

换完之后，桌面显存终于到了期望的 5-6GB 区间。

---

## 第四个坑：RPC Buffer 上限

想把笔电分压推到 75%，tensor-split 调到 3:1，直接炸：

```
alloc_tensor_range: failed to allocate RPC0 buffer of size 1073741824
```

1GB buffer 分配失败。llama.cpp 的 Windows RPC server 有大约 800MB 的 buffer 上限，超过就拒绝。75% 和 80% 全炸在同一个地方。

**70% 是当前版本的天花板。**

---

## 第五个坑：上下文是怎么撑上去的

最早 20K 上下文也炸——同样是 RPC buffer 不够。后来发现是 KV cache 公式自己算错了：

Qwen2.5-14B 用 GQA（8 个 KV 头，128 维），不是全 attention 头：

```
每 token KV = 48层 × 8头 × 128维 × 2(K/V) × 2字节 = 192KB
16K 上下文 = 16384 × 192KB ≈ 3GB
```

之前误用 40 头算了 6GB，多估了一倍。真实需求只有一半。

加上 `--cache-type-k q8_0` 把 K cache 量化为 8-bit，KV 总体再减半，桌面 6600 轻松装下 20K 上下文。量化几乎不掉速度，模型输出质量也没降。

---

## 最终配置

```bash
llama-server b9305 \
  -m qwen2.5-14b-instruct-q4_k_m.gguf \
  --rpc 192.168.1.49:50052 \
  --split-mode row --tensor-split 7,3 -ngl 999 \
  --main-gpu 1 -c 20480 \
  --cache-type-k q8_0 --cache-type-v f16
```

- 笔电 4070：70% 推理层，~6.3GB
- 桌面 6600：30% 推理层 + 全量 KV，~5GB
- 纯独显，零共享内存溢出

---

## 效果

| 上下文 | KV 配置 | 生成速度 |
|---|---|---|
| 8K | fp16 | 18 tok/s |
| 12K | fp16 | 18 tok/s |
| 16K | fp16 | 19 tok/s |
| 20K | K q8_0 | 17 tok/s |

8 次连续采样：最小 6，最大 19，均值 15，标准差 3.4。波动来自 RPC 层的 buffer 分配抖动，跟模型质量无关。

---

## 为什么分布式比单机慢这么多？

4070 单机跑 4B 是 81 tok/s，分布式跑 14B 只有 15 tok/s。慢了 5 倍，但不是 GPU 的问题。

每生成一个 token，隐状态（一个 2560 维的 fp16 向量，大约 5KB）必须穿过网络边界——从笔电的最后一层传到桌面的下一层，或者反过来。千兆 LAN 下一次 TCP 往返大约 1ms，每个 token 都要等这一下。

单机 12ms/token，分布式加 1ms 网络延迟，看起来只差 8%，但 row 模式的同步屏障把这个差距放大了——两台 GPU 算完各自的部分后要互相等，慢的那边决定了全局速度。

Linux 上有 GPUDirect RDMA，GPU 显存直接往网卡拉数据，跳过 CPU 和协议栈。Windows 没这条路。所以这个速度已经是 llama.cpp RPC 在 Windows 上能做到的极限了。

---

## 其他碰到的碎坑

- **WinRM 进程管理**：PSSession 断开时，`Start-Process` 的子进程会被系统回收。短期用 `-AsJob` 延活，长期用 `schtasks` 创建计划任务保活。
- **内存踩线**：coordinator 必须把完整 8.99GB 模型加载进 RAM 再分发张量。桌面 16GB 刚好踩线，mmap 模式下 OS 页缓存帮了大忙。
- **重复启动**：每次测试前没检查旧进程，多个 llama 实例叠在一起把显存和内存同时撑死，笔记本硬重启了两次。

---

## 总结

AMD + NVIDIA 混跑分布式推理，在 Windows 上是可行的。但限制也很明确：

1. **架构兼容性第一**：非标准 transformer（GDN/SSM/MLA）在非 NVIDIA GPU 上大概率跑不了
2. **RPC buffer 是硬上限**：llama.cpp Windows 版约 800MB，决定了最大分压比例
3. **网络延迟是分布式推理的隐形成本**：不是带宽不够，是每 token 都要等一次往返
4. **KV 量化几乎免费**：K q8_0 不掉速度不掉质量，省一半显存

在 Linux 上用 NCCL/GPUDirect 同等硬件能跑得更好。但如果你只有 Windows 机器，这套配置是能用的——而且网上似乎没其他人这么干过。

---

*本文由 Hermes Agent 辅助记录，所有实验在真实硬件上复现验证。*
