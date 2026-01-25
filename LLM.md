# ⚙️ LLM Training Optimization – Interview Prep

---

## ✅ What is mixed-precision training and how does it work?

**Mixed-precision training** uses both 16-bit (e.g. FP16 or BF16) and 32-bit (FP32) floating point formats during training to reduce memory usage and speed up computation — especially on GPUs like NVIDIA Tensor Cores.

### 🔧 Key idea:
- Forward/backward passes use **FP16/BF16** for speed and memory.
- Master weights and critical operations (e.g. loss scaling, accumulation) use **FP32** for numerical stability.

### 🔁 Workflow:
1. Convert model and data to FP16/BF16
2. Maintain FP32 master weights for updates
3. Use **loss scaling** (in FP16) to prevent underflow

---

## ✅ What are the benefits and tradeoffs of FP16 vs BF16?

| Feature        | **FP16 (IEEE 754)**             | **BF16 (bfloat16)**              |
|----------------|-------------------------------|-------------------------------|
| Exponent bits  | 5                             | 8                             |
| Mantissa bits  | 10                            | 7                             |
| Dynamic range  | **Smaller**                   | **Larger (same as FP32)**     |
| Precision      | Higher                        | Lower                         |
| Hardware support | NVIDIA/TPU (w/ loss scaling) | TPU / Newer NVIDIA GPUs       |
| Requires loss scaling | ✅ Yes                 | ❌ Often no                   |

**Summary:**
- **FP16** has better precision, but smaller range → needs loss scaling
- **BF16** has large dynamic range → more stable, easier to use

---

## ✅ What is gradient checkpointing?

**Gradient checkpointing** is a memory-saving technique where **some intermediate activations are not stored during the forward pass**. Instead, they are recomputed during backpropagation as needed.

### 🚀 Trade-off:
- **Saves memory** (up to 3x)
- **Increases compute** due to recomputation

Used in very large models like GPT-3, PaLM, etc.

---

## ✅ What is Distributed Data Parallel (DDP)?

**DDP** is a training strategy where each GPU holds a **copy of the full model** and trains on a **different subset of the data**.

### ⛓️ Key characteristics:
- Gradients are **averaged across GPUs** after backward pass
- Each GPU has **independent forward/backward**, but syncs gradients

### ✅ Benefits:
- Simple and efficient
- Scales well across nodes
- Supported in PyTorch via `torch.nn.parallel.DistributedDataParallel`

---

## ✅ What is Fully Sharded Data Parallel (FSDP)?

**FSDP** shards model **weights**, **gradients**, and **optimizer states** across GPUs to reduce memory usage.

### 📦 What it shards:
- Model weights
- Gradients
- Optimizer states

### 🧠 Benefits:
- Trains very large models on fewer GPUs
- More memory efficient than DDP

### 🛠️ Tools:
- PyTorch’s `torch.distributed.fsdp` module

---

## ✅ What is ZeRO (Zero Redundancy Optimizer) and its stages?

**ZeRO** is a deep learning optimization technique designed by Microsoft DeepSpeed to reduce memory redundancy in large-scale training.

### 🔢 Three stages:

| Stage | What is sharded                        |
|-------|----------------------------------------|
| **ZeRO-1** | Optimizer states                    |
| **ZeRO-2** | + Gradients                         |
| **ZeRO-3** | + Model parameters (full weight sharding) |

Each stage builds on the previous to further reduce memory usage.

---

## ✅ How does ZeRO enable large-scale model training?

By sharding and offloading redundant memory components across GPUs, ZeRO:
- **Eliminates memory duplication**
- Enables **larger model size** (100B+ parameters)
- Works with **Mixed Precision + Offloading (CPU, NVMe)**

It allows training models like BLOOM, GPT-NeoX, and others on clusters with limited GPU memory.

---

## 🧠 Bonus: Combine for max scalability

In practice, large-scale LLM training stacks combine:
- **FSDP / ZeRO-3** for sharding
- **Mixed precision (FP16/BF16)**
- **Gradient checkpointing**
- **Activation offloading / CPU offload**
- **Flash Attention / KV cache optimization**

