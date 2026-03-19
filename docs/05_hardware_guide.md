# Hardware Guide

Practical guide to GPU memory, training times, cloud scaling, and getting the most
out of your hardware.

No amount of clever code can fix "not enough VRAM." This guide helps you pick the
right config for your GPU, estimate training time, troubleshoot out-of-memory errors,
and scale to the cloud when your local GPU is not enough.

---

## Table of Contents

1. [Check Your GPU](#1-check-your-gpu)
2. [VRAM Budget Breakdown](#2-vram-budget-breakdown)
3. [bf16 vs fp16: Which Precision to Use](#3-bf16-vs-fp16-which-precision-to-use)
4. [Training Time Estimates](#4-training-time-estimates)
5. [Monitoring GPU Usage](#5-monitoring-gpu-usage)
6. [Dealing with OOM (Out of Memory)](#6-dealing-with-oom-out-of-memory)
7. [How to Benchmark Your Throughput](#7-how-to-benchmark-your-throughput)
8. [Scaling to Cloud GPUs](#8-scaling-to-cloud-gpus)
9. [Multi-GPU Training](#9-multi-gpu-training)
10. [Cost Estimates for Cloud Training](#10-cost-estimates-for-cloud-training)
11. [Tips for Efficient Training](#11-tips-for-efficient-training)
12. [Inference VRAM Requirements](#12-inference-vram-requirements)

---

## 1. Check Your GPU

Before doing anything, find out what you are working with.

### From the command line

```bash
nvidia-smi
```

This shows your GPU name, VRAM total and used, driver version, and CUDA version.
You want CUDA 12.0+ and a GPU with at least 8GB VRAM.

### From Python (more detailed)

```python
import torch

if not torch.cuda.is_available():
    print("No CUDA GPU found. Training requires a CUDA-capable GPU.")
else:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:                {props.name}")
    print(f"VRAM:               {props.total_mem / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"CUDA cores:         {props.multi_processor_count * 128}")  # Approximate
    print(f"bf16 support:       {props.major >= 8}")  # sm_80+ (Ampere and newer)
    print(f"PyTorch version:    {torch.__version__}")
    print(f"CUDA version:       {torch.version.cuda}")
```

### What to look for

| GPU                 | VRAM  | Compute | bf16 Native | Good For          |
|---------------------|-------|---------|-------------|-------------------|
| RTX 3060            | 12GB  | 8.6     | Partial     | Tiny model only   |
| RTX 3080            | 10GB  | 8.6     | Partial     | Tiny, small       |
| RTX 3090            | 24GB  | 8.6     | Partial     | Up to medium      |
| RTX 4070 Ti         | 12GB  | 8.9     | Yes         | Tiny, small       |
| RTX 4080            | 16GB  | 8.9     | Yes         | Up to medium      |
| RTX 4090            | 24GB  | 8.9     | Yes         | Up to large (tight)|
| A100                | 40/80GB| 8.0    | Yes         | Everything        |
| H100                | 80GB  | 9.0     | Yes         | Everything, fast  |

"Partial" bf16 on RTX 30-series means the hardware can do bf16 operations but at
reduced throughput compared to fp16. Use `precision: "fp16"` with a GradScaler on
these cards for best performance.

---

## 2. VRAM Budget Breakdown

Every byte of GPU memory is accounted for during training. Here is where it goes,
for each model size with the default config settings.

### Tiny Model (50M params, configs/tiny.yaml)

```
Component                                  bf16 Memory
---------------------------------------------------------
Model weights (50M * 2 bytes)              ~100 MB
Optimizer states (2 AdamW buffers, fp32)   ~400 MB
Gradients (same size as weights, bf16)     ~100 MB
Activations (batch=32, seq=1024)           ~2.0 GB
PyTorch/CUDA overhead                      ~1.0 GB
---------------------------------------------------------
TOTAL                                      ~3.6 GB
```

Fits on any modern GPU (even 6GB cards). Use `batch_size: 32` with no accumulation.

### Small Model (125M params, configs/small.yaml)

```
Component                                  bf16 Memory
---------------------------------------------------------
Model weights (125M * 2 bytes)             ~250 MB
Optimizer states (2 buffers, fp32)         ~1.0 GB
Gradients (bf16)                           ~250 MB
Activations (batch=8, seq=2048)            ~4.0 GB
PyTorch/CUDA overhead                      ~1.0 GB
---------------------------------------------------------
TOTAL                                      ~6.5 GB
```

Comfortable on 16GB GPUs with `batch_size: 8`.
On 10GB (RTX 3080): use `batch_size: 4, gradient_accumulation: 8`.

### Medium Model (350M params, configs/medium.yaml)

```
Component                                  bf16 Memory
---------------------------------------------------------
Model weights (350M * 2 bytes)             ~700 MB
Optimizer states (2 buffers, fp32)         ~2.8 GB
Gradients (bf16)                           ~700 MB
Activations (batch=2, seq=2048, grad ckpt) ~3.0 GB
PyTorch/CUDA overhead                      ~1.0 GB
---------------------------------------------------------
TOTAL                                      ~8.2 GB
```

**Requires gradient checkpointing** on 16GB GPUs (`gradient_checkpointing: true`).
Without checkpointing, activations alone would use ~6 GB even at batch_size=2.

On 10GB (RTX 3080): `batch_size: 1, gradient_accumulation: 32, max_seq_len: 1024`.
It is tight but doable.

### Large Model (1B+ params, configs/large.yaml)

```
Component                                  bf16 Memory
---------------------------------------------------------
Model weights (1B * 2 bytes)               ~2.0 GB
Optimizer states (2 buffers, fp32)         ~8.0 GB
Gradients (bf16)                           ~2.0 GB
Activations (batch=8, seq=4096, grad ckpt) ~10.0 GB
PyTorch/CUDA overhead                      ~2.0 GB
---------------------------------------------------------
TOTAL                                      ~24 GB
```

Needs 24GB+ VRAM. On a single GPU, this means RTX 3090, RTX 4090, or datacenter
GPUs (A100, H100). **Do not attempt on 16GB or less.**

### Why optimizer states use so much memory

AdamW keeps two extra copies of every weight (momentum and squared gradient averages),
stored in fp32 even when the model is in bf16. For a 1B parameter model:

```
Model weights:    1B params * 2 bytes (bf16)  = 2 GB
Optimizer buffer: 1B params * 4 bytes (fp32)  = 4 GB  (momentum)
Optimizer buffer: 1B params * 4 bytes (fp32)  = 4 GB  (second moment)
Total optimizer:                                8 GB
```

The optimizer state is often the single largest consumer of VRAM during training.
This is why inference (no optimizer) uses far less memory than training.

---

## 3. bf16 vs fp16: Which Precision to Use

### The short answer

- **RTX 4080, 4090, A100, H100:** Use `precision: "bf16"`. Simpler, no GradScaler
  needed, same performance.
- **RTX 3080, 3090, 3060:** Use `precision: "fp16"` with GradScaler for best speed.
  bf16 technically works on these cards but at reduced throughput.

### The difference

```
float32:  [1 sign] [8 exponent] [23 mantissa]   range: huge, precision: high
bfloat16: [1 sign] [8 exponent] [ 7 mantissa]   range: huge, precision: reduced
float16:  [1 sign] [5 exponent] [10 mantissa]   range: small, precision: medium
```

**bfloat16** has the same exponent range as float32 (8 exponent bits). It can
represent the same range of very large and very small numbers. The tradeoff is less
precision in the mantissa (7 vs 23 bits). For neural network training, this tradeoff
is excellent -- range matters more than precision.

**float16** has a smaller exponent range (5 bits). Very small gradient values can
**underflow** to zero, which means the model stops learning those weights. The
GradScaler fixes this by scaling the loss up before the backward pass (so gradients
are larger) and scaling them back down before the optimizer step.

### GradScaler in practice

```python
# With bf16: straightforward
with autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model.compute_loss(input_ids)
loss.backward()
optimizer.step()

# With fp16: need GradScaler to prevent underflow
scaler = GradScaler()
with autocast(device_type="cuda", dtype=torch.float16):
    loss = model.compute_loss(input_ids)
scaler.scale(loss).backward()         # Loss scaled up -> gradients are larger
scaler.unscale_(optimizer)             # Unscale before gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)                 # Unscale and step (skips if inf/nan)
scaler.update()                        # Adjust scale factor
```

If you are on an RTX 3080 and see NaN losses, the most likely fix is ensuring the
GradScaler is properly enabled. Our trainer handles this automatically based on the
`precision` config setting.

---

## 4. Training Time Estimates

### GPU throughput benchmarks

Throughput varies by model size, batch size, and sequence length. These are measured
with our configs:

| GPU             | VRAM  | Tiny (tok/s) | Small (tok/s) | Medium (tok/s) |
|-----------------|-------|-------------|--------------|----------------|
| RTX 3080 (10GB) | 10GB  | ~30,000     | ~20,000      | ~12,000        |
| RTX 3090 (24GB) | 24GB  | ~40,000     | ~30,000      | ~20,000        |
| RTX 4080 (16GB) | 16GB  | ~45,000     | ~35,000      | ~22,000        |
| RTX 4090 (24GB) | 24GB  | ~65,000     | ~50,000      | ~35,000        |
| A100 (80GB)     | 80GB  | ~120,000    | ~100,000     | ~70,000        |
| H100 (80GB)     | 80GB  | ~200,000    | ~170,000     | ~120,000       |

### Estimated training time (wall clock)

Based on the total tokens in each config and throughput above:

| Config          | Total Tokens | RTX 3080 | RTX 4080 | RTX 4090 | A100    |
|-----------------|-------------|----------|----------|----------|---------|
| Tiny (50M)      | 650M        | ~6 hrs   | ~4 hrs   | ~3 hrs   | ~2 hrs  |
| Small (125M)    | 6.5B        | ~4 days  | ~2 days  | ~1.5 days| ~18 hrs |
| Medium (350M)   | 13B         | ~13 days | ~7 days  | ~4 days  | ~2 days |
| Large (1B+)     | 130B        | N/A      | N/A      | ~23 days | ~15 days|

**How to calculate:** `time = total_tokens / throughput_tokens_per_sec`

```
Small model on RTX 4080:
  Total tokens = effective_batch(32) * seq_len(2048) * max_steps(100000)
               = 32 * 2048 * 100000 = 6.55 billion tokens
  Time = 6.55B / 35,000 tok/s = 187,142 seconds = ~2.2 days
```

**Important:** These are continuous training times. Account for:
- Checkpoint saving (minor overhead, ~1 minute per save).
- GPU thermal throttling (can reduce throughput 10-20% in sustained runs).
- System interruptions (crashes, power loss -- always save checkpoints).

---

## 5. Monitoring GPU Usage

### Real-time monitoring during training

```bash
# Watch GPU usage, updates every second
# On Linux/WSL:
watch -n 1 nvidia-smi

# On Windows (PowerShell):
while ($true) { nvidia-smi; Start-Sleep -Seconds 1; Clear-Host }
```

What to look at in `nvidia-smi`:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.84       Driver Version: 545.84       CUDA Version: 12.3    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Memory-Usage         |
|  0   NVIDIA GeForce RTX 4080  |  00000000:01:00.0 On | 12345MiB / 16380MiB |
|                                |                      |                      |
|  GPU-Util  Compute M.  MIG M. |                      |                      |
|  97%       Default      N/A   |                      |                      |
+-------------------------------+----------------------+----------------------+

What you want to see:
  Memory-Usage:  Close to total (e.g., 14000/16380 MiB) -- you're using the GPU
  GPU-Util:      >90% -- the GPU is busy, not starved for data
```

### Programmatic monitoring from Python

```python
import torch

# Check current VRAM usage
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
max_allocated = torch.cuda.max_memory_allocated() / 1e9

print(f"Currently allocated: {allocated:.2f} GB")
print(f"Reserved by PyTorch: {reserved:.2f} GB")
print(f"Peak allocated:      {max_allocated:.2f} GB")

# Reset peak stats (useful for comparing different configs)
torch.cuda.reset_peak_memory_stats()
```

### What the numbers mean

- **Allocated:** Memory actively used by tensors right now.
- **Reserved:** Memory PyTorch has claimed from CUDA (may include unused cached blocks).
- **Max allocated:** Peak memory usage since the last reset. This is the number that
  determines whether you will OOM.

If `max_allocated` is within 500MB of your total VRAM, you are at risk of OOM. Reduce
batch size or enable gradient checkpointing.

---

## 6. Dealing with OOM (Out of Memory)

The dreaded `CUDA out of memory` error. Here is how to fix it, in order from least
disruptive to most disruptive.

### Fix 1: Reduce batch_size (best first step)

Activations scale linearly with batch size. Halving the batch size roughly halves
activation memory. Increase `gradient_accumulation` to keep the same effective batch.

```yaml
# Before (OOM):
training:
  batch_size: 8
  gradient_accumulation: 4

# After (fits in memory, same effective batch of 32):
training:
  batch_size: 4
  gradient_accumulation: 8

# Still OOM? Go smaller:
training:
  batch_size: 2
  gradient_accumulation: 16

# Extreme (minimum possible):
training:
  batch_size: 1
  gradient_accumulation: 32
```

### Fix 2: Enable gradient checkpointing

Saves roughly 40-50% of activation memory at the cost of ~30% slower training.
The model and results are identical.

```yaml
training:
  gradient_checkpointing: true
```

### Fix 3: Reduce sequence length

Activations also scale with sequence length. Reducing `max_seq_len` frees memory
but means the model sees less context per example.

```yaml
model:
  max_seq_len: 1024   # Down from 2048 (halves activation memory per sequence)
```

### Fix 4: Use a smaller model

If none of the above work, drop to a smaller model config:

```bash
# Instead of medium (350M), use small (125M)
python scripts/train.py --config configs/small.yaml
```

### Fix 5: Kill other GPU processes

```bash
# Check what else is using your GPU
nvidia-smi

# Common culprits:
# - Web browsers (some use GPU for rendering)
# - Other Python/Jupyter processes
# - Desktop environment compositors
```

### Quick reference: what fits where

| GPU VRAM | Tiny (50M) | Small (125M) | Medium (350M) | Large (1B+)  |
|----------|-----------|-------------|---------------|--------------|
| 6 GB     | Yes       | Tight       | No            | No           |
| 8 GB     | Yes       | Yes         | No            | No           |
| 10 GB    | Yes       | Yes         | Tight*        | No           |
| 12 GB    | Yes       | Yes         | Yes*          | No           |
| 16 GB    | Yes       | Yes         | Yes*          | No           |
| 24 GB    | Yes       | Yes         | Yes           | Tight*       |
| 40 GB    | Yes       | Yes         | Yes           | Yes          |
| 80 GB    | Yes       | Yes         | Yes           | Yes          |

*Requires gradient checkpointing, reduced batch size, or reduced seq length.

---

## 7. How to Benchmark Your Throughput

Before committing to a multi-day training run, measure your actual throughput.

### Quick benchmark

```bash
# Run training for 200 steps and note the tokens/sec
python scripts/train.py --config configs/tiny.yaml
# Watch the tok/s column in the output:
# step   100 | loss 5.8934 | ppl  362.40 | lr 3.00e-04 | tok/s 45,102
#                                                         ^^^^^^^^^^^^^
```

### Calculating training time from throughput

```python
# Example calculation
throughput = 35000  # tokens/sec (from your benchmark)

# Small model config
effective_batch = 8 * 4          # batch_size * gradient_accumulation = 32
seq_len = 2048
max_steps = 100000
total_tokens = effective_batch * seq_len * max_steps  # 6.55 billion

time_seconds = total_tokens / throughput
time_hours = time_seconds / 3600
time_days = time_hours / 24

print(f"Total tokens:  {total_tokens/1e9:.1f}B")
print(f"Throughput:    {throughput:,} tok/s")
print(f"Training time: {time_hours:.0f} hours ({time_days:.1f} days)")
```

### Optimizing throughput

If your throughput is lower than expected:

1. **Check GPU utilization** with `nvidia-smi`. If below 90%, the GPU is waiting
   for data. Increase `num_workers`.

2. **Check batch size.** Very small batches (1-2) underutilize the GPU's parallel
   hardware. If VRAM allows, increase `batch_size` even if you decrease
   `gradient_accumulation` to compensate.

3. **Try `torch.compile()`** (PyTorch 2.0+). This can give 10-30% speedup by fusing
   operations. There is a compilation overhead on the first step.

4. **Check thermal throttling.** Long runs can cause GPU temperatures to rise above
   80C, triggering throttling. Ensure good case airflow. `nvidia-smi` shows the
   temperature.

---

## 8. Scaling to Cloud GPUs

When your local GPU is not enough -- either too slow or not enough VRAM -- cloud
GPUs are the answer.

### Cloud GPU Providers

| Provider    | GPUs Available     | Price (approx)  | Best For              |
|-------------|-------------------|------------------|-----------------------|
| RunPod      | A100, H100, 4090  | $1-4/hr          | Quick experiments     |
| Lambda Labs | A100, H100        | $1-3/hr          | Longer training runs  |
| vast.ai     | Various (marketplace)| $0.50-2/hr    | Budget training       |
| AWS (p4/p5) | A100, H100        | $3-10/hr         | Enterprise, reliable  |
| GCP         | A100, H100, TPUs  | $3-8/hr          | Enterprise, TPU option|

### What to look for

- **A100 80GB:** The sweet spot for training. Enough VRAM for the large (1B+) model
  with room to spare. Good availability and reasonable price.
- **H100 80GB:** 2-3x faster than A100, but more expensive. Use for time-critical
  runs or when the hourly rate difference is small.
- **RTX 4090 (cloud):** Available on vast.ai and some RunPod instances. Good for
  small/medium models at lower cost. Not enough VRAM for the large model.
- **Multi-GPU:** 2x or 4x A100 setups for the large model or faster medium training.

### Setting up a cloud instance

```bash
# 1. SSH into your cloud instance
ssh user@your-cloud-instance

# 2. Clone the project
git clone https://github.com/your-username/codeformer.git
cd codeformer

# 3. Set up the environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,logging]"

# 4. Verify GPU access
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.0f} GB')
print(f'bf16: {torch.cuda.get_device_properties(0).major >= 8}')
"

# 5. Download and prepare data
make prepare
make tokenizer

# 6. Start training in tmux (survives SSH disconnect)
tmux new -s training
python scripts/train.py --config configs/large.yaml --wandb
# Press Ctrl+B then D to detach from tmux

# 7. Reconnect later to check progress
ssh user@your-cloud-instance
tmux attach -t training
```

### Tips for cloud training

- **Always use tmux or screen.** SSH connections drop. Without tmux, your training
  dies when the connection drops.

- **Always use wandb.** You can monitor training from your phone/laptop without
  SSH-ing back in. `--wandb` flag in the training command.

- **Download checkpoints before the instance terminates.** Cloud instances can be
  preempted (especially on spot/interruptible pricing). Periodically `rsync` or `scp`
  your checkpoints to a persistent location.

```bash
# Download checkpoints to your local machine
scp -r user@cloud-instance:~/codeformer/checkpoints/large/ ./checkpoints/large/
```

- **Use spot/interruptible instances** for 50-70% cost savings. Just make sure your
  checkpoint interval is frequent enough (every 1000-2000 steps) so you do not lose
  much progress if preempted.

---

## 9. Multi-GPU Training

For 2+ GPUs, PyTorch's **DistributedDataParallel (DDP)** is the standard approach.
Each GPU processes a different batch, and gradients are synchronized before each
weight update. The result is identical to single-GPU training, just faster.

### How DDP works

```
GPU 0: processes batch A -> computes gradients A
GPU 1: processes batch B -> computes gradients B
GPU 2: processes batch C -> computes gradients C
GPU 3: processes batch D -> computes gradients D
                |
                v
        All-Reduce: average all gradients across GPUs
                |
                v
        All GPUs: apply identical optimizer step
        (all GPUs end up with identical weights)
```

Effective batch size = `batch_size * gradient_accumulation * num_gpus`.
With 4 GPUs, you get a 4x larger effective batch at the same speed per GPU. Or you
can reduce `gradient_accumulation` by 4x and get roughly 4x faster training.

### Code changes required

Our codebase is designed to be extended for DDP. The key changes to `trainer.py`:

```python
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Initialize process group
dist.init_process_group("nccl")  # NCCL backend (fastest for NVIDIA GPUs)
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Wrap model with DDP
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler so each GPU gets different data
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, ...)

# Everything else stays the same!
# loss.backward(), optimizer.step(), etc. all work identically.
# DDP handles gradient synchronization automatically.
```

### Launch command

```bash
# 4 GPUs on one machine
torchrun --nproc_per_node=4 scripts/train.py --config configs/large.yaml

# 2 machines with 4 GPUs each (8 GPUs total)
# On machine 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=machine1-ip --master_port=29500 \
    scripts/train.py --config configs/large.yaml

# On machine 2:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=machine1-ip --master_port=29500 \
    scripts/train.py --config configs/large.yaml
```

### Adjusting config for multi-GPU

When going from 1 GPU to 4 GPUs, you typically want the same effective batch size.
Divide `gradient_accumulation` by 4:

```yaml
# Single GPU config
training:
  batch_size: 8
  gradient_accumulation: 8  # effective = 64

# 4-GPU config (same effective batch)
training:
  batch_size: 8
  gradient_accumulation: 2  # effective = 8 * 2 * 4 GPUs = 64
```

---

## 10. Cost Estimates for Cloud Training

Assuming A100 80GB at ~$2/hr (typical for RunPod/Lambda Labs):

| Config          | GPU Setup    | Training Time | Estimated Cost |
|-----------------|-------------|---------------|----------------|
| Tiny (50M)      | 1x A100     | ~2 hours      | ~$4            |
| Small (125M)    | 1x A100     | ~18 hours     | ~$36           |
| Medium (350M)   | 1x A100     | ~2 days       | ~$96           |
| Large (1B+)     | 1x A100     | ~15 days      | ~$720          |
| Large (1B+)     | 4x A100     | ~4 days       | ~$768*         |

*Multi-GPU costs more per hour but less wall-clock time. The total dollar cost is
similar because you pay for 4 GPUs. The benefit is faster iteration.

### Cheaper alternatives

- **Spot/preemptible instances:** 50-70% cheaper, but can be terminated at any time.
  Save checkpoints frequently.
- **vast.ai marketplace:** Often has GPUs at below-market rates. Quality varies.
- **Off-peak hours:** Some providers have lower rates at night/weekends.
- **Smaller GPUs for smaller models:** An RTX 4090 at $0.50/hr is much cheaper than
  an A100 at $2/hr. Fine for the tiny and small models.

### Budget planning

A reasonable budget for learning and experimentation:

```
Getting started (tiny model, local GPU):           $0 (your own hardware)
Serious training (small model, 1x A100):           ~$40
Full pipeline (medium model, 1x A100):             ~$100
Reasoning experiments (GRPO on small, local GPU):  $0
Large model (1B+, 1x A100 for 2 weeks):           ~$700
```

---

## 11. Tips for Efficient Training

### Data loading: num_workers

Set `num_workers > 0` to load data in parallel while the GPU trains. The data loading
process runs on CPU and prepares the next batch while the GPU processes the current
one.

```yaml
data:
  num_workers: 4  # Good default. Try 8 if your CPU has many cores.
```

If `num_workers: 0`, the GPU sits idle during data loading. This can reduce
throughput by 20-50%.

### pin_memory

Pre-loads data tensors into page-locked (pinned) CPU memory, which enables faster
transfer to GPU. Already enabled in our DataLoader. Costs a small amount of CPU
RAM.

```python
dataloader = DataLoader(dataset, pin_memory=True, ...)  # Already set in our code
```

### torch.compile()

PyTorch 2.0+ can compile the model's forward pass into optimized GPU kernels. This
fuses multiple small operations into fewer large ones, reducing GPU memory bandwidth
bottlenecks.

```python
# Add after model creation, before training
model = torch.compile(model)
```

Expected speedup: 10-30%. There is a one-time compilation overhead (1-5 minutes)
on the first training step. May not work with all model architectures or PyTorch
versions. If you get errors, just remove it -- it is purely an optimization.

### Logging frequency

Logging metrics every step adds overhead (especially with wandb). Our default of
logging every 100 steps is a good balance. If you need maximum throughput, increase
to 500 or 1000.

### Profile before long runs

Before committing to a multi-day training run:

```bash
# Run 200 steps and check:
# 1. tok/s (throughput) -- is it what you expected?
# 2. GPU utilization -- is it >90%?
# 3. VRAM usage -- how close to the limit?
python scripts/train.py --config configs/small.yaml
# Ctrl+C after 200 steps
```

This 5-minute test can save you hours of debugging mid-training.

### Fused optimizer kernels

Our AdamW optimizer uses `fused=True` when CUDA is available:

```python
return AdamW(param_groups, lr=learning_rate, fused=torch.cuda.is_available())
```

The fused kernel runs the entire optimizer step in a single GPU operation instead of
multiple separate ones. This is a free 5-10% speedup with no downsides.

---

## 12. Inference VRAM Requirements

Inference uses **much less** VRAM than training because there is no optimizer state,
no gradients, and no stored activations (we use KV-cache instead, which is far
smaller).

### Training vs Inference memory comparison

| Config          | Training VRAM | Inference VRAM | Ratio |
|-----------------|---------------|----------------|-------|
| Tiny (50M)      | ~3.6 GB       | ~0.3 GB        | 12x   |
| Small (125M)    | ~6.5 GB       | ~0.5 GB        | 13x   |
| Medium (350M)   | ~8.2 GB       | ~1.2 GB        | 7x    |
| Large (1B+)     | ~24 GB        | ~3.5 GB        | 7x    |

### What this means in practice

A model trained on an A100 (80GB) can serve inference on:
- An RTX 3060 (12GB) -- even the large model.
- A laptop GPU with 4-6GB -- tiny and small models.
- CPU only -- very slow but works for small models.

### Inference memory breakdown (Small model, 125M)

```
Component                                   Memory
---------------------------------------------------
Model weights (bf16)                        ~250 MB
KV-cache (varies with sequence length)     ~50-200 MB
Input/output buffers                        ~10 MB
PyTorch overhead                            ~200 MB
---------------------------------------------------
TOTAL                                       ~0.5-0.7 GB
```

The KV-cache grows with the sequence length being processed. For short completions
(a few hundred tokens), it is negligible. For long sequences (4096 tokens), it can
be a few hundred MB.

### Running inference

```bash
# Interactive generation (minimal VRAM needed)
python scripts/generate.py --checkpoint ./checkpoints/small/latest

# HTTP server (slightly more VRAM for request buffering)
python scripts/serve.py --checkpoint ./checkpoints/small/latest
```

You do NOT need the same GPU you trained on to run inference. Any CUDA GPU with
enough VRAM for the model weights will work. You can even run on CPU (set
`device="cpu"` in the code), though it will be 10-100x slower.
