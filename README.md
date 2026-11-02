# vllm-hardware-profiler

# Adaptive GPU Hardware Memory Profiler for vLLM

Production-ready, multi-architecture GPU telemetry collector designed for vLLM fleets.

Supports:
- Discrete NVIDIA GPUs (Turing → Blackwell) via NVML + CUDA
- Jetson / Tegra (Orin Nano, etc.) via CUDA + sysfs + host memory
- PyTorch CUDA allocator statistics & fragmentation
- Prometheus metrics endpoint
- Background sampling with failure isolation

## Architecture (Layer 1 – Always-on)

| Capability                    | Discrete GPU | Jetson / Tegra |
|-------------------------------|--------------|----------------|
| CUDA-visible memory           | ✅           | ✅             |
| PyTorch allocator + fragmentation | ✅       | ✅             |
| NVML (SM / mem util, power, clocks) | ✅     | ❌ (disabled)  |
| Host RAM / swap               | ✅           | ✅             |
| EMC / GPU frequency (sysfs)   | –            | ✅ (best-effort) |
| CUPTI / Nsight                | Separate Layer 2/3 | Separate |

**Important**: Memory-controller utilization from NVML is **not** memory bandwidth. L2 hit rates and true DRAM throughput require CUPTI / Nsight (not included in this always-on collector).

## Quick Start (local)

```bash
# Create venv & install
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run
python hardware_profiler.py --interval 2 --port 9108



# Docker build 

```sh 
docker build -t <your-registry>/vllm-hardware-profiler:latest .
docker push <your-registry>/vllm-hardware-profiler:latest


