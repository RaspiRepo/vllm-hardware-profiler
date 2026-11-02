# Design Specification  

## GPU Hardware Memory Profiler for vLLM

### 1. Goal

Provide a reliable, low-overhead, always-on telemetry layer that works across:

- Discrete NVIDIA GPUs (Turing → Blackwell, e.g. RTX 2060 Super, RTX 5060 Ti)
- Integrated Tegra GPUs (Jetson Orin Nano)

while correctly distinguishing measured metrics from unavailable hardware counters.

### 2. Architecture Overview



| Layer | Scope / Type | Core Metrics & Components |
| :--- | :--- | :--- |
| **Layer 1** | Always-on Python collector (this project) | • NVML (discrete only)<br>• CUDA mem_get_info<br>• PyTorch allocator statistics & fragmentation<br>• Host RAM / swap<br>• Jetson sysfs (EMC / GPU freq)<br>• Prometheus /metrics endpoint |
| **Layer 2** | CUPTI telemetry (separate native collector) | • Activity API, Range Profiling, PM Sampling<br>• DRAM / L2 / SM / Tensor Core counters (discovered) |
| **Layer 3** | Nsight Compute (offline / scheduled) | • Detailed per-kernel L2, DRAM, memory-pipeline analysis |
| **Layer 4** | Observability | • Prometheus + Grafana + ServiceMonitor |


### 3. Hardware Capability Matrix

| Capability                    | RTX 5060 Ti (Blackwell) | RTX 2060 Super (Turing) | Orin Nano (Ampere) |
|-------------------------------|-------------------------|-------------------------|--------------------|
| Compute capability            | 12.0                    | 7.5                     | 8.7                |
| Memory                        | GDDR7                   | GDDR6                   | Shared LPDDR5      |
| NVML discrete telemetry       | Enabled                 | Enabled                 | Disabled           |
| SM utilization                | NVML                    | NVML                    | Tegra telemetry    |
| Memory-controller utilization | NVML                    | NVML                    | EMC (sysfs)        |
| L2 hit/miss                   | CUPTI (discover)        | CUPTI (discover)        | Platform dependent |
| DRAM transactions             | CUPTI (discover)        | CUPTI (discover)        | Platform dependent |
| PyTorch allocator             | Enabled                 | Enabled                 | Enabled            |
| Host RAM / swap               | Optional                | Optional                | Enabled            |
| CUDA-visible memory           | Enabled                 | Enabled                 | Enabled            |

**Blackwell note**: Do not hard-code FP4/FP8/L2 counter names. Discover metrics from the installed CUPTI database. NVML memory utilization is **controller activity**, not bytes/s.

### 4. Design Decisions

1. **No NVML on Tegra** – integrated GPUs do not expose meaningful discrete NVML counters.
2. **CUDA `mem_get_info` always collected** – works on both discrete and integrated GPUs.
3. **PyTorch allocator is first-class** – allocated / reserved / inactive_split / retries / fragmentation ratio.
4. **Background sampling thread** – never blocks the vLLM forward path; failures are isolated.
5. **CUPTI / Nsight are out-of-band** – they can require replay or exclusive access and must not run continuously inside the production process.
6. **Architecture detection is runtime** – compute capability + Tegra heuristics, not hard-coded model names.

### 5. Metrics Exposed (Layer 1)

See README for the full Prometheus metric list and recommended PromQL.

### 6. Jetson / Orin Nano Specifics

- GPU shares system memory → never treat NVML memory util as GDDR bandwidth.
- Monitor: CUDA-visible free/total, PyTorch allocated/reserved, system RAM, swap, EMC frequency, GPU frequency.
- `tegrastats` is a complementary system tool, not a replacement for this collector.

### 7. Production Recommendations

| Setting                | Recommendation                          |
|------------------------|-----------------------------------------|
| Sampling interval      | 2–5 s                                   |
| Prometheus scrape      | 5 s                                     |
| NVML                   | Discrete GPUs only                      |
| Jetson                 | sysfs + RAM + swap + CUDA memory        |
| CUPTI                  | Separate profiling subsystem            |
| Nsight Compute         | Scheduled / offline diagnostics         |
| Multi-process vLLM     | Prefer sidecar or dedicated metrics process |

### 8. Future Extension Points

- CUPTI Range Profiling / PM Sampling as a side-car or separate DaemonSet.
- Automatic metric discovery for Blackwell FP4/FP8 counters.
- Integration with vLLM’s internal metrics registry.

