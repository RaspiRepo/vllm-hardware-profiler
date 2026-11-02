#!/usr/bin/env python3
"""
Adaptive GPU Hardware Memory Profiler for vLLM
Layer-1 always-on telemetry:
  - Runtime architecture / capability detection
  - NVML for discrete GPUs only
  - Explicit Tegra / Jetson exclusion of NVML
  - CUDA-visible memory (works on integrated GPUs)
  - PyTorch CUDA allocator statistics + fragmentation
  - Host RAM / swap
  - Jetson sysfs (EMC / GPU freq where available)
  - Prometheus metrics endpoint
  - Background sampling thread with failure isolation
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from prometheus_client import (
    Counter,
    Gauge,
    start_http_server,
)

# Optional NVML – only used on discrete GPUs
try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

logger = logging.getLogger("vllm.hardware_profiler")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

# Labels: device_index, device_name, compute_capability, is_tegra
LABEL_NAMES = ["device", "name", "cc", "tegra"]

gpu_memory_used = Gauge(
    "vllm_gpu_memory_used_bytes",
    "CUDA-visible used memory (bytes)",
    LABEL_NAMES,
)
gpu_memory_free = Gauge(
    "vllm_gpu_memory_free_bytes",
    "CUDA-visible free memory (bytes)",
    LABEL_NAMES,
)
gpu_memory_total = Gauge(
    "vllm_gpu_memory_total_bytes",
    "CUDA-visible total memory (bytes)",
    LABEL_NAMES,
)

# NVML (discrete only)
nvml_sm_util = Gauge(
    "vllm_nvml_sm_utilization_percent",
    "NVML SM utilization %",
    LABEL_NAMES,
)
nvml_mem_util = Gauge(
    "vllm_nvml_memory_utilization_percent",
    "NVML memory-controller utilization % (NOT bandwidth)",
    LABEL_NAMES,
)
nvml_power = Gauge(
    "vllm_nvml_power_watts",
    "NVML power draw (W)",
    LABEL_NAMES,
)
nvml_temp = Gauge(
    "vllm_nvml_temperature_celsius",
    "NVML temperature (°C)",
    LABEL_NAMES,
)
nvml_clock_sm = Gauge(
    "vllm_nvml_sm_clock_mhz",
    "NVML SM clock (MHz)",
    LABEL_NAMES,
)
nvml_clock_mem = Gauge(
    "vllm_nvml_mem_clock_mhz",
    "NVML memory clock (MHz)",
    LABEL_NAMES,
)

# PyTorch allocator
torch_allocated = Gauge(
    "vllm_torch_allocated_bytes",
    "PyTorch allocated bytes",
    LABEL_NAMES,
)
torch_reserved = Gauge(
    "vllm_torch_reserved_bytes",
    "PyTorch reserved bytes",
    LABEL_NAMES,
)
torch_inactive_split = Gauge(
    "vllm_torch_inactive_split_bytes",
    "PyTorch inactive split bytes (fragmentation indicator)",
    LABEL_NAMES,
)
torch_fragmentation_ratio = Gauge(
    "vllm_torch_fragmentation_ratio",
    "inactive_split / reserved (0-1)",
    LABEL_NAMES,
)
torch_allocation_retries = Gauge(
    "vllm_torch_allocation_retries_total",
    "Cumulative allocation retries",
    LABEL_NAMES,
)
torch_allocation_events = Counter(
    "vllm_torch_allocation_events_total",
    "Cumulative allocation events (approx via num_alloc_retries + deltas)",
    LABEL_NAMES,
)
torch_deallocation_events = Counter(
    "vllm_torch_deallocation_events_total",
    "Cumulative deallocation events (approx)",
    LABEL_NAMES,
)

# Host
host_ram_used = Gauge(
    "vllm_host_ram_used_bytes",
    "Host RAM used (bytes)",
)
host_ram_total = Gauge(
    "vllm_host_ram_total_bytes",
    "Host RAM total (bytes)",
)
host_swap_used = Gauge(
    "vllm_host_swap_used_bytes",
    "Host swap used (bytes)",
)
host_swap_total = Gauge(
    "vllm_host_swap_total_bytes",
    "Host swap total (bytes)",
)

# Jetson / Tegra extras
tegra_emc_freq = Gauge(
    "vllm_tegra_emc_freq_mhz",
    "EMC frequency (MHz) if exposed by sysfs",
    LABEL_NAMES,
)
tegra_gpu_freq = Gauge(
    "vllm_tegra_gpu_freq_mhz",
    "GPU frequency (MHz) if exposed by sysfs",
    LABEL_NAMES,
)

# Health
collection_errors = Counter(
    "vllm_hardware_collection_errors_total",
    "Number of collection failures",
    ["component"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_tegra() -> bool:
    """Detect Jetson / Tegra platforms."""
    # Common indicators
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
            if "jetson" in model or "orin" in model or "tegra" in model:
                return True
    except Exception:
        pass
    # CUDA device name heuristic
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i).lower()
            if "orin" in name or "tegra" in name or "jetson" in name:
                return True
    return False


def _read_sysfs_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _find_tegra_freq_nodes() -> Dict[str, Optional[str]]:
    """Best-effort discovery of EMC / GPU frequency nodes (JetPack dependent)."""
    candidates = {
        "emc": [
            "/sys/kernel/debug/bpmp/debug/clk/emc/rate",
            "/sys/devices/platform/13800000.emc/devfreq/*/cur_freq",
            "/sys/class/devfreq/*/cur_freq",
        ],
        "gpu": [
            "/sys/devices/platform/17000000.gpu/devfreq/*/cur_freq",
            "/sys/class/devfreq/*/cur_freq",
            "/sys/kernel/debug/bpmp/debug/clk/gpc0clk/rate",
        ],
    }
    found = {"emc": None, "gpu": None}
    for kind, paths in candidates.items():
        for p in paths:
            # Expand globs simply
            if "*" in p:
                import glob
                matches = glob.glob(p)
                if matches:
                    found[kind] = matches[0]
                    break
            elif os.path.exists(p):
                found[kind] = p
                break
    return found


# ---------------------------------------------------------------------------
# Main profiler class
# ---------------------------------------------------------------------------

class HardwareProfiler:
    def __init__(
        self,
        sample_interval: float = 2.0,
        prometheus_port: int = 9108,
        enable_prometheus: bool = True,
        devices: Optional[List[int]] = None,
    ):
        self.sample_interval = max(0.5, sample_interval)
        self.prometheus_port = prometheus_port
        self.enable_prometheus = enable_prometheus
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.is_tegra = _is_tegra()
        self.device_count = torch.cuda.device_count()
        self.devices = devices if devices is not None else list(range(self.device_count))

        # Per-device metadata
        self.device_info: Dict[int, Dict] = {}
        for idx in self.devices:
            props = torch.cuda.get_device_properties(idx)
            cc = f"{props.major}.{props.minor}"
            name = torch.cuda.get_device_name(idx)
            self.device_info[idx] = {
                "name": name,
                "cc": cc,
                "tegra": "1" if self.is_tegra else "0",
                "label": (str(idx), name, cc, "1" if self.is_tegra else "0"),
            }

        # NVML init (discrete only)
        self.nvml_handles: Dict[int, object] = {}
        if _NVML_AVAILABLE and not self.is_tegra:
            try:
                pynvml.nvmlInit()
                for idx in self.devices:
                    # Map CUDA ordinal → NVML device (usually 1:1)
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    self.nvml_handles[idx] = handle
                logger.info("NVML initialized for discrete GPU(s)")
            except Exception as e:
                logger.warning(f"NVML init failed: {e}")
                self.nvml_handles = {}
        else:
            if self.is_tegra:
                logger.info("Tegra/Jetson detected – NVML telemetry disabled")
            else:
                logger.info("pynvml not available – NVML metrics disabled")

        # Jetson sysfs nodes
        self.tegra_nodes = _find_tegra_freq_nodes() if self.is_tegra else {}

        # Track previous allocator stats for event deltas
        self._prev_stats: Dict[int, Dict] = {}

        if self.enable_prometheus:
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics listening on :{self.prometheus_port}")

        atexit.register(self.stop)

    # ------------------------------------------------------------------
    # Collection methods (never raise into the sampling loop)
    # ------------------------------------------------------------------

    def _collect_cuda_memory(self, device: int) -> None:
        label = self.device_info[device]["label"]
        try:
            free_b, total_b = torch.cuda.mem_get_info(device)
            used_b = total_b - free_b
            gpu_memory_free.labels(*label).set(free_b)
            gpu_memory_total.labels(*label).set(total_b)
            gpu_memory_used.labels(*label).set(used_b)
        except Exception as e:
            collection_errors.labels("cuda_memory").inc()
            logger.debug(f"CUDA mem_get_info failed on {device}: {e}")

    def _collect_torch_allocator(self, device: int) -> None:
        label = self.device_info[device]["label"]
        try:
            # Switch context only if necessary; mem_get_info / memory_stats
            # accept the device argument in recent PyTorch.
            stats = torch.cuda.memory_stats(device=device)

            allocated = stats.get("allocated_bytes.all.current", 0)
            reserved = stats.get("reserved_bytes.all.current", 0)
            inactive = stats.get("inactive_split_bytes.all.current", 0)
            retries = stats.get("num_alloc_retries", 0)

            torch_allocated.labels(*label).set(allocated)
            torch_reserved.labels(*label).set(reserved)
            torch_inactive_split.labels(*label).set(inactive)
            torch_allocation_retries.labels(*label).set(retries)

            frag = (inactive / reserved) if reserved > 0 else 0.0
            torch_fragmentation_ratio.labels(*label).set(frag)

            # Approximate allocation / deallocation event rates
            prev = self._prev_stats.get(device, {})
            prev_alloc = prev.get("allocated", allocated)
            prev_reserved = prev.get("reserved", reserved)

            # crude but useful for rate() in Prometheus
            if allocated > prev_alloc:
                torch_allocation_events.labels(*label).inc(allocated - prev_alloc)
            if reserved < prev_reserved:
                # rough deallocation signal
                torch_deallocation_events.labels(*label).inc(prev_reserved - reserved)

            self._prev_stats[device] = {
                "allocated": allocated,
                "reserved": reserved,
            }
        except Exception as e:
            collection_errors.labels("torch_allocator").inc()
            logger.debug(f"torch.memory_stats failed on {device}: {e}")

    def _collect_nvml(self, device: int) -> None:
        if device not in self.nvml_handles:
            return
        label = self.device_info[device]["label"]
        handle = self.nvml_handles[device]
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            nvml_sm_util.labels(*label).set(util.gpu)
            nvml_mem_util.labels(*label).set(util.memory)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                nvml_power.labels(*label).set(power)
            except pynvml.NVMLError:
                pass

            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                nvml_temp.labels(*label).set(temp)
            except pynvml.NVMLError:
                pass

            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                )
                mem_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
                nvml_clock_sm.labels(*label).set(sm_clock)
                nvml_clock_mem.labels(*label).set(mem_clock)
            except pynvml.NVMLError:
                pass

        except Exception as e:
            collection_errors.labels("nvml").inc()
            logger.debug(f"NVML collection failed on {device}: {e}")

    def _collect_host(self) -> None:
        try:
            vm = psutil.virtual_memory()
            host_ram_used.set(vm.used)
            host_ram_total.set(vm.total)

            swap = psutil.swap_memory()
            host_swap_used.set(swap.used)
            host_swap_total.set(swap.total)
        except Exception as e:
            collection_errors.labels("host").inc()
            logger.debug(f"Host memory collection failed: {e}")

    def _collect_tegra(self, device: int) -> None:
        if not self.is_tegra:
            return
        label = self.device_info[device]["label"]
        try:
            # EMC frequency
            emc_path = self.tegra_nodes.get("emc")
            if emc_path:
                val = _read_sysfs_int(emc_path)
                if val is not None:
                    # rate is often in Hz
                    mhz = val / 1_000_000 if val > 10_000 else val
                    tegra_emc_freq.labels(*label).set(mhz)

            # GPU frequency
            gpu_path = self.tegra_nodes.get("gpu")
            if gpu_path:
                val = _read_sysfs_int(gpu_path)
                if val is not None:
                    mhz = val / 1_000_000 if val > 10_000 else val
                    tegra_gpu_freq.labels(*label).set(mhz)
        except Exception as e:
            collection_errors.labels("tegra_sysfs").inc()
            logger.debug(f"Tegra sysfs collection failed: {e}")

    def _sample_once(self) -> None:
        for device in self.devices:
            self._collect_cuda_memory(device)
            self._collect_torch_allocator(device)
            self._collect_nvml(device)
            self._collect_tegra(device)
        self._collect_host()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        logger.info(
            f"Hardware profiler started "
            f"(interval={self.sample_interval}s, tegra={self.is_tegra}, "
            f"devices={self.devices})"
        )
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                self._sample_once()
            except Exception as e:
                collection_errors.labels("sample_loop").inc()
                logger.warning(f"Unexpected error in sample loop: {e}")
            elapsed = time.monotonic() - start
            sleep_for = max(0.0, self.sample_interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="vllm-hardware-profiler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if _NVML_AVAILABLE and self.nvml_handles:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        logger.info("Hardware profiler stopped")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive GPU Hardware Memory Profiler for vLLM"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Sampling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9108,
        help="Prometheus metrics port (default: 9108)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated CUDA device indices (default: all)",
    )
    args = parser.parse_args()

    devices = None
    if args.devices:
        devices = [int(x) for x in args.devices.split(",")]

    profiler = HardwareProfiler(
        sample_interval=args.interval,
        prometheus_port=args.port,
        devices=devices,
    )
    profiler.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        profiler.stop()


if __name__ == "__main__":
    main()