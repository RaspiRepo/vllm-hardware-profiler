# Metrics `curl http://localhost:9108/metrics`

```sh
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 6306.0
python_gc_objects_collected_total{generation="1"} 731.0
python_gc_objects_collected_total{generation="2"} 73.0
# HELP python_gc_objects_uncollectable_total Uncollectable objects found during GC
# TYPE python_gc_objects_uncollectable_total counter
python_gc_objects_uncollectable_total{generation="0"} 0.0
python_gc_objects_uncollectable_total{generation="1"} 0.0
python_gc_objects_uncollectable_total{generation="2"} 0.0
# HELP python_gc_collections_total Number of times this generation was collected
# TYPE python_gc_collections_total counter
python_gc_collections_total{generation="0"} 342.0
python_gc_collections_total{generation="1"} 31.0
python_gc_collections_total{generation="2"} 2.0
# HELP python_info Python platform information
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="12",patchlevel="15",version="3.12.15"} 1.0
# HELP process_virtual_memory_bytes Virtual memory size in bytes.
# TYPE process_virtual_memory_bytes gauge
process_virtual_memory_bytes 9.192443904e+09
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 6.4708608e+08
# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.
# TYPE process_start_time_seconds gauge
process_start_time_seconds 1.79365818145e+09
# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 2.41
# HELP process_open_fds Number of open file descriptors.
# TYPE process_open_fds gauge
process_open_fds 47.0
# HELP process_max_fds Maximum number of open file descriptors.
# TYPE process_max_fds gauge
process_max_fds 1024.0
# HELP vllm_gpu_memory_used_bytes CUDA-visible used memory (bytes)
# TYPE vllm_gpu_memory_used_bytes gauge
vllm_gpu_memory_used_bytes{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 1.631453184e+09
# HELP vllm_gpu_memory_free_bytes CUDA-visible free memory (bytes)
# TYPE vllm_gpu_memory_free_bytes gauge
vllm_gpu_memory_free_bytes{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 6.530531328e+09
# HELP vllm_gpu_memory_total_bytes CUDA-visible total memory (bytes)
# TYPE vllm_gpu_memory_total_bytes gauge
vllm_gpu_memory_total_bytes{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 8.161984512e+09
# HELP vllm_nvml_sm_utilization_percent NVML SM utilization %
# TYPE vllm_nvml_sm_utilization_percent gauge
vllm_nvml_sm_utilization_percent{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 0.0
# HELP vllm_nvml_memory_utilization_percent NVML memory-controller utilization % (NOT bandwidth)
# TYPE vllm_nvml_memory_utilization_percent gauge
vllm_nvml_memory_utilization_percent{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 1.0
# HELP vllm_nvml_power_watts NVML power draw (W)
# TYPE vllm_nvml_power_watts gauge
vllm_nvml_power_watts{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 41.525
# HELP vllm_nvml_temperature_celsius NVML temperature (°C)
# TYPE vllm_nvml_temperature_celsius gauge
vllm_nvml_temperature_celsius{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 42.0
# HELP vllm_nvml_sm_clock_mhz NVML SM clock (MHz)
# TYPE vllm_nvml_sm_clock_mhz gauge
vllm_nvml_sm_clock_mhz{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 1470.0
# HELP vllm_nvml_mem_clock_mhz NVML memory clock (MHz)
# TYPE vllm_nvml_mem_clock_mhz gauge
vllm_nvml_mem_clock_mhz{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 6801.0
# HELP vllm_torch_allocated_bytes PyTorch allocated bytes
# TYPE vllm_torch_allocated_bytes gauge
vllm_torch_allocated_bytes{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 0.0
# HELP vllm_torch_reserved_bytes PyTorch reserved bytes
# TYPE vllm_torch_reserved_bytes gauge
vllm_torch_reserved_bytes{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 0.0
# HELP vllm_torch_inactive_split_bytes PyTorch inactive split bytes (fragmentation indicator)
# TYPE vllm_torch_inactive_split_bytes gauge
vllm_torch_inactive_split_bytes{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 0.0
# HELP vllm_torch_fragmentation_ratio inactive_split / reserved (0-1)
# TYPE vllm_torch_fragmentation_ratio gauge
vllm_torch_fragmentation_ratio{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 0.0
# HELP vllm_torch_allocation_retries_total Cumulative allocation retries
# TYPE vllm_torch_allocation_retries_total gauge
vllm_torch_allocation_retries_total{cc="7.5",device="0",name="NVIDIA GeForce RTX 2060 SUPER",tegra="0"} 0.0
# HELP vllm_torch_allocation_events_total Cumulative allocation events (approx via num_alloc_retries + deltas)
# TYPE vllm_torch_allocation_events_total counter
# HELP vllm_torch_deallocation_events_total Cumulative deallocation events (approx)
# TYPE vllm_torch_deallocation_events_total counter
# HELP vllm_host_ram_used_bytes Host RAM used (bytes)
# TYPE vllm_host_ram_used_bytes gauge
vllm_host_ram_used_bytes 1.2830117888e+10
# HELP vllm_host_ram_total_bytes Host RAM total (bytes)
# TYPE vllm_host_ram_total_bytes gauge
vllm_host_ram_total_bytes 3.3348734976e+10
# HELP vllm_host_swap_used_bytes Host swap used (bytes)
# TYPE vllm_host_swap_used_bytes gauge
vllm_host_swap_used_bytes 1.75583232e+08
# HELP vllm_host_swap_total_bytes Host swap total (bytes)
# TYPE vllm_host_swap_total_bytes gauge
vllm_host_swap_total_bytes 2.147479552e+09
# HELP vllm_tegra_emc_freq_mhz EMC frequency (MHz) if exposed by sysfs
# TYPE vllm_tegra_emc_freq_mhz gauge
# HELP vllm_tegra_gpu_freq_mhz GPU frequency (MHz) if exposed by sysfs
# TYPE vllm_tegra_gpu_freq_mhz gauge
# HELP vllm_hardware_collection_errors_total Number of collection failures
# TYPE vllm_hardware_collection_errors_total counter
