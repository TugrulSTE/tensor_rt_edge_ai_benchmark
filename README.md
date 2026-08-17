# tensor_rt_edge_ai_benchmark
Edge AI Pipeline: Custom CUDA Pre-processing Kernel &amp; TensorRT Inference via Different Quantization Techniques on Jetson Orin Nano


## Data Preprocessing Benchmark

| Implementation | Platform / Execution | Latency | Speedup |
| :--- | :--- | :---: | :---: |
| **CPU Baseline** | Host Thread (`g++ -O2`) | `0.0851 ms` | Reference (1.0x) |
| **Custom CUDA Kernel** | GPU Parallel (`nvcc -O2`) | `0.0205 ms` | **~4.15x Faster** |

---

## TensorRT Engine & Inference Performance

| Metric / Resource | FP16 (Half) | FP32 (Full) | INT8 (Calibrated) |
| :--- | :---: | :---: | :---: |
| **Deserialization Time** | 11.92 ms | 19.53 ms | 13.39 ms |
| **Device Persistent Memory** | 0 B | 154,112 B | 0 B |
| **Host Persistent Memory** | 38,304 B | 36,704 B | 47,632 B |
| **Scratch Memory** | 3.38 MB | 6.66 MB | 1.69 MB |
| **Inference Latency (GPU)** | **1.666 ms** | **2.767 ms** | **0.976 ms** |
| **Accuracy** | Unknown | 96.20% | 94.73% |
| **Weighted F1 Score** | Unknown | 0.96 | 0.95 |
