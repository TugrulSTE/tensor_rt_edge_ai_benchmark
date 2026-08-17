# TensorRT Edge AI One-Image Inferencing Via Native C++ APIs & Benchmark
Edge AI Pipeline: Custom CUDA Pre-processing Kernel &amp; TensorRT Inference via Different Quantization Techniques on Jetson Orin Nano

## Goals & Pre-Conditions
This project aims to classify the images on Edge AI devices, in this case, it is NVIDIA Jetson Orin Nano, analyze the performance across different techniques. Instead of using standard Edge Impulse SDK files, the raw **.tflite** file is converted into device-specific **.engine** file, native CUDA API methods are utilized to enable the GPU cores and accelerate the inferencing process.

## Methodology
**.engine** file includes all the necessary mathematical operation from beginning to producing output classes. 


## Flowchart
[Host CPU Memory]
       │
       │  raw_features (uint32 hex array - 4 bytes/pixel)
       │  cudaMemcpyHostToDevice (Reduced PCIe / Memory Traffic)
       ▼
[GPU VRAM (Device Memory)]
       │
       │  d_raw_in (Packed Hex)
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Custom CUDA Kernel: listExtract<<<blocks, threads>>>       │
│  - 1 Thread per Pixel                                       │
│  - Parallel Bitwise Unpacking: (Hex >> Shift) & 0xFF        │
│  - In-place FP32 Conversion & Channel Interleaving (HWC)    │
└─────────────────────────────────────────────────────────────┘
       │
       │  d_input (Extracted FP32 RGB Tensor)
       ▼
┌─────────────────────────────────────────────────────────────┐
│  TensorRT Execution Context (INT8 / FP32 Engine)            │
│  - context->executeV2(bindings)                             │
│  - Zero-Copy / Zero-Overhead Input Consumption              │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
[GPU VRAM: d_output] ──► cudaMemcpyDeviceToHost ──► [Host: Results]

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
