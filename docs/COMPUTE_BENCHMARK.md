# Measured Compute Load Estimate

## Measurement environment

This is a short, local inference benchmark, not a hardware-independent
performance guarantee.

- CPU: AMD Ryzen 7 250 with Radeon 780M Graphics (8 physical cores, 16 logical
  processors)
- Installed system RAM: 15.31 GiB
- Discrete GPU present: NVIDIA GeForce RTX 5050 Laptop GPU (8151 MiB reported)
- Active application device: CPU, as configured in `config.json`
- PyTorch: `2.14.1+cpu`; CUDA is unavailable to PyTorch
- ONNX Runtime: `1.31.0`, with `AzureExecutionProvider` and
  `CPUExecutionProvider`; `CUDAExecutionProvider` is unavailable
- Input: `sample_source/video_sample1.mp4`, 3840 x 2160, 29.970 FPS,
  240 frames
- Settings: frame skip 8, YOLO inference size 960, face confidence 0.2,
  recognition threshold 0.45

## Results

The YOLO and InsightFace models were initialized on CPU and warmed with one
frame. The following figures cover six subsequent processed frames. Model
startup, warm-up, frame decoding, database writes, image saving, display, and
web serving are excluded.

| Measurement | Result |
| --- | ---: |
| YOLO inference mean | 0.122 seconds per processed frame |
| InsightFace inference mean | 3.283 seconds per processed frame |
| Combined inference mean | 3.405 seconds per processed frame |
| Combined inference median | 3.418 seconds per processed frame |
| Inference-only throughput | 0.294 processed frames per second |
| Source-frame rate at frame skip 8 | Approximately 2.35 source frames per second |
| Process CPU use during measurement | 1250.8% of one logical processor, approximately 78.2% of all 16 logical processors |
| Total system CPU use during measurement | 99.2% |
| Process resident memory after model warm-up | 1.08 GiB |

CPU percentage is reported both per logical processor and normalized across
the machine. Windows process CPU percentages can exceed 100% because the
process uses multiple logical processors. System-wide CPU use also includes
other processes, so it is not attributable solely to this application.

At the measured rate, the CPU configuration does **not** sustain real-time
processing of this 29.970 FPS source. The source-frame-rate figure is an
inference-only estimate: it multiplies processed-frame throughput by the
configured frame interval and excludes capture, persistence, rendering, and
other pipeline work. Actual end-to-end throughput may be lower.

## GPU estimate and interpretation

Although an NVIDIA GPU is present, this Python environment has a CPU-only
PyTorch build and ONNX Runtime does not provide CUDA. The application was
therefore benchmarked only on CPU. **No GPU inference time, GPU utilization,
GPU memory use, or GPU speed-up is claimed.** A separate measurement requires
compatible CUDA-enabled PyTorch and ONNX Runtime GPU packages, a verified GPU
execution path in both inference components, and a repeated benchmark.

The benchmark is a short six-frame sample and may vary with system load,
thermal behavior, model warm-up, drivers, and library versions. Treat these
values as an estimate for this machine and sample only. For deployment
planning, repeat the measurement over a longer representative video, collect
peak memory and utilization, and include capture, persistence, and display in
the end-to-end measurement.
