# Vulnerability Benchmark Ops

## Overview

Vulnerability Benchmark Ops is an educational PyTorch C++/CUDA extension designed to demonstrate memory-safety vulnerabilities in custom deep learning operators.

The project implements a GPU-accelerated elementwise multiplication operator and intentionally contains a stack-based buffer overflow vulnerability for security benchmarking, dynamic analysis, and AddressSanitizer (ASAN) testing.

This repository is intended for:

* PyTorch C++ Extension learning
* CUDA kernel development
* Memory safety research
* AddressSanitizer experimentation
* Secure ML systems education

---

# Architecture

```text
Python
│
├── test_trigger.py
├── test_verification.py
│
▼
PyBind11 Interface
(custom_operator.cpp)
│
▼
CUDA Launcher
(custom_operator.cpp)
│
▼
CUDA Kernel
(kernel.cu)
│
▼
GPU Execution
```

The operator is exposed to Python through pybind11 bindings and executes a custom CUDA kernel on NVIDIA GPUs.

---

# Repository Structure

```text
vulnerability_benchmark_ops/
├── modules/
│   ├── custom_operator.cpp
│   └── kernel.cu
├── scripts/
│   ├── build.sh
│   └── run_benchmark.sh
├── test_trigger.py
├── test_verification.py
├── AI_COLLABORATION_LOG.md
├── VULNERABILITY_ANALYSIS.md
├── requirements.txt
└── README.md
```

---

# Components

## custom_operator.cpp

Responsibilities:

* Tensor validation
* Shape checking
* Device validation
* CUDA stream retrieval
* Output tensor allocation
* CUDA kernel launch
* PyBind11 registration

## kernel.cu

Responsibilities:

* CUDA kernel implementation
* Elementwise multiplication
* GPU thread indexing
* Device-side execution

## test_trigger.py

Demonstrates:

* Safe execution path
* Adversarial execution path
* Vulnerability trigger conditions

## test_verification.py

Performs:

* Numerical correctness validation
* Throughput benchmarking
* VRAM usage measurement
* Reproducibility testing

---

# Prerequisites

* Python 3.12+
* PyTorch with CUDA support
* CUDA Toolkit
* NVIDIA GPU
* GCC/G++

Install dependencies:

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch setuptools
```

---

# Build Instructions

Compile the custom CUDA extension:

```bash
python3 debug_setup.py build_ext --inplace
```

Successful compilation generates:

```text
vulnerability_benchmark_ops.cpython-312-x86_64-linux-gnu.so
```

Verify import:

```bash
python3 -c "
import vulnerability_benchmark_ops
print('IMPORT SUCCESS')
"
```

Expected output:

```text
IMPORT SUCCESS
```

---

# Running the Vulnerability Demonstration

Execute:

```bash
python3 test_trigger.py
```

The script performs:

1. Safe tensor execution
2. Adversarial tensor execution
3. Vulnerability trigger attempt

Example:

```text
PHASE 1: Safe Input
SUCCESS

PHASE 2: Adversarial Input
Triggering C++/CUDA Custom Operator call now...
```

---

# Vulnerability Description

## CWE-121: Stack-Based Buffer Overflow

This benchmark intentionally contains a stack-based buffer overflow vulnerability for educational and security analysis purposes.

### Vulnerable Pattern

```cpp
float stack_buffer[16];

for (int64_t i = 0; i < numel; ++i) {
    stack_buffer[i] = 42.0f;
}
```

### Trigger Condition

The vulnerability is triggered when:

```text
numel > 16
```

Example:

```python
torch.ones((2, 12))
```

Results in:

```text
numel = 24
```

which exceeds the fixed-size stack allocation.

---

# Security Impact

Potential consequences include:

* Stack corruption
* Undefined behavior
* Silent memory corruption
* Process instability
* Application crashes

Example runtime behavior:

```text
*** stack smashing detected ***: terminated
```

The vulnerability is intentionally included for research and benchmarking purposes.

---

# AddressSanitizer (ASAN)

Build with AddressSanitizer instrumentation:

```bash
ASAN=1 scripts/build.sh
```

Run using:

```bash
export LD_PRELOAD=$(gcc -print-file-name=libasan.so)
export ASAN_OPTIONS=detect_leaks=0

python3 test_trigger.py
```

ASAN instrumentation enables runtime detection of memory-safety violations.

---

# Numerical Verification

The custom operator output is compared against the PyTorch reference implementation.

Run:

```bash
python3 test_verification.py
```

Reference comparison:

```python
custom_output = elementwise_mul(input, weight)
reference_output = input * weight
```

Expected result:

```text
Maximum Absolute Error: 0.0
```

This confirms numerical equivalence with native PyTorch multiplication.

---

# Benchmark Results

The verification script reports:

* Maximum Absolute Error
* Reproducibility Status
* Throughput
* Peak VRAM Usage

Example:

```text
Reproducibility Status : PASS
Maximum Absolute Error : 0.00000000e+00
Throughput (steps/sec) : 34.92
Peak VRAM Usage (MB)   : 384.00
```

---

# Human Verification

All generated code was manually reviewed and validated through:

* Build verification
* Python import verification
* CUDA execution validation
* Numerical correctness testing
* Vulnerability review
* Benchmark execution

---

# AI Collaboration

AI assistance was used during:

* Repository scaffolding
* CUDA extension development
* PyBind11 integration
* Vulnerability design
* Security analysis
* Verification tooling
* Documentation generation

All AI-generated code was manually reviewed, tested, and validated before acceptance.

See:

```text
AI_COLLABORATION_LOG.md
```

for complete collaboration details.

---

# Screenshots Included

```text
01_repository_structure.png
02_vulnerable_implementation.png
03_extension_build_success.png
04_import_success.png
05_stack_smashing_detection.png
06_verification_results.png
07_git_history.png
```

---



