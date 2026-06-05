# Professional ML Systems Operator Verification & Throughput Benchmark
# Senior ML Systems Engineer Implementation.

import torch
import sys
import time

# Coloured diagnostic output
def info(msg): print(f"\033[92m[INFO]\033[0m {msg}")
def success(msg): print(f"\033[94m[SUCCESS]\033[0m {msg}")
def error(msg): print(f"\033[91m[ERROR]\033[0m {msg}")

# 1. Import Compiled C++/CUDA Extension
try:
    import vulnerability_benchmark_ops
    info("Successfully imported custom dynamic extension module.")
except ImportError as e:
    error(f"Failed to import the compiled custom operator module: {e}")
    error("Please run the build script first: scripts/build.sh or ASAN=1 scripts/build.sh")
    sys.exit(1)

def run_verification_suite():
    print("\n=========================================================================")
    print("        PyTorch Custom C++/CUDA Operator Verification & Performance      ")
    print("=========================================================================\n")

    # Verify active GPU device availability
    if not torch.cuda.is_available():
        error("CUDA GPU is not available on this host. A GPU is required to run the verification.")
        sys.exit(1)

    device = torch.device("cuda:0")

    # Configure representative heavy benchmark shapes
    shape = (4096, 4096)  # 16 Million float32 elements
    info(f"Configuring benchmark inputs with matrix dimensions: {shape} ({shape[0]*shape[1]} elements)")

    # 1. Generate Identical Contiguous Inputs
    # We use random float values to ensure full dynamic numerical validation.
    torch.manual_seed(42)
    input_tensor = torch.randn(shape, dtype=torch.float32, device=device)
    weight_tensor = torch.randn(shape, dtype=torch.float32, device=device)

    # Clean VRAM tracking buffers before validation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------------------
    # PART 1: Numerical Correctness Validation
    # -------------------------------------------------------------------------
    info("Executing reference PyTorch implementation (output = input * weight)...")
    torch.cuda.synchronize()
    reference_output = input_tensor * weight_tensor
    torch.cuda.synchronize()

    info("Executing custom C++/CUDA elementwise_mul operator...")
    torch.cuda.synchronize()
    custom_output = vulnerability_benchmark_ops.elementwise_mul(input_tensor, weight_tensor)
    torch.cuda.synchronize()

    # Calculate Maximum Absolute Error
    max_absolute_error = torch.max(torch.abs(reference_output - custom_output)).item()
    
    # Establish dynamic tolerance validation (strict float32 threshold is typically 1e-7)
    tolerance = 1e-7
    reproducible_status = "PASS" if max_absolute_error <= tolerance else "FAIL"

    # -------------------------------------------------------------------------
    # PART 2: Peak VRAM Tracking
    # -------------------------------------------------------------------------
    # Retrieve peak GPU virtual memory allocation recorded during custom operator execution
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_vram_bytes / (1024 * 1024)

    # -------------------------------------------------------------------------
    # PART 3: Throughput Benchmark
    # -------------------------------------------------------------------------
    info("Starting warmup iterations (10 runs to bypass driver initialization latency)...")
    for _ in range(10):
        _ = vulnerability_benchmark_ops.elementwise_mul(input_tensor, weight_tensor)
    torch.cuda.synchronize()

    iterations = 100
    info(f"Running throughput benchmark ({iterations} iterations)...")
    
    # Accurate CUDA timing using GPU dynamic events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        _ = vulnerability_benchmark_ops.elementwise_mul(input_tensor, weight_tensor)
    end_event.record()
    
    # Block CPU execution until GPU queue is fully synchronized and completed
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    throughput_steps_per_sec = iterations / (elapsed_time_ms / 1000.0)

    # -------------------------------------------------------------------------
    # DIAGNOSTIC SUMMARY PRINTING
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("               BENCHMARK RESULT REPORT            ")
    print("="*50)
    print(f"  Reproducibility Status : {reproducible_status}")
    print(f"  Maximum Absolute Error : {max_absolute_error:.8e}")
    print(f"  Throughput (steps/sec) : {throughput_steps_per_sec:.2f} steps/sec")
    print(f"  Peak VRAM Usage (MB)   : {peak_vram_mb:.2f} MB")
    print("="*50 + "\n")

    if reproducible_status == "PASS":
        success("Custom C++/CUDA operator is mathematically identical to standard PyTorch implementation!")
    else:
        error("VRAM numerical verification failed. Check kernel calculation logic.")

if __name__ == "__main__":
    run_verification_suite()
