# Dynamic Security Benchmark Trigger Script for Custom C++/CUDA Operator
# Senior Security & ML Systems Engineer Implementation.

import torch
import sys

# Coloured diagnostic output
def info(msg): print(f"\033[92m[INFO]\033[0m {msg}")
def warning(msg): print(f"\033[93m[WARNING]\033[0m {msg}")
def error(msg): print(f"\033[91m[ERROR]\033[0m {msg}")

# 1. Import Compiled C++/CUDA Extension
try:
    import vulnerability_benchmark_ops
    info("Successfully imported compiled CUDA extension 'vulnerability_benchmark_ops'.")
except ImportError as e:
    error(f"Failed to import the compiled custom operator module: {e}")
    error("Please run the build script first: scripts/build.sh or ASAN=1 scripts/build.sh")
    sys.exit(1)

def run_security_benchmark():
    print("\n=========================================================================")
    print("        Educational Custom C++/CUDA Operator Vulnerability Benchmark     ")
    print("=========================================================================\n")

    # Verify active GPU device availability (required for CUDA custom extensions)
    if not torch.cuda.is_available():
        error("CUDA GPU is not available on this host. A GPU is required to execute CUDA kernels.")
        sys.exit(1)

    device = torch.device("cuda:0")

    # =========================================================================
    # PHASE 1: Normal Tensor Example (Within Safe Bounds)
    # =========================================================================
    print("-------------------------------------------------------------------------")
    print("PHASE 1: Execution with a Normal (Safe) Tensor Input")
    print("-------------------------------------------------------------------------")
    
    # We define a 1D tensor of shape (10,). Total elements (numel) = 10.
    normal_shape = (10,)
    info(f"Target Tensor Shape: {normal_shape}")
    info(f"Total Elements (numel): {10} (Threshold Limit = 16)")
    info("Execution Phase: Safe profiling path (No Stack Overflow).")
    info("Expected Result: Perfect execution, returning contiguous elementwise product.")

    # Allocate inputs on GPU
    input_normal = torch.ones(normal_shape, dtype=torch.float32, device=device)
    weight_normal = torch.full(normal_shape, 2.0, dtype=torch.float32, device=device)

    # Invoke operator
    output_normal = vulnerability_benchmark_ops.elementwise_mul(input_normal, weight_normal)
    print(f"-> SUCCESS! Output elements: {output_normal.tolist()}")
    print("-> Verification: Valid input size successfully executed without memory safety warnings.\n")

    # =========================================================================
    # PHASE 2: Adversarial Tensor Example (Exceeding Bounds)
    # =========================================================================
    print("-------------------------------------------------------------------------")
    print("PHASE 2: Execution with an Adversarial (Oversized) Tensor Input")
    print("-------------------------------------------------------------------------")
    
    # We define a 2D tensor of shape (2, 12). Total elements (numel) = 2 * 12 = 24.
    # This exceeds the static compile-time STACK_LIMIT of 16 float elements in C++.
    adversarial_shape = (2, 12)
    warning(f"Target Adversarial Tensor Shape: {adversarial_shape}")
    warning(f"Total Elements (numel): {2 * 12} (Threshold Limit = 16)")
    warning("Execution Phase: Out-of-bounds stack write (Vulnerability Trigger).")
    warning("Expected Result (ASAN Build): Immediate runtime abort with Stack-Buffer-Overflow report.")
    warning("Expected Result (Normal Build): Silent stack corruption (unstable state).")

    # Allocate inputs on GPU
    input_adv = torch.ones(adversarial_shape, dtype=torch.float32, device=device)
    weight_adv = torch.full(adversarial_shape, 3.0, dtype=torch.float32, device=device)

    print("\n-> Triggering C++/CUDA Custom Operator call now...\n")
    sys.stdout.flush()

    # Invoke operator (This triggers the stack buffer overflow write loop in C++)
    output_adv = vulnerability_benchmark_ops.elementwise_mul(input_adv, weight_adv)
    
    # If ASAN is disabled, execution will continue here silently showing the danger of memory corruption
    print(f"-> SUCCESS (Normal Build)! Output shape: {output_adv.shape}")
    print(f"-> Output first row: {output_adv[0].tolist()}")
    warning("WARNING: Out-of-bounds write occurred SILENTLY without warning or crash!")
    warning("This demonstrates how hidden memory bugs compromise safety without dynamic instrumentation.")

if __name__ == "__main__":
    run_security_benchmark()
