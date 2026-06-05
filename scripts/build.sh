#!/bin/bash
# Professional Build System Script for Custom PyTorch C++/CUDA Extensions
# Supports both Production (O3 Optimized) and AddressSanitizer (ASAN Debugging) modes.
# Patched for Linker RPath, CUDA Architectures, and ASAN Sanitizer suppression.
# Senior Build Systems Engineer Implementation.

set -e

# Coloured diagnostic output
info() { echo -e "\e[32m[INFO]\e[0m $1"; }
warning() { echo -e "\e[33m[WARNING]\e[0m $1"; }
error() { echo -e "\e[31m[ERROR]\e[0m $1"; exit 1; }

# Change to the directory of this script, then go to parent (project root)
cd "$(dirname "$0")/.."
info "Navigated to project workspace: $(pwd)"

# Detect if ASAN build is requested (usage: ASAN=1 ./scripts/build.sh)
ASAN_ENABLED=0
if [ "${ASAN}" = "1" ]; then
    ASAN_ENABLED=1
    info "AddressSanitizer (ASAN) Mode is ENABLED."
else
    info "Normal Production Mode is ENABLED. (Run 'ASAN=1 $0' for AddressSanitizer build)"
fi

# Clean up previous builds to avoid caching issues and state pollution
info "Cleaning up previous build artifacts..."
rm -rf build dist vulnerability_benchmark_ops.egg-info *.so modules/*.o

# Generate setup.py dynamically to prevent manual synchronization issues
info "Generating setup.py dynamically..."
cat << EOF > setup.py
import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# 1. Automatic GPU Architecture Target Detection
# Prevents warnings and builds specifically for the host's GPU capability or fallback options.
if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f"{major}.{minor}"
        print(f"[BUILD SYSTEM] Detected GPU Compute Capability: {major}.{minor}. Targeting code generation for sm_{major}{minor}.")
    else:
        # Fallback list covering standard modern architectures (Ampere, Ada Lovelace, Hopper)
        os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0;8.6;8.9;9.0"
        print("[BUILD SYSTEM] No active GPU detected during compilation. Targeting fallback architectures: 8.0;8.6;8.9;9.0")

# Setup the compiler argument containers
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17', '-march=native', '-fPIC'],
    'nvcc': ['-O3', '-std=c++17', '--use_fast_math', '--expt-relaxed-constexpr', '--allow-unsupported-compiler']
}

# 2. PyTorch Runtime Library RPath and Link Paths Injection
# Resolves: "ImportError: libc10.so: cannot open shared object file: No such file or directory"
# This embeds the absolute PyTorch dynamic library path inside the compiled .so header,
# allowing the library to be imported standalone without forcing a prior "import torch".
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
extra_link_args = [
    f'-Wl,-rpath,{torch_lib_dir}',
    f'-L{torch_lib_dir}',
    '-lc10',
    '-ltorch',
    '-ltorch_cpu',
    '-ltorch_python'
]

# Phase 2: Add AddressSanitizer (ASAN) configuration if requested
if "${ASAN_ENABLED}" == "1":
    # 1. Host Compiler Flags (CXXFLAGS)
    # -fsanitize=address: Enables instrumenting memory accesses to detect bounds errors and leaks
    # -fno-omit-frame-pointer: Keeps the frame pointer in register for precise stack traces
    # -O0: Disables all compiler optimizations to ensure variable alignment and clear control flow
    # -g: Generates complete DWARF debug symbols to translate memory addresses back to line numbers
    extra_compile_args['cxx'] = [
        '-fsanitize=address', 
        '-fno-omit-frame-pointer', 
        '-O0', 
        '-g', 
        '-std=c++17', 
        '-fPIC'
    ]

    # 2. CUDA Device and Host Compiler Flags (NVCC FLAGS)
    # -Xcompiler: Forwards subsequent comma-separated flags directly to the host C++ compiler (GCC)
    # -G: Generates device-side debug information and disables CUDA optimizations
    # -g: Generates host-side debug information
    extra_compile_args['nvcc'] = [
        '-Xcompiler', '-fsanitize=address,-fno-omit-frame-pointer',
        '-G',
        '-g',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        '--allow-unsupported-compiler'
    ]

    # 3. Linker Flags (LDFLAGS)
    # -fsanitize=address: Compiles/links in runtime ASAN hook libraries (libasan.so)
    extra_link_args.append('-fsanitize=address')

setup(
    name='vulnerability_benchmark_ops',
    ext_modules=[
        CUDAExtension(
            name='vulnerability_benchmark_ops',
            sources=[
                'modules/custom_operator.cpp',
                'modules/kernel.cu'
            ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
EOF

# Perform compilation in-place so Python can import the .so library directly from the root folder
info "Compiling custom CUDA extension in-place..."
python3 setup.py build_ext --inplace

# Clean up setup.py after build is done to maintain repository integrity
rm -f setup.py

info "Build complete!"
if [ "${ASAN_ENABLED}" = "1" ]; then
    warning "=========================================================================================="
    warning "ASAN WARNING: To run this compiled module, you MUST preload"
    warning "AddressSanitizer runtime library and suppress background leak detection noise."
    warning "Run using the following environment wrappers:"
    warning "  export LD_PRELOAD=\$(gcc -print-file-name=libasan.so)"
    warning "  export ASAN_OPTIONS=detect_leaks=0"
    warning "  python3 test_trigger.py"
    warning "=========================================================================================="
fi
