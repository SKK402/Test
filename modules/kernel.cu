// Professional ML Systems / CUDA Implementation
// Purpose: High-performance CUDA kernel for elementwise float32 tensor multiplication.
// Compatible with PyTorch custom C++/CUDA extensions.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

/**
 * CUDA Kernel for elementwise multiplication of two float32 tensors.
 *
 * @param input  Pointer to the input tensor memory.
 * @param weight Pointer to the weight tensor memory.
 * @param output Pointer to the output tensor memory.
 * @param numel  Total number of elements in the tensors (1D flat size).
 */
__global__ void elementwise_mul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t numel) {
    
    /*
     * CUDA Thread Indexing Variables Explanation:
     * -------------------------------------------
     * 1. threadIdx.x:
     *    - Represents the unique 1-dimensional index of the current thread within its thread block.
     *    - Range: [0, blockDim.x - 1].
     *
     * 2. blockIdx.x:
     *    - Represents the unique 1-dimensional index of the current thread block within the execution grid.
     *    - Range: [0, gridDim.x - 1].
     *
     * 3. blockDim.x:
     *    - Specifies the size (number of threads) of each thread block along the X-dimension.
     *    - Typically set to powers of 2 (e.g., 256, 512, 1024) to optimize warp scheduling.
     *
     * 4. gridDim.x:
     *    - Specifies the total number of thread blocks in the execution grid along the X-dimension.
     *    - Usually calculated dynamically based on the total elements and block size: (numel + blockDim.x - 1) / blockDim.x.
     */

    // Calculate the unique global 1-dimensional index of this thread across the entire grid.
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Bounds Checking:
     * ----------------
     * Because block size is static (e.g., 256 threads) and dataset sizes are dynamic,
     * the total threads launched (gridDim.x * blockDim.x) may exceed the actual size of the data (`numel`).
     * The condition `idx < numel` ensures that threads beyond the size of the array do not
     * read or write to invalid memory, which would trigger Segmentation Faults, CUDA illegal memory access,
     * or data corruption.
     */
    if (idx < numel) {
        output[idx] = input[idx] * weight[idx];
    }
}

/**
 * Host launcher function exposed to PyTorch / C++ bindings (custom_operator.cpp).
 * Assumes inputs and outputs are pre-allocated on the GPU device.
 *
 * @param input  Device pointer to float32 input.
 * @param weight Device pointer to float32 weight.
 * @param output Device pointer to float32 output.
 * @param numel  Number of elements.
 * @param stream CUDA stream to run the execution asynchronously.
 */
extern "C" void launch_elementwise_mul_kernel(
    const float* input,
    const float* weight,
    float* output,
    int64_t numel,
    cudaStream_t stream) {
    
    // Quick exit for zero-element or empty tensors
    if (numel <= 0) {
        return;
    }

    // Configure thread block size (256 threads is generally optimal for simple elementwise memory bound operations)
    const int block_size = 256;
    
    // Calculate the number of blocks needed in the grid.
    // Uses integer division rounding up to ensure we cover the entire array.
    const int64_t grid_size = (numel + block_size - 1) / block_size;

    // Launch the asynchronous CUDA kernel on the requested stream.
    elementwise_mul_kernel<<<grid_size, block_size, 0, stream>>>(input, weight, output, numel);
}
