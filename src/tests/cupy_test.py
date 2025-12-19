import cupy as cp
print(cp.__version__)

from PIL import Image
import numpy as np

import python_raytracer.cudaloader.loader as cudaloader


# ======================
# 1. Define the kernel
# ======================
kernel = r'''

#include "common.cu"

extern "C" __global__
void radial_gradient(float* output, int width, int height) {
    constexpr float PI = 3.14159f;                         
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float sigma_x = 1.0f;
    const float sigma_y = 0.1f;                          
   
    int cx = calc_half(width);
    int cy = calc_half(height);

    float dx = (float)(x - cx)/width;
    float dy = (float)(y - cy)/height;
      
    float nx = dx / sigma_x;
    float ny = dy / sigma_y;

    // Elliptical radius
    float r = sqrtf(nx * nx + ny * ny);

    // Example gradient
    float g = expf(-(10* dx * dx + 100 * dy * dy));
    float b = 1-g;
                             
    int idx = (y * width + x) * 4;                         
    output[idx + 0] = 0.0f;     // Red
    output[idx + 1] = g;          // Green
    output[idx + 2] = b;          // Blue
    output[idx + 3] = 1.0f;          // Alpha
}
'''
kernel = cudaloader.cuda_preprocessor(["../python_raytracer/pathtracer/cuda"], kernel)
print(kernel)
mod = cp.RawModule(code=kernel)
radial_kernel = mod.get_function("radial_gradient")

# radial_kernel = cp.RawKernel(r'''
# extern "C" __global__
# void radial_gradient(float* output, int width, int height) {
#     constexpr float PI = 3.14159f;                         
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#     int y = blockIdx.y * blockDim.y + threadIdx.y;

#     if (x >= width || y >= height) return;

#     const float sigma_x = 1.0f;
#     const float sigma_y = 0.1f;                          
   
#     int cx = width / 2;
#     int cy = height / 2;

#     float dx = (float)(x - cx)/width;
#     float dy = (float)(y - cy)/height;
      
#     float nx = dx / sigma_x;
#     float ny = dy / sigma_y;

#     // Elliptical radius
#     float r = sqrtf(nx * nx + ny * ny);

#     // Example gradient
#     float g = expf(-(10* dx * dx + 100 * dy * dy));
#     float b = 1-g;
                             
#     int idx = (y * width + x) * 4;                         
#     output[idx + 0] = 0.0f;     // Red
#     output[idx + 1] = g;          // Green
#     output[idx + 2] = b;          // Blue
#     output[idx + 3] = 1.0f;          // Alpha
# }
# ''', 'radial_gradient')

# ======================
# 2. Set up image size
# ======================
width = 800
height = 600

# ======================
# 3. Allocate GPU memory
# ======================
# Create empty array on GPU: (height, width, 4) for RGBA
output_gpu = cp.empty((height, width, 4), dtype=cp.float32)

# ======================
# 4. Launch kernel
# ======================
block_size = (16, 16)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1]
)

radial_kernel(grid_size, block_size, (output_gpu, width, height))

# ======================
# 5. Get result back to CPU and save
# ======================

# Step 5a: Convert GPU array → CPU NumPy array
output_cpu = cp.asnumpy(output_gpu)  # Shape: (600, 800, 4), dtype: float32 [0.0-1.0]

# Step 5b: Convert float32 [0.0, 1.0] → uint8 [0, 255]
output_uint8 = (output_cpu * 255).astype(np.uint8)

# Step 5c: Create PIL Image
img = Image.fromarray(output_uint8, mode='RGBA')

# Step 5d: Save to file
img.save("test_outputs/radial_gradient.png")

print("✅ Radial gradient saved as 'radial_gradient.png'")
