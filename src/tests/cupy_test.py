import cupy as cp
print(cp.__version__)

import cupy as cp
from PIL import Image
import numpy as np

# ======================
# 1. Define the kernel
# ======================
radial_kernel = cp.RawKernel(r'''
extern "C" __global__
void radial_gradient(float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float dx = x - cx;
    float dy = y - cy;
    float dist = sqrtf(dx*dx + dy*dy);
    float max_dist = sqrtf(cx*cx + cy*cy);

    float intensity = 1.0f - fminf(dist / max_dist, 1.0f);

    int idx = (y * width + x) * 4;
    output[idx + 0] = intensity;     // Red
    output[idx + 1] = 0.0f;          // Green
    output[idx + 2] = 0.0f;          // Blue
    output[idx + 3] = 1.0f;          // Alpha
}
''', 'radial_gradient')

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
img.save("radial_gradient.png")

print("✅ Radial gradient saved as 'radial_gradient.png'")
