
from python_raytracer.cudaloader.loader import preprocess_cuda

from pathlib import Path
import os

# Get the current working directory
print(f"Current Working Directory: {Path.cwd()}")

include_paths = ["../python_raytracer/pathtracer/cuda"]

if __name__ == "__main__":
   kernel_code = r"""
   #include "common.cu"
   __global__ void my_kernel() {
      // Kernel code here
   }
   """

   processed_code = preprocess_cuda(include_paths, kernel_code)
   print(processed_code)