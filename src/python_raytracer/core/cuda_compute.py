import os
import pathlib
import logging
import time
import subprocess

import numpy as np
import cupy as cp


class CudaKernel:

   __slots__ = ['logger',
                'config',
                'cmake_dir',
                'srcf',
                'kernel_name',
                'last_modified',
                '_kernel',
                '_module']

   def _compile_cuda_src(self,debug:bool):
      srcf_path = self.srcf
      cmake_dir = self.cmake_dir
      build_dir = pathlib.Path(self.cmake_dir / "build")
      try:
         subprocess.run(
            ["cmake", "-S", str(cmake_dir), "-B", str(build_dir)],
            check=True
         )
         config = "Debug" if debug else "Release"
         subprocess.run(
            ["cmake", "--build", str(build_dir), "--config",config],
            check=True
         )
         self.logger.info(f"CUDA src compiled with config= {config}")
      except BaseException as e:
         self.logger.error(f"Failed to compile CUDA src error= {e}")
      
   def _is_src_modified(self):
      srcf = self.srcf
      try:
         current_modified = os.stat(srcf).st_mtime
         if current_modified != self.last_modified:
            self.logger.warning(f"CUDA srcf={self.srcf} with kernel={self.kernel_name} modified")
            self.last_modified = current_modified            
            return True
         else:
            return False
      except BaseException as e:
         self.logger.error(e)
         return False
      
   def _load_module(self):
      # Fully release previous module
      if self._module is not None:
         del self._kernel
         del self._module
         cp.cuda.runtime.deviceSynchronize()
         mempool = cp.get_default_memory_pool()
         pinned_mempool = cp.get_default_pinned_memory_pool()
         mempool.free_all_blocks()
         pinned_mempool.free_all_blocks()
         del mempool
         del pinned_mempool

      ptx_name = f"{self.srcf.stem}.ptx"
      ptx_path = (
         self.cmake_dir / f"build/ptx/{self.config}/{ptx_name}"
      ).resolve()

      self._module = cp.RawModule(
         path=str(ptx_path),
         options=(f"-DRELOAD_{int(self.last_modified)}",),
      )
      self._kernel = self._module.get_function(self.kernel_name)
   
   def launch(self, blocks:tuple, threads:tuple, args:list, debug: bool):
      if self._is_src_modified():
         self.logger.info("CUDA source changed -> recompiling")
         self._compile_cuda_src(debug)
         self._load_module()

      try:
         self.logger.info(f"Launching CUDA kernel: {self.kernel_name}")
         self._kernel(blocks, threads, tuple(args))
         cp.cuda.runtime.deviceSynchronize()
      except cp.cuda.runtime.CUDARuntimeError as e:
         self.logger.error(f"CUDA ERROR: {e}")
         raise

   def __init__(self, cmake_dir:str, srcf:str, name:str, debug:bool):
      self.logger = logging.getLogger(__name__)
      self.config = "Debug" if debug else "Release"
      self.cmake_dir = pathlib.Path(cmake_dir).resolve()
      self.srcf = pathlib.Path(srcf).resolve()
      self.kernel_name = name
      self._compile_cuda_src(debug)
      # time when kernel with name={name} in file={srcf} was last modified
      self.last_modified = os.stat(self.srcf).st_mtime
      self._module = None
      self._kernel = None

      self._compile_cuda_src(debug)
      self._load_module()