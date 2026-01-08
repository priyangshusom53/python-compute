
import subprocess
from pathlib import Path

src_dir_str = "src/python_raytracer/pathtracer/cuda"
src_dir = Path(src_dir_str).resolve()
build_dir_str = f"{src_dir_str}/build"
build_dir = Path(build_dir_str).resolve()
build_dir.mkdir(exist_ok=True)

subprocess.run(
    ["cmake", "-S", str(src_dir), "-B", str(build_dir)],
    check=True
)

subprocess.run(
    ["cmake", "--build", str(build_dir)],
    check=True
)
