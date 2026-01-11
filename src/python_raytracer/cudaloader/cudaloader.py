
import subprocess
from pathlib import Path
import argparse

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

def load(path:str,config:str):
    src_dir_str = path
    src_dir = Path(src_dir_str).resolve()
    build_dir_str = f"{src_dir_str}/build"
    build_dir = Path(build_dir_str).resolve()
    build_dir.mkdir(exist_ok=True)

    subprocess.run(
    ["cmake", "-S", str(src_dir), "-B", str(build_dir)],
    check=True
    )

    subprocess.run(
    ["cmake", "--build", str(build_dir), "--config",config],
    check=True
    )


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="run CMake to compile cuda src")

   parser.add_argument("-path","--path",type=str,required=True, help="path to cuda project src tree, relative path is relative to current working dir or absolute path",dest="path")

   parser.add_argument("-config","--config",type=str,default="Release",choices=["Debug","Release"],help="build type",dest="config")

   args = parser.parse_args()

   load(args.path, args.config)