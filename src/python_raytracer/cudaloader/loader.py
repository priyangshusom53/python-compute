
from typing import List
from pathlib import Path
import re

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')


def find_include_file(include_name:str, search_paths:List[Path]) -> Path:
   print(include_name)
   print(search_paths)

   include_path:Path = None
   for path in search_paths:
      matches = list(path.glob(include_name))
      
      if matches:
         include_path = matches[0]
         return include_path
   raise FileNotFoundError(f"Include file not found: {include_name}")



def inline_includes(
    file_path: Path,
    already_included:set=None,
    include_paths:List[Path]=[]
) -> str:
   """
   Recursively inline local #include "file" directives.
   """
   if already_included is None:
      already_included = set()

   file_path = file_path.resolve()
   print(file_path)
   # Prevent infinite include loops
   if file_path in already_included:
      print(f"\nSkipping already included: {file_path.name}")
      return

   # Mark this file as included
   already_included.add(file_path) 

   output_lines = []

   with file_path.open("r", encoding="utf-8") as f:

      # Add the directory of the current file to the search paths
      include_paths.append(file_path.parent)
      for line in f:
         match = INCLUDE_RE.match(line)
         if match:
               include_name = match.group(1)
               include_path = find_include_file(include_name, include_paths).resolve()

               if include_path in already_included:
                  continue

               if not include_path.exists():
                  raise FileNotFoundError(
                     f"Include file not found: {include_path}"
                  )

               output_lines.append(
                  f"\n// BEGIN INLINE {include_name}\n"
               )
               output_lines.append(
                  inline_includes(include_path, already_included)
               )
               output_lines.append(
                  f"\n// END INLINE {include_name}\n"
               )
         else:
               output_lines.append(line)

      # Remove the directory after processing
      include_paths.remove(file_path.parent)

   return "".join(output_lines)



def preprocess_cuda(paths:list[str], kernel:str)-> str:

   include_paths:list[Path] = []

   for p in paths:
      include_paths.append(Path(p).resolve())

   print(include_paths)

   # set for already included files
   cuda_includes = set()

   kernel_includes = kernel.splitlines()
   for line in kernel_includes:
      match = INCLUDE_RE.match(line)
      if match:
            include_name = match.group(1)

            include_path = find_include_file(include_name, include_paths).resolve()

            # continue if file with include_path is 
            # already included
            if include_path in cuda_includes:
               continue

            if not include_path.exists():
               raise FileNotFoundError(
                     f"Include file not found: {include_path}"
               )
            
            cuda_module:str = f"\n// BEGIN INLINE {include_name}\n"

            cuda_module+=inline_includes(include_path, cuda_includes)

            cuda_module += f"\n// END INLINE {include_name}\n"

            kernel = kernel.replace(line, cuda_module)

   return kernel
   
