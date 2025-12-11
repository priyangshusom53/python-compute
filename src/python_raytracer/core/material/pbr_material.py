

from dataclasses import (dataclass,field)
from typing import (Optional, Union, Literal)
import numpy as np
from PIL.Image import Image as PILImage

def _default_base_color() -> list[float]:
    return [1.0, 1.0, 1.0, 1.0]

@dataclass
class PBRMaterial:
   name:str
   base_color_factor:Optional[list[float]] = field(default_factory=_default_base_color)
   metallic_factor:Optional[float] = 0.0
   roughness_factor:Optional[float] = 1.0

   # PIL RGBA images if exists
   base_color_texture:Optional[PILImage] = None
   normal_texture:Optional[PILImage] = None
   # The metalness values are sampled from the B channel. The 
   # roughness values are sampled from the G channel.
   metallic_roughness_texture:Optional[PILImage] = None

TextureType = Literal['base_color', 'normal', 'metallic_roughness']
def texture_to_np_array(texture:PILImage, type: TextureType) -> np.ndarray:
   """Convert a PIL Image texture to a numpy array."""
   
   if not isinstance(texture, PILImage) or texture is None:
      if type == 'base_color':
         # Default normal pointing up
         np_array = np.ones((1, 1, 4), dtype=np.float32)
         return np_array
      elif type == 'metallic_roughness':
         # Default metallic=0, roughness=1
         np_array = np.array([[[0.0, 1.0, 0.0, 1.0]]], dtype=np.float32)
         return np_array
      
   
   if texture.mode == 'RGBA':
      width, height = texture.size
      np_array = np.array(texture).reshape((height, width, 4)).astype(np.float32) / 255.0
      return np_array
   if texture.mode == 'RGB':
      width, height = texture.size
      np_array = np.array(texture).reshape((height, width, 3)).astype(np.float32) / 255.0
      return np_array
   else:
      raise ValueError(f"Unsupported texture mode: {texture.mode}. Only 'RGB' and 'RGBA' are supported.")
   
