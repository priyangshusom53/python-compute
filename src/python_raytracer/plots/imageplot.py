
import logging
from pathlib import Path

import numpy as np
from PIL import Image

# mode is 'RGB' or 'RGBA'
def save_to_image(data:np.ndarray, dimx:int, dimy:int, mode:str, path:str, debug:bool):
   logger = logging.getLogger(__name__)
   if debug:
      assert data.shape == (dimx,dimy)
      assert data.dtype == np.uint8
   path_to_image = Path(path).resolve()
   img = Image.fromarray(data, mode=mode)
   img.save(str(path_to_image))
   logger.info(f"Image saved as ${path_to_image}")

