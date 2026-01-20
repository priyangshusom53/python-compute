
import logging 
import sys

import coloredlogs

def log_config():
   coloredlogs.install(level='DEBUG')
   logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      handlers=[
            logging.StreamHandler(sys.stdout)
        ])