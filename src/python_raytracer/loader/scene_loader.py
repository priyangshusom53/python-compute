
from abc import ABC, abstractmethod

class FLoader(ABC):

   @abstractmethod
   def load(self,path:str):
      pass

