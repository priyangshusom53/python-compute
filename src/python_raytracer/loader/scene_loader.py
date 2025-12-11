
from abc import ABC, abstractmethod

class Loader(ABC):

   @abstractmethod
   def load(self,path:str):
      pass

