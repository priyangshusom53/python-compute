

from abc import ABC, abstractmethod

class Renderer(ABC):

   @abstractmethod
   def render(self,scene):
      pass

   @abstractmethod
   def render_screen_extent(self,scene,extent):
      pass