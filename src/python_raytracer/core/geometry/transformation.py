

import numpy as np

class Transform:

   __slots__ = ["_matrix", "_inverse_matrix"]

   def __init__(self, matrix:np.ndarray=None, inverse_matrix:np.ndarray=None):
      if matrix is None:
         self._matrix = np.identity(4, dtype=np.float32)
         self._inverse_matrix = np.identity(4, dtype=np.float32)
      elif inverse_matrix is not None:
         if inverse_matrix.shape != (4,4):
            raise ValueError("Inverse matrix must be of shape (4,4).")
         self._inverse_matrix = inverse_matrix.astype(np.float32)
         self._matrix = matrix.astype(np.float32)
      else:
         if matrix.shape != (4,4):
            raise ValueError("Matrix must be of shape (4,4).")
         self._matrix = matrix.astype(np.float32)
         self._inverse_matrix = np.linalg.inv(self._matrix).astype(np.float32)

   def identity(self):
      self._matrix = np.identity(4, dtype=np.float32)
      self._inverse_matrix = np.identity(4, dtype=np.float32)

   @property
   def matrix(self) -> np.ndarray:
      return self._matrix
   
   @property
   def inverse_matrix(self) -> np.ndarray:
      return self._inverse_matrix

   @staticmethod
   def inverse(transform:'Transform') -> 'Transform':
      transform._matrix, transform._inverse_matrix = transform._inverse_matrix, transform._matrix
      return transform

   @staticmethod
   def transpose(transform:'Transform') -> 'Transform':
      transform._matrix = transform._matrix.T
      transform._inverse_matrix = transform._inverse_matrix.T
      return transform
   
   @staticmethod
   def translate(tx:float, ty:float, tz:float) -> 'Transform':
      matrix = np.array([[1, 0, 0, tx],
                         [0, 1, 0, ty],
                         [0, 0, 1, tz],
                         [0, 0, 0, 1]], dtype=np.float32)
      inverse_matrix = np.array([[1, 0, 0, -tx],
                                 [0, 1, 0, -ty],
                                 [0, 0, 1, -tz],
                                 [0, 0, 0, 1]], dtype=np.float32)
      return Transform(matrix, inverse_matrix)
   
   @staticmethod
   def scale(sx:float, sy:float, sz:float) -> 'Transform':
      matrix = np.array([[sx, 0, 0, 0],
                         [0, sy, 0, 0],
                         [0, 0, sz, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
      inverse_matrix = np.array([[1/sx if sx != 0 else 0, 0, 0, 0],
                                 [0, 1/sy if sy != 0 else 0, 0, 0],
                                 [0, 0, 1/sz if sz != 0 else 0, 0],
                                 [0, 0, 0, 1]], dtype=np.float32) 
      return Transform(matrix, inverse_matrix)
   
   
   
