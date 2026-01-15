import numpy as np
import math
from PIL import Image
from python_raytracer.core.geometry.transformation import Transform

n = 1e-10
f = 1000
fov = 90
# <Transfom to perspective view volume>
persp = Transform(
   matrix=np.array([[1, 0, 0,              0],
                     [0, 1, 0,              0],
                     [0, 0, f/(f-n), -f*n/(f-n)],
                     [0, 0, 1,              0]]))
# <Scale with fov>
inv_tan_ang = 1 / math.tan(math.radians(fov) / 2)
scale = Transform.scale(inv_tan_ang, inv_tan_ang, 1)
t = Transform(matrix=np.linalg.matmul(scale.matrix, persp.matrix))
print(persp.matrix)
print(scale.matrix)
print(t.matrix)
print(t.inverse_matrix)
pix_x = np.indices((1920,))[0]
pix_y = np.indices((1080,))[0]
pix_grid_x, pix_grid_y = np.meshgrid(pix_x,pix_y)
pix_grid = np.stack((pix_grid_x,pix_grid_y),axis=-1)
print(pix_grid.shape)
# <Transform raster coordinates to camera space>
zeros = np.zeros((1080,1920,1))
ones = np.ones((1080,1920,1))
pix_grid_cam = np.concat((pix_grid,zeros,ones),axis=-1)
print(pix_grid_cam)