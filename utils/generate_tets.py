import open3d as o3d
import numpy as np
import os
import tetgen
import pyvista as pv

# create a cube
cube = o3d.geometry.TriangleMesh.create_box()
cube_vertices = np.asarray(cube.vertices)
cube_vertices -= 0.5
cube_vertices[...,2] = cube_vertices[...,2] * 0.4 # canonicalized humans take less space along z-axis
cube.vertices = o3d.utility.Vector3dVector(cube_vertices)
o3d.io.write_triangle_mesh("cube.obj", cube)

# convert it to tetrahedral grid
# https://github.com/huangyangyi/TeCH/blob/main/core/lib/tet_utils.py
mesh = pv.read(os.path.join("cube.obj"))
tet = tetgen.TetGen(mesh)
tet.make_manifold(verbose=True)
vertices, indices = tet.tetrahedralize( fixedvolume=1, 
                                        maxvolume=0.00000005, # resolution manually set
                                        regionattrib=1, 
                                        nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
shell = tet.grid.extract_surface()
os.makedirs('data/tets', exist_ok=True)
np.savez('data/tets/cube_tet_grid.npz', vertices=vertices, indices=indices)
print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))