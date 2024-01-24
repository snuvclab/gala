import open3d as o3d
import numpy as np
import argparse
import os
import pickle
from pathlib import Path

def find_startswith(file_content, target):
    # Find the line that starts with 'mtllib'
    for line in file_content.split('\n'):
        if line.startswith(target):
            return line

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--mesh_dir", default="data/rp/rp_christopher_posed_008", help="mesh obj")
parser.add_argument("--dataset_type", default='rp')
parser.add_argument("--norm_with_smplx", action='store_true')
args = parser.parse_args()

mesh_dir = Path(args.mesh_dir)
dataset_type = args.dataset_type

if dataset_type == 'rp':
    mesh_name = mesh_dir.stem + '_100k'
elif dataset_type == 'tada':
    mesh_name = 'mesh'
else:
    mesh_name = f'{mesh_dir.stem}'
    
mesh_file = mesh_dir / f'{mesh_name}.obj'

mesh = o3d.io.read_triangle_mesh(str(mesh_file))
vertices = np.asarray(mesh.vertices)

# if not yet normlized with smplx scale and translation
if args.norm_with_smplx:
    smplx_params = np.load(mesh_dir / 'smplx_param.pkl', allow_pickle=True)
    if dataset_type == 'thuman':  # dict keys are slightly different for given thuman smplx param files
        transl = smplx_params['translation']
        scale = smplx_params['scale']
    else:
        transl = smplx_params['global_body_translation']
        scale = smplx_params['body_scale']

    vertices = (vertices - transl) / scale

vertices[:, 1] += 0.4 # shift all ys by 0.4

# NOTE: trimesh export does not allow the different number of vertices and uv coordinates, which leads to artifacts.
mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d.io.write_triangle_mesh(str(mesh_dir / f'{mesh_name}_norm.obj'), mesh)

os.remove(f'{mesh_dir}/{mesh_name}_norm.mtl')

# update normalized obj file to refer to original mtl file
with open(mesh_dir / f'{mesh_name}.obj', 'r') as obj_orig:
    mtllib_line_orig = find_startswith(obj_orig.read(), 'mtllib')

mtl_file = mtllib_line_orig.split()[-1]  # ex. "mtllib material0.mtl"
with open(mesh_dir / mtl_file, 'r') as mtl:
    material_name = find_startswith(mtl.read(), 'newmtl').split()[-1]

mtl_lines = mtllib_line_orig + f'\nusemtl {material_name}'

with open(mesh_dir / f'{mesh_name}_norm.obj', 'r') as obj_norm:
    obj_norm_content = obj_norm.read()
    mtllib_line_norm = find_startswith(obj_norm_content, 'mtllib')

obj_norm_content_updated = obj_norm_content.replace(mtllib_line_norm, mtl_lines)

with open(mesh_dir / f'{mesh_name}_norm.obj', 'w') as obj_norm:
    obj_norm.write(obj_norm_content_updated)
