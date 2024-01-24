# Mesh rendering code
# Mesh is normalized such that its smplx scale is 1 and smplx transl is 0.

import os
import argparse
import cv2
import torch
import pickle
import nvdiffrast.torch as dr
import numpy as np
import ipdb

from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--mesh_dir", default="data/rp/rp_christopher_posed_008", help="folder directory where mesh obj file exists")
parser.add_argument("-d", "--device", default="cuda:0")
parser.add_argument("--use_opengl", action="store_true")
args = parser.parse_args()

data_dir = args.mesh_dir
out_dir = data_dir + '/render'
device = args.device
use_opengl = args.use_opengl

res = 512

mesh_names = sorted(os.listdir(data_dir))
mesh_names = [m for m in mesh_names if "_norm.obj" in m]

print(mesh_names)
for mesh_name in tqdm(mesh_names):

    os.makedirs(f'{out_dir}/images', exist_ok=True)

    verts, faces, aux = load_obj(f'{data_dir}/{mesh_name}', device=device)
    uv = aux.verts_uvs
    uv[:, 1] = 1 - uv[:, 1]  # flip v coordinate to match opengl convention
    uv_idx = faces.textures_idx.to(torch.int32)
    tex = list(aux.texture_images.values())[0].to(device)  # dealing with arbitrary key name for texture

    # pix_to_faces = {}
    # ipdb.set_trace()

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
    # body
    for x in range(150, 211, 30):
        for y in tqdm(range(0, 360, 30)):
            r_xpi = R.from_euler('x', x, degrees=True).as_matrix()
            r_ypi = R.from_euler('y', y, degrees=True).as_matrix()
            r = r_xpi @ r_ypi

            transformed_vertices = (torch.tensor(r, device=device).float() @ (verts.T)).T
            transformed_vertices_h = torch.cat([transformed_vertices, torch.ones_like(transformed_vertices[:, 0:1])], axis=1)
            
            rast, rast_out_db = dr.rasterize(glctx, transformed_vertices_h[None], faces.verts_idx.int(), 
                                             resolution=np.array([res, res]))
            
            texc, _ = dr.interpolate(uv[None, ...], rast, uv_idx)
            color = dr.texture(tex[None, ...], texc, filter_mode='linear')
            color = color * torch.clamp(rast[..., -1:], 0, 1)  # Mask out background

            transforms.ToPILImage()(color[0].permute(2, 0, 1)).save(f'{out_dir}/images/{x:03d}_{y:03d}.png')

