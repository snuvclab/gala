import pickle
import os
import cv2
import trimesh
import numpy as np
import argparse
import nvdiffrast.torch as dr
import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.io import load_obj
from tqdm import tqdm
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--mesh_dir", type=Path, default="data/rp/rp_george_posed_002", help="directory to the mesh")
parser.add_argument("--dataset_type", default='rp')
parser.add_argument("--use_opengl", action="store_true")
parser.add_argument("-d", "--device", default="cuda:0")
args = parser.parse_args()

data_dir = args.mesh_dir
dataset_type = args.dataset_type
device = args.device
use_opengl = args.use_opengl

res_upsample = 2048  # high resolution rendering for finding pixel-face correspondences
num_views = 36

if dataset_type == 'rp':
    mesh_name = data_dir.stem + '_100k'
elif dataset_type == 'tada':
    mesh_name = 'mesh'
else:
    mesh_name = f'{data_dir.stem}'

verts, faces, aux = load_obj(f'{data_dir}/{mesh_name}_norm.obj', device=device)
indices = faces.verts_idx.int()
face_mask_visible_stack = np.zeros((len(indices), num_views))
face_mask_segm_stack = np.zeros((len(indices), num_views))

glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

view_idx = 0
for x in range(150, 211, 30):
    for y in range(0, 360, 30):
        segm_path = f'{data_dir}/render/segms/{x:03d}_{y:03d}.png'
        if not os.path.exists(segm_path):  # SAM failed
            continue
        segm_mask = cv2.imread(segm_path) // 255
        segm_mask = torch.tensor(segm_mask[..., 0:1], dtype=torch.float32, device=device)

        r_xpi = R.from_euler('x', x, degrees=True).as_matrix()
        r_ypi = R.from_euler('y', y, degrees=True).as_matrix()
        r = r_xpi @ r_ypi

        transformed_vertices = (torch.tensor(r, device=device).float() @ (verts.T)).T
        transformed_vertices_h = torch.cat([transformed_vertices, torch.ones_like(transformed_vertices[:, 0:1])], axis=1)
        rast, rast_out_db = dr.rasterize(glctx, transformed_vertices_h[None], faces.verts_idx.int(), 
                                            resolution=np.array([res_upsample, res_upsample]))

        uv = 0.5 * (transformed_vertices[:, :2].contiguous() + 1) 
        texc, _ = dr.interpolate(uv[None, ...], rast, indices)
        segm_mask_upsample = (dr.texture(segm_mask[None, ...], texc, filter_mode='linear') >= 0.5)[0, ..., 0].cpu().numpy()

        face_ids = rast[0, ..., -1].cpu().numpy().astype(np.int32)
        face_ids_visible = face_ids[face_ids != 0] - 1
        face_mask_visible_stack[face_ids_visible, view_idx] = 1

        face_ids_segm = face_ids[segm_mask_upsample] - 1  # face ids segmented as target
        face_mask_segm_stack[face_ids_segm, view_idx] = 1

        view_idx += 1
        
face_mask_vote = ((face_mask_segm_stack.sum(axis=-1) / (face_mask_visible_stack.sum(axis=-1) + 1e-8))) >= 0.5
np.save(f'{data_dir}/faces_segms.npy', face_mask_vote)

mesh_segms_obj = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=indices.cpu().numpy())
# leave only obj mesh
mesh_segms_obj.update_faces(face_mask_vote)

# clean up isolated components
# reference: https://github.com/mikedh/trimesh/issues/895
cc = trimesh.graph.connected_components(mesh_segms_obj.face_adjacency, min_len=25)
mask_cc = np.zeros(len(mesh_segms_obj.faces), dtype=bool)
mask_cc[np.concatenate(cc)] = True
mesh_segms_obj.update_faces(mask_cc)

mesh_segms_obj.export(f'{data_dir}/segms_obj.obj')

# vertex_colors = np.ones(verts.shape)
# indices = indices.cpu().numpy()
# for face in enumerate(indices[face_mask_vote]): 
#     for vertex_idx in face:
#         vertex_colors[vertex_idx] = [0, 255, 0]  # obj as green

# import open3d as o3d
# mesh_o3d = o3d.geometry.TriangleMesh()
# mesh_o3d.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
# mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
# mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# o3d.io.write_triangle_mesh(f'{data_dir}/segms.obj', mesh_o3d, write_ascii=True)
# o3d.io.write_triangle_mesh(f'{data_dir}/segms_hum.obj', mesh_o3d, write_ascii=True)
# o3d.io.write_triangle_mesh(f'{data_dir}/segms_obj.obj', mesh_o3d, write_ascii=True)

# mesh_segms_hum = trimesh.load(f'{data_dir}/segms_hum.obj', maintain_order=True, process=False)
# mesh_segms_hum.update_faces(1 - face_mask_vote)
# mesh_segms_hum.export(f'{data_dir}/segms_hum.obj')

# mesh_segms_obj = trimesh.load(f'{data_dir}/segms_obj.obj', maintain_order=True, process=False)
# mesh_segms_obj.update_faces(face_mask_vote)
# mesh_segms_obj.export(f'{data_dir}/segms_obj.obj')