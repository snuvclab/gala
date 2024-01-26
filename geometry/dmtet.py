# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
import open3d as o3d
import pickle

from render import mesh
from render import render
from geometry.dmtet_network import Decoder
from render import regularizer
import torch.nn.functional as F
import tinycudann as tcnn
from torch.cuda.amp import custom_bwd, custom_fwd
from pytorch3d.io import load_obj
from deformer.smplx import SMPLX
from deformer.lib import rotation_converter, helpers
from utils.tet_utils import compact_tets, sort_edges, batch_subdivide_volume
from dataset.dataset_mesh import DatasetMesh


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

# def laplacian_loss(mesh: Mesh):
#     """ Compute the Laplacian term as the mean squared Euclidean norm of the differential coordinates.

#     Args:
#         mesh (Mesh): Mesh used to build the differential coordinates.
#     """

#     L = mesh.laplacian
#     V = mesh.vertices
    
#     loss = L.mm(V)
#     loss = loss.norm(dim=1)**2
    
#     return loss.mean()

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
class SpecifyGradient2(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([   
                [-1, -1, -1, -1, -1, -1],     
                [ 1,  0,  2, -1, -1, -1],     
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],     
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long).cuda()   
        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda') #三角面片数量表，一共有16种类型的四面体，每一种四面体都对应要提取出的三角面片数量
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda') #基本的边，每一个数字是顶点索引，总共有12条边

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0 
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4) 
            occ_sum = torch.sum(occ_fx4, -1) 
            valid_tets = (occ_sum>0) & (occ_sum<4) 
            occ_sum = occ_sum[valid_tets] 
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2) 
            all_edges = self.sort_edges(all_edges) 
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)    
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1 
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1 
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda") 
            idx_map = mapping[idx_map] 
            interp_v = unique_edges[mask_edges] 
            
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3) 
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1) 
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True) 
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator 
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1) 
        idx_map = idx_map.reshape(-1,6)
        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda")) 
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1) 
        num_triangles = self.num_triangles_table[tetindex] 

        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)
       
        # Get global face index (static, does not depend on topology)
        # num_tets = tet_fx4.shape[0] 
        # tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets] 
        # face_gidx = torch.cat((
        #     tet_gidx[num_triangles == 1]*2,
        #     torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        # ), dim=0)

        # uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        return verts, faces
    
###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
 
        # self.decoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
        if self.FLAGS.enable_canonical:
            if self.FLAGS.high_res_tets:
                tets = np.load('data/tets/cube_tet_grid.npz')
            else:
                tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        else:
            tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))

        verts = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')

        self.verts, self.indices = verts, indices

        # compact tet grid
        self.verts = verts[verts[:, 2].abs() < 0.3]
        prev_ind = torch.where(verts[:, 2].abs() < 0.3)[0]
        curr_ind = torch.arange(0, len(prev_ind), device='cuda')
        indices = indices[torch.isin(indices, prev_ind).all(1)]
        replace = torch.vstack([prev_ind, curr_ind])
        mask = torch.isin(indices, replace[0, :])
        indices[mask] = replace[1, torch.searchsorted(replace[0, :], indices[mask])]
        self.indices = indices

        self.decoder = Decoder(multires=0 , AABB= self.getAABB(), mesh_scale= scale)

        print("tet grid min/max", torch.min(self.verts).item(), torch.max(self.verts).item())
        self.generate_edges()

        # SMPLX related initializations
        with open(self.FLAGS.smplx, 'rb') as f:
            codedict = pickle.load(f)
            param_dict = {}
            for key in codedict.keys():
                if isinstance(codedict[key], str):
                    param_dict[key] = codedict[key]
                else:
                    param_dict[key] = torch.from_numpy(codedict[key])
            self.smplx_params = param_dict
        self.set_smplx(self.smplx_params)
        
        # zoom ins
        self.enable_zoom_in = self.FLAGS.enable_zoom_in
        self.zoom_in_body = self.FLAGS.zoom_in_body
        # if self.zoom_in_body: self.set_zoom_body_cam()
        self.set_zoom_obj_cam()
        self.zoom_in_head_hands = self.FLAGS.zoom_in_head_hands
        self.zoom_in_params = None

        # seg masks
        self.dilate_seg_mask = self.FLAGS.dilate_seg_mask
        if self.dilate_seg_mask:
            self.dilation_size = 3
            self.dilate_seg = torch.nn.MaxPool2d(kernel_size=self.dilation_size, stride=1, padding=self.dilation_size // 2)

        # dmtets
        self.init_sdf = self.FLAGS.init_sdf
        if self.init_sdf: self.init_scence_vertices = None
        self.init_obj_smplx = self.FLAGS.init_obj_smplx
        self.is_obj_most_outer = self.FLAGS.is_obj_most_outer
        self.subdivision = self.FLAGS.subdivision
        self.num_subdivision = self.FLAGS.subdivision

        # controlnet
        self.use_op_control = self.FLAGS.enable_controlnet and self.FLAGS.controlnet_type == 'op'
        self.op_3d = None
        if self.use_op_control:
            self.op_3d_mapping = np.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 75, 82, 106, 125], dtype=np.int32)
            self.op_3d = self.gt_joints[self.op_3d_mapping] # get from gt SMPLX joints

        # canonicalization
        self.enable_canonical = FLAGS.enable_canonical
        self.use_canonical_sds = self.enable_canonical and FLAGS.use_canonical_sds
        self.multi_pose = self.enable_canonical and self.FLAGS.multi_pose
        if self.enable_canonical:
            d, h, w = 64, 256, 256
            grid = helpers.create_voxel_grid(d, h, w, device='cuda') # precompute lbs grid
            self.shape_pose_offsets_grid = helpers.query_weights_smpl(grid, self.smplx_verts_cano.cuda(), self.gt_smplx_offsets[0].cuda()).permute(0,2,1).reshape(1,-1,d,h,w)
            self.lbs_weights_grid = helpers.query_weights_smpl(grid, self.smplx_verts_cano.cuda(), self.smplx.lbs_weights.cuda()).permute(0,2,1).reshape(1,-1,d,h,w)
            if self.use_op_control:
                self.op_3d_cano = self.canonical_joints[self.op_3d_mapping] # get from canonical SMPLX joints

        # optimization
        self.take_turns = self.FLAGS.take_turns
        self.detach_hum = self.FLAGS.detach_hum
        self.detach_obj = self.FLAGS.detach_obj

        # hands
        self.sds_for_hands = self.FLAGS.sds_for_hands
        self.use_smplx_hands = self.FLAGS.use_smplx_hands

        self.gt_mesh = None
        self.gt_mesh_hum = None
        self.gt_mesh_obj = None
        self.gt_segs_posed = None

        # loss weights
        self.enable_lap = FLAGS.enable_lap
        self.enable_norm_smooth = FLAGS.enable_norm_smooth
        self.enable_min_surface = FLAGS.enable_min_surface and FLAGS.enable_canonical # use min surface loss only when canonicalizing
        self.enable_3d_segm = FLAGS.enable_3d_segm

        self.w_recon_segm = FLAGS.w_recon_segm
        self.w_recon_full = FLAGS.w_recon_full
        
        self.w_reg_lap = FLAGS.w_reg_lap
        self.w_reg_norm_smooth = FLAGS.w_reg_norm_smooth
        self.w_reg_min_surface = FLAGS.w_reg_min_surface
        self.w_reg_min_surface_init = FLAGS.w_reg_min_surface_init
        self.w_reg_3d = FLAGS.w_reg_3d

        self.w_sds_reg = FLAGS.w_sds_reg
        self.w_sds = FLAGS.w_sds

        self.tick_count = 0


    def set_smplx(self, smplx_params):
        smplx_config = {
            'topology_path': "deformer/data/SMPL_X_template_FLAME_uv.obj",
            'smplx_model_path': "deformer/data/SMPLX_NEUTRAL_2020.npz",
            'extra_joint_path': "deformer/data/smplx_extra_joints.yaml",
            'j14_regressor_path': "deformer/data/SMPLX_to_J14.pkl",
            'mano_ids_path': "deformer/data/MANO_SMPLX_vertex_ids.pkl",
            'flame_vertex_masks_path': "deformer/data/FLAME_masks.pkl",
            'flame_ids_path': "deformer/data/SMPL-X__FLAME_vertex_ids.npy",
            'n_shape': 10,
            'n_exp': 10
        }
        self.smplx_config = Struct(**smplx_config)
        self.op_3d_mapping = np.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 75, 82, 106, 125], dtype=np.int32)

        # gather bone transformations for poses that are to be used for good canonicalization
        self.refine_poses = []
        self.refine_tfs = []
        self.refine_op3ds = []

        # set canonical space
        self.smplx = SMPLX(self.smplx_config)
        pose = torch.zeros([55,3], dtype=torch.float32, ) # 55
        # legs
        angle = 15*np.pi/180.
        pose[1, 2] = angle
        pose[2, 2] = -angle
        # arms
        angle = 0*np.pi/180.
        pose[13, 2] = angle
        pose[14, 2] = -angle
        # waist
        angle = 0*np.pi/180.
        pose[3, 2] = angle
        pose[6, 2] = angle
        pose_euler = rotation_converter.batch_euler2axis(pose)
        pose = rotation_converter.batch_euler2matrix(pose)
        pose = pose[None,...]
        xyz_c, _, joints_c, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = pose, return_T=True, transl=torch.tensor([0, 0.4, 0],dtype=torch.float32, ))
        A_inv = A.squeeze(0).inverse()
        self.A_inv = A_inv
        self.refine_poses.append(pose_euler[1:22].reshape(1, -1))
        self.refine_tfs.append(torch.einsum('bnij,njk->bnik', A, self.A_inv).cuda())
        self.refine_op3ds.append(joints_c[0][self.op_3d_mapping])
        self.v_template = xyz_c.squeeze()
        self.canonical_joints = joints_c.squeeze()
        self.canonical_transform = T
        self.canonical_offsets = shape_offsets + pose_offsets
        ##--- load vertex mask
        with open(self.smplx_config.mano_ids_path, 'rb') as f:
            hand_idx = pickle.load(f)
        flame_idx = np.load(self.smplx_config.flame_ids_path)
        with open(self.smplx_config.flame_vertex_masks_path, 'rb') as f:
            flame_vertex_mask = pickle.load(f, encoding='latin1')
        # verts = torch.nn.Parameter(self.v_template, requires_grad=True)
        exclude_idx = []
        exclude_idx += list(hand_idx['left_hand'])
        exclude_idx += list(hand_idx['right_hand'])
        exclude_idx += list(flame_vertex_mask['face'])
        exclude_idx += list(flame_vertex_mask['left_eyeball'])
        exclude_idx += list(flame_vertex_mask['right_eyeball'])
        exclude_idx += list(flame_vertex_mask['left_ear'])
        exclude_idx += list(flame_vertex_mask['right_ear'])
        all_idx = range(xyz_c.shape[1])
        face_idx = list(flame_vertex_mask['face'])
        body_idx = [i for i in all_idx if i not in face_idx]

        self.part_idx_dict = {
            'face': flame_vertex_mask['face'],
            'left_hand': hand_idx['left_hand'],
            'right_hand': hand_idx['right_hand'], 
            'exclude': exclude_idx,
            'body': body_idx
        }
        ## smplx topology
        _, faces, _ = load_obj(self.smplx_config.topology_path)
        self.faces = faces.verts_idx[None,...]

        self.smplx_verts_cano = self.v_template
        self.smplx_faces = self.smplx.faces_tensor
        

        # GT posed smplx
        beta = smplx_params['betas'].squeeze()[:10]
        body_pose = smplx_params['body_pose'] if smplx_params['body_pose'].shape[1] == 63 else smplx_params['body_pose'][:, 3:]
        full_pose = torch.cat([smplx_params['global_orient'], body_pose,
                            smplx_params['jaw_pose'], smplx_params['leye_pose'], smplx_params['reye_pose'],  
                            smplx_params['left_hand_pose'][:, :6], smplx_params['right_hand_pose'][:, :6]], dim=1)
        exp = smplx_params['expression'].squeeze()[:10]
        xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx(full_pose=full_pose[None], shape_params=beta[None], return_T=True, 
                                                                  transl=torch.tensor([0, 0.4, 0],dtype=torch.float32)[None],
                                                                  expression_params=exp[None], axis_pca=True)
        self.gt_beta = beta
        self.gt_full_pose = full_pose
        self.canonical_full_pose = torch.zeros_like(full_pose)
        self.gt_exp = exp
        self.gt_smplx_verts = xyz.squeeze()
        self.gt_joints = joints.squeeze()
        self.gt_A = A
        self.gt_smplx_tfs = torch.einsum('bnij,njk->bnik', self.gt_A, self.A_inv).cuda()
        self.refine_poses.append(self.gt_full_pose[:, 3:66])
        self.refine_tfs.append(self.gt_smplx_tfs)
        self.refine_op3ds.append(joints[0][self.op_3d_mapping])
        self.gt_smplx_offsets = shape_offsets + pose_offsets
        self.gt_smplx_mesh = mesh.Mesh(self.gt_smplx_verts.cuda(), self.smplx_faces.cuda())
        self.smplx_mesh_cano = mesh.Mesh(self.smplx_verts_cano.cuda(), self.smplx_faces.cuda())

        # move waist a bit
        for ang in [-30, 30]:
            for ax in [0, 1, 2]:
                pose = torch.zeros([55,3], dtype=torch.float32)
                # legs
                angle = 15*np.pi/180.
                pose[1, 2] = angle
                pose[2, 2] = -angle
                # waist
                angle = ang * np.pi/180.
                pose[3, ax] = angle
                pose[6, ax] = angle
                pose_euler = rotation_converter.batch_euler2axis(pose)
                pose = rotation_converter.batch_euler2matrix(pose)
                pose = pose[None,...]
                xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = pose, return_T=True, transl=torch.tensor([0, 0.4, 0],dtype=torch.float32, ))
                self.refine_poses.append(pose_euler[1:22].reshape(1, -1))
                self.refine_tfs.append(torch.einsum('bnij,njk->bnik', A, self.A_inv).cuda())
                self.refine_op3ds.append(joints[0][self.op_3d_mapping])

        # move arms a bit
        for ang in [60]:
            for ax in [2]:
                pose = torch.zeros([55,3], dtype=torch.float32)
                # legs
                angle = 15*np.pi/180.
                pose[1, 2] = angle
                pose[2, 2] = -angle
                # arms
                angle = ang * np.pi/180.
                pose[13, ax] = angle
                pose[14, ax] = -angle
                pose_euler = rotation_converter.batch_euler2axis(pose)
                pose = rotation_converter.batch_euler2matrix(pose)
                pose = pose[None,...]
                xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = pose, return_T=True, transl=torch.tensor([0, 0.4, 0],dtype=torch.float32, ))
                self.refine_poses.append(pose_euler[1:22].reshape(1, -1))
                self.refine_tfs.append(torch.einsum('bnij,njk->bnik', A, self.A_inv).cuda())
                self.refine_op3ds.append(joints[0][self.op_3d_mapping])

        # move arms a bit
        for ang in [-90]:
            for ax in [0]:
                pose = torch.zeros([55,3], dtype=torch.float32)
                # legs
                angle = 15*np.pi/180.
                pose[1, 2] = angle
                pose[2, 2] = -angle
                # arms
                angle = ang * np.pi/180.
                pose[13, ax] = angle
                pose[14, ax] = angle
                pose_euler = rotation_converter.batch_euler2axis(pose)
                pose = rotation_converter.batch_euler2matrix(pose)
                pose = pose[None,...]
                xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = pose, return_T=True, transl=torch.tensor([0, 0.4, 0],dtype=torch.float32, ))
                self.refine_poses.append(pose_euler[1:22].reshape(1, -1))
                self.refine_tfs.append(torch.einsum('bnij,njk->bnik', A, self.A_inv).cuda())
                self.refine_op3ds.append(joints[0][self.op_3d_mapping])

        # # move legs a bit
        # for ang in [30]:
        #     for ax in [0, 2]:
        #         pose = torch.zeros([55,3], dtype=torch.float32)
        #         # legs
        #         angle = ang*np.pi/180.
        #         pose[1, ax] = angle
        #         pose[2, ax] = -angle
        #         pose_euler = rotation_converter.batch_euler2axis(pose)
        #         pose = rotation_converter.batch_euler2matrix(pose)
        #         pose = pose[None,...]
        #         xyz, _, joints, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = pose, return_T=True, transl=torch.tensor([0, 0.4, 0],dtype=torch.float32, ))
        #         self.init_refine_poses.append(pose_euler[1:22].reshape(1, -1))
        #         self.init_refine_tfs.append(torch.einsum('bnij,njk->bnik', A, self.A_inv).cuda())
        #         self.init_refine_op3ds.append(joints[0][self.op_3d_mapping])


    def set_zoom_body_cam(self):
        target_views = {'mv': [], 'mvp': [], 'campos': [], 'background': [], 'normal_rotate': []}
        # for i in range(8):
        #     target_views['mv'].append(DatasetMesh.train_scene_body_zoom(i, 1)['mv'])
        #     target_views['mvp'].append(DatasetMesh.train_scene_body_zoom(i, 1)['mvp'])
        #     target_views['campos'].append(DatasetMesh.train_scene_body_zoom(i, 1)['campos'])
        #     target_views['background'].append(DatasetMesh.train_scene_body_zoom(i, 1)['background'])
        #     target_views['normal_rotate'].append(DatasetMesh.train_scene_body_zoom(i, 1)['normal_rotate'])
        for i in range(6):
            target_views['mv'].append(DatasetMesh.train_scene_body_zoom(i, 5)['mv'])
            target_views['mvp'].append(DatasetMesh.train_scene_body_zoom(i, 5)['mvp'])
            target_views['campos'].append(DatasetMesh.train_scene_body_zoom(i, 5)['campos'])
            target_views['background'].append(DatasetMesh.train_scene_body_zoom(i, 5)['background'])
            target_views['normal_rotate'].append(DatasetMesh.train_scene_body_zoom(i, 5)['normal_rotate'])
        for i in range(6):
            target_views['mv'].append(DatasetMesh.train_scene_body_zoom(i, 9)['mv'])
            target_views['mvp'].append(DatasetMesh.train_scene_body_zoom(i, 9)['mvp'])
            target_views['campos'].append(DatasetMesh.train_scene_body_zoom(i, 9)['campos'])
            target_views['background'].append(DatasetMesh.train_scene_body_zoom(i, 9)['background'])
            target_views['normal_rotate'].append(DatasetMesh.train_scene_body_zoom(i, 9)['normal_rotate'])
        self.target_body_views = {'mv': torch.cat(target_views['mv'], dim=0), 'mvp': torch.cat(target_views['mvp'], dim=0), 'campos': torch.cat(target_views['campos'], dim=0), \
                            'background': torch.cat(target_views['background'], dim=0), 'normal_rotate': torch.cat(target_views['normal_rotate'], dim=0),
                            'resolution' : torch.tensor([512, 512]), 'spp' : 1, 'prompt_index' : 0}
        
    def set_zoom_obj_cam(self):
        target_views = {'mv': [], 'mvp': [], 'campos': [], 'background': [], 'normal_rotate': []}
        for i in range(4):
            target_views['mv'].append(DatasetMesh.train_scene_segm_2(i)['mv'])
            target_views['mvp'].append(DatasetMesh.train_scene_segm_2(i)['mvp'])
            target_views['campos'].append(DatasetMesh.train_scene_segm_2(i)['campos'])
            target_views['background'].append(DatasetMesh.train_scene_segm_2(i)['background'])
            target_views['normal_rotate'].append(DatasetMesh.train_scene_segm_2(i)['normal_rotate'])
        self.target_obj_views = {'mv': torch.cat(target_views['mv'], dim=0), 'mvp': torch.cat(target_views['mvp'], dim=0), 'campos': torch.cat(target_views['campos'], dim=0), \
                            'background': torch.cat(target_views['background'], dim=0), 'normal_rotate': torch.cat(target_views['normal_rotate'], dim=0),
                            'resolution' : torch.tensor([512, 512]), 'spp' : 1, 'prompt_index' : 0}


    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2) 
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)


    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    

    # mesh cleaning code from pifuhd
    @torch.no_grad()
    def get_obj_bbox(self, mesh_obj):
        verts, faces = mesh_obj.v_pos.clone().detach(), mesh_obj.t_pos_idx.clone().detach()
        import trimesh
        mesh_tri = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())
        cc = mesh_tri.split(only_watertight=False)
        out_mesh = cc[0]
        bbox = out_mesh.bounds
        height = bbox[1,0] - bbox[0,0]
        for c in cc:
            bbox = c.bounds
            if height < bbox[1,0] - bbox[0,0]:
                height = bbox[1,0] - bbox[0,0]
                out_mesh = c
        verts, faces = torch.tensor(out_mesh.vertices).to(device=mesh_obj.v_pos.device, dtype=torch.float32), torch.tensor(out_mesh.faces)
        return verts.min(axis=0), verts.max(axis=0)

    # get vert and faces via MT
    def get_verts_and_faces(self, pred, mesh_type='comp', detach=False, subdiv=False):
        sdf, deform = pred[:, 0], pred[:, 1:]
        if detach:
            sdf, deform = sdf.clone().detach(), deform.clone().detach()
        v_deformed = self.verts + 1 / (self.grid_res) * torch.tanh(deform)
        tet = self.indices
        if subdiv:
            for _ in range(self.num_subdivision):
                v_deformed, _, tet = compact_tets(v_deformed, sdf, tet)
                v_deformed, tet = batch_subdivide_volume(v_deformed.unsqueeze(0), tet.unsqueeze(0))
                v_deformed, tet = v_deformed[0], tet[0]
                pred, pred_obj = self.decoder(v_deformed, mesh_type=mesh_type)
                sdf = pred[:,0] if mesh_type == 'human' else pred_obj[:,0]
        verts, faces = self.marching_tets(v_deformed, sdf, tet)
        return verts, faces, sdf
    

    # transforms verts in canonical space to posed space using NN smplx vertex (incorporates shape param offsets)
    def cano_to_posed(self, verts, tfs):
        offsets = torch.nn.functional.grid_sample(self.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
        lbs_weights = torch.nn.functional.grid_sample(self.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
        verts = verts + offsets[0]  # shape pose blend shapes
        verts = helpers.skinning(verts, lbs_weights, tfs)[0]  # lbs
        return verts


    def getMesh(self, material, mesh_type='comp', detach_hum=False, detach_obj=False, canonical=False, val=False):
        '''
            mesh_type: [human, obj, comp]
        '''
        pred_hum, pred_obj = self.decoder(self.verts, mesh_type=mesh_type)
        
        if mesh_type == 'human':
            verts, faces, sdf_hum = self.get_verts_and_faces(pred_hum, mesh_type, detach=detach_hum, \
                                                         subdiv=(self.subdivision and self.tick_count > self.FLAGS.subdiv_iter) or val)
            imesh_hum = mesh.Mesh(verts, faces, material=material)
            imesh_hum = mesh.auto_normals(imesh_hum)

            if self.enable_canonical and not canonical: 
                verts = self.cano_to_posed(verts, self.gt_smplx_tfs)
                imesh_hum_posed = mesh.Mesh(verts, faces, material=material)
                imesh_hum_posed = mesh.auto_normals(imesh_hum_posed)

        elif mesh_type == 'obj':
            verts_obj, faces_obj, sdf_obj = self.get_verts_and_faces(pred_obj, mesh_type, detach=detach_obj, \
                                                         subdiv=(self.subdivision and self.tick_count > self.FLAGS.subdiv_iter) or val)
            imesh_obj = mesh.Mesh(verts_obj, faces_obj, material=material)
            imesh_obj = mesh.auto_normals(imesh_obj)
        
            if self.enable_canonical and not canonical:
                verts_obj = self.cano_to_posed(verts_obj, self.gt_smplx_tfs)
                imesh_obj_posed = mesh.Mesh(verts_obj, faces_obj, material=material)
                imesh_obj_posed = mesh.auto_normals(imesh_obj_posed)
        
        elif mesh_type == 'comp':
            verts_hum, faces_hum, sdf_hum = self.get_verts_and_faces(pred_hum, mesh_type='human', detach=detach_hum, \
                                                         subdiv=(self.subdivision and self.tick_count > self.FLAGS.subdiv_iter) or val)
            imesh_hum = mesh.Mesh(verts_hum, faces_hum, material=material)
            imesh_hum = mesh.auto_normals(imesh_hum)             

            verts_obj, faces_obj, sdf_obj = self.get_verts_and_faces(pred_obj, mesh_type='obj', detach=detach_obj, \
                                                         subdiv=(self.subdivision and self.tick_count > self.FLAGS.subdiv_iter) or val)
            imesh_obj = mesh.Mesh(verts_obj, faces_obj, material=material)
            imesh_obj = mesh.auto_normals(imesh_obj)

            imesh_comp = mesh.combine(imesh_hum, imesh_obj)
            imesh_comp.material = material
            imesh_comp = mesh.auto_normals(imesh_comp)

            if self.enable_canonical and not canonical:
                verts_hum_posed = self.cano_to_posed(verts_hum, self.gt_smplx_tfs)
                imesh_hum_posed = mesh.Mesh(verts_hum_posed, faces_hum, material=material)
                imesh_hum_posed = mesh.auto_normals(imesh_hum_posed)
                
                verts_obj_posed = self.cano_to_posed(verts_obj, self.gt_smplx_tfs)
                imesh_obj_posed = mesh.Mesh(verts_obj_posed, faces_obj, material=material)
                imesh_obj_posed = mesh.auto_normals(imesh_obj_posed)

                imesh_comp_posed = mesh.combine(imesh_hum_posed, imesh_obj_posed)
                imesh_comp_posed.material = material
                imesh_comp_posed = mesh.auto_normals(imesh_comp_posed)
        else:
            raise NotImplementedError

        mesh_dict = {}
        if self.enable_canonical and canonical:
            if mesh_type == 'comp': 
                mesh_dict['mesh_hum_cano'] = imesh_hum
                mesh_dict['mesh_obj_cano'] = imesh_obj
                mesh_dict['mesh_comp_cano'] = imesh_comp
                mesh_dict['sdf_hum'] = sdf_hum
                mesh_dict['sdf_obj'] = sdf_obj
            elif mesh_type == 'human': 
                mesh_dict['mesh_hum_cano'] = imesh_hum
                mesh_dict['sdf_hum'] = sdf_hum
            elif mesh_type == 'obj': 
                mesh_dict['mesh_obj_cano'] = imesh_obj
                mesh_dict['sdf_obj'] = sdf_obj

        elif self.enable_canonical and not canonical:
            if mesh_type == 'comp': 
                mesh_dict['mesh_hum_posed'] = imesh_hum_posed
                mesh_dict['mesh_obj_posed'] = imesh_obj_posed
                mesh_dict['mesh_comp_posed'] = imesh_comp_posed
                mesh_dict['mesh_hum_cano'] = imesh_hum
                mesh_dict['mesh_obj_cano'] = imesh_obj
                mesh_dict['mesh_comp_cano'] = imesh_comp
                mesh_dict['sdf_hum'] = sdf_hum
                mesh_dict['sdf_obj'] = sdf_obj 
            elif mesh_type == 'human': 
                mesh_dict['mesh_hum_posed'] = imesh_hum_posed
                mesh_dict['mesh_hum_cano'] = imesh_hum
                mesh_dict['sdf_hum'] = sdf_hum
            elif mesh_type == 'obj': 
                mesh_dict['mesh_obj_posed'] = imesh_obj_posed
                mesh_dict['mesh_obj_cano'] = imesh_obj
                mesh_dict['sdf_obj'] = sdf_obj
        
        else:  # no canonical modeling
            if mesh_type == 'comp': 
                mesh_dict['mesh_hum_posed'] = imesh_hum
                mesh_dict['mesh_obj_posed'] = imesh_obj
                mesh_dict['mesh_comp_posed'] = imesh_comp
                mesh_dict['sdf_hum'] = sdf_hum
                mesh_dict['sdf_obj'] = sdf_obj 
            elif mesh_type == 'human': 
                mesh_dict['mesh_hum_posed'] = imesh_hum
                mesh_dict['sdf_hum'] = sdf_hum
            elif mesh_type == 'obj': 
                mesh_dict['mesh_obj_posed'] = imesh_obj
                mesh_dict['sdf_obj'] = sdf_obj
        
        return mesh_dict

    def render(self, mesh_type, glctx, target, lgt, opt_material, opt_mesh=None, bsdf=None, if_normal=False, mode='geometry_modeling', 
               if_flip_the_normal = False, if_use_bump = False, zoom_in_params=None, detach_hum=False, detach_obj=False, render_control_op=False, canonical=False, val=False, return_rast=False):
        if opt_mesh is None:
            mesh_dict = self.getMesh(opt_material, mesh_type=mesh_type, detach_hum=detach_hum, detach_obj=detach_obj, 
                                     canonical=canonical, val=val)
            if self.enable_canonical and canonical:
                if mesh_type == 'comp': 
                    opt_mesh = mesh_dict['mesh_comp_cano']
                elif mesh_type == 'human': 
                    opt_mesh = mesh_dict['mesh_hum_cano']
                elif mesh_type == 'obj':
                    opt_mesh = mesh_dict['mesh_obj_cano']
                else:
                    raise NotImplementedError
            
            elif self.enable_canonical and not canonical:
                if mesh_type == 'comp':  
                    opt_mesh = mesh_dict['mesh_comp_posed']
                elif mesh_type == 'human': 
                    opt_mesh = mesh_dict['mesh_hum_posed']
                elif mesh_type == 'obj': 
                    opt_mesh = mesh_dict['mesh_obj_posed']
                else:
                    raise NotImplementedError
            else:
                if mesh_type == 'comp': 
                    opt_mesh = mesh_dict['mesh_comp_posed']
                elif mesh_type == 'hum':
                    opt_mesh = mesh_dict['mesh_hum_posed']
                elif mesh_type == 'obj':
                    opt_mesh = mesh_dict['mesh_obj_posed']
                else: 
                    raise NotImplementedError

        op_3d = None
        if render_control_op:
            op_3d = self.op_3d
            if canonical : op_3d = self.op_3d_cano
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True, background= target['background'],
                                  bsdf= bsdf,if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,if_flip_the_normal = if_flip_the_normal,if_use_bump = if_use_bump,
                                  zoom_in_params=zoom_in_params, op_3d = op_3d, return_rast=return_rast)
      
    def choose_buffer(self, mesh_type, mesh, mesh_cano, glctx, target, lgt, opt_material, if_normal, mode, if_flip_the_normal, if_use_bump, zoom_in):
        if self.enable_canonical:
            if self.use_canonical_sds:
                buffers_cano = self.render(mesh_type, glctx, target, lgt, opt_material, opt_mesh=mesh_cano, if_normal= if_normal, mode=mode, \
                                            if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, zoom_in_params=zoom_in,\
                                                render_control_op=self.use_op_control, canonical=True)
                buffers_sds = buffers_cano
            else:
                buffers = self.render(mesh_type, glctx, target, lgt, opt_material, opt_mesh=mesh, if_normal= if_normal, mode=mode, zoom_in_params=zoom_in,\
                                            if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)
                buffers_sds = buffers
        else:
            buffers = self.render(mesh_type, glctx, target, lgt, opt_material, opt_mesh=mesh, if_normal= if_normal, mode=mode, zoom_in_params=zoom_in,\
                                            if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)
            buffers_sds = buffers
        return buffers_sds

    def apply_SDS(self, buffers, text_embeddings, guidance, iteration):
        if iteration <= self.FLAGS.coarse_iter:
            t = torch.randint( guidance.min_step_early, guidance.max_step_early + 1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
            # t = torch.randint( guidance.min_step_early, guidance.max_step_early + 1, [1], dtype=torch.long, device='cuda') # [B]
            normal_maps =  buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous() # [B, 3, 512, 512]
            latents_512 =  buffers['shaded'][..., 0:4].permute(0, 3, 1, 2).contiguous() # [B, 4, 512, 512]
            if guidance.use_legacy:
                latents = F.interpolate(latents_512, (64, 64), mode='bilinear', align_corners=False)  # [B, 4, 64, 64]
            pred_rgb_512 = latents_512
            downsample_for_latent = True
        else:
            t = torch.randint(guidance.min_step_late, guidance.max_step_late + 1, [self.FLAGS.batch], dtype=torch.long, device='cuda')
            srgb =  buffers['shaded'][..., 0:3]
            pred_rgb_512 = srgb.permute(0, 3, 1, 2).contiguous()  # [B, 3, 512, 512]
            # pred_rgb_512 = pred_rgb_512 * 0.5 + 0.5 # -1 to 1 -> 0 to 1
            # pred_rgb_512 = torch.clamp(pred_rgb_512, 0, 1)
            if guidance.use_legacy:
                latents = guidance.encode_imgs(pred_rgb_512)  # [B, 4, 64, 64]
            downsample_for_latent = False

        # prepare inpaint mask
        mask_inpaint = None
        latents_masked = None
        if guidance.use_inpaint:
            if self.use_canonical_sds:
                mask_inpaint = (buffers['segm'][..., 0]) * buffers['shaded'][..., -1]
            else:
                mask_inpaint = (self.dilate_seg(1 - gt_seg[..., 1])) if self.dilate_seg_mask else 1 - gt_seg[..., 1]


        # prepare control images
        control_images = None
        if guidance.enable_controlnet:
            if guidance.control_type == 'op':
                control_images = buffers['op_img']
            else:
                raise NotImplementedError

        if guidance.use_legacy:
            if guidance.use_inpaint:
                raise ValueError("Do not use legacy inpainting code!")

            # generate noise and predict noise
            noise = torch.randn_like(latents)
            noise_pred = guidance(latents, noise, t, text_embeddings, control_images=control_images)
        else:
            # diffusers single step
            negative_prompt_embeds, prompt_embeds = text_embeddings.chunk(2)
            latents, noise, noise_pred = guidance.pipe_single_step.single_step(
                                t, 
                                downsample_for_latent=downsample_for_latent,
                                image=pred_rgb_512, 
                                mask_image=mask_inpaint, 
                                prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                control_image=control_images)

        if iteration <= self.FLAGS.coarse_iter:
            w = (1 - guidance.alphas[t]) # [B]
        else:
            w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
            # w = 1 / (1 - guidance.alphas[t])
        w = w[:, None, None, None] # [B, 1, 1, 1]
        grad =  w * (noise_pred - noise) #*w1 s
        grad = torch.nan_to_num(grad)
        return SpecifyGradient.apply(latents, grad) 


    def tick(self, glctx, target, lgt, opt_material, iteration, if_normal, guidance, scene_and_vertices, mode, if_flip_the_normal, if_use_bump):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================

        sds_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        loss_fn = torch.nn.MSELoss()
        self.tick_count = iteration

        if iteration > self.FLAGS.subdiv_iter + 1: self.FLAGS['vis_subdiv'] = True

        ################################################################################
        # Initialization
        ################################################################################
        if iteration < self.FLAGS.init_iter:              

            if self.gt_mesh_hum is None:
                gt_mesh_hum = self.smplx_mesh_cano if self.enable_canonical else self.gt_smplx_mesh
                gt_mesh_hum.material = opt_material
                gt_mesh_hum = mesh.auto_normals(gt_mesh_hum)
                self.gt_mesh_hum = gt_mesh_hum
            gt_buffers_hum = render.render_mesh(glctx, self.gt_mesh_hum, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                            background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                            if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
            
            if self.init_sdf and self.init_scence_vertices is None:
                vertices = np.asarray(self.gt_mesh_hum.v_pos.cpu())
                faces = np.asarray(self.gt_mesh_hum.t_pos_idx.cpu())
                init_shape = o3d.geometry.TriangleMesh()
                init_shape.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
                init_shape.triangles = o3d.cuda.pybind.utility.Vector3iVector(faces)
                points_surface = np.asarray(init_shape.sample_points_poisson_disk(5000).points)
                init_shape = o3d.t.geometry.TriangleMesh.from_legacy(init_shape)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(init_shape)
                self.init_scence_vertices = [scene, points_surface]

            if self.gt_mesh_obj is None:
                gt_mesh_obj = scene_and_vertices[6]
                gt_mesh_obj.material = opt_material
                gt_mesh_obj = mesh.auto_normals(gt_mesh_obj)
                self.gt_mesh_obj = gt_mesh_obj
            gt_buffers_obj = render.render_mesh(glctx, self.gt_mesh_obj, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                            background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                            if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
            
            if self.enable_canonical:
                mesh_dict = self.getMesh(opt_material, mesh_type='human', canonical=True)
                mesh_hum = mesh_dict['mesh_hum_cano']
                sdf_hum_cano = mesh_dict['sdf_hum']
                if self.init_obj_smplx:
                    mesh_dict = self.getMesh(opt_material, mesh_type='obj', canonical=True)
                    mesh_obj = mesh_dict['mesh_obj_cano']
                    sdf_obj = mesh_dict['sdf_obj']
                else:
                    mesh_dict = self.getMesh(opt_material, mesh_type='obj')
                    mesh_obj = mesh_dict['mesh_obj_posed']
                    mesh_obj_cano = mesh_dict['mesh_obj_cano']
                    sdf_obj = mesh-dict['sdf_obj']
            else:
                mesh_dict_hum = self.getMesh(opt_material, mesh_type='human')
                mesh_hum = mesh_dict_hum['mesh_hum_posed']
                sdf_hum = mesh_dict_hum['sdf_hum']
                mesh_dict_obj = self.getMesh(opt_material, mesh_type='obj')
                mesh_obj = mesh_dict_obj['mesh_obj_posed']
                sdf_obj = mesh_dict_obj['sdf_obj']

            buffers = self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal= if_normal, mode = mode, 
                                  if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
        
            buffers_obj = self.render('obj', glctx, target, lgt, opt_material, opt_mesh=mesh_obj, if_normal= if_normal, mode = mode, 
                                         if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
            
            gt_img_hum = gt_buffers_hum['shaded'][..., :3]
            gt_mask_hum = gt_buffers_hum['shaded'][..., -1]

            if self.init_obj_smplx:
                gt_img_obj = gt_buffers_hum['shaded'][..., :3]
                gt_mask_obj = gt_buffers_hum['shaded'][..., -1]
            else:
                 # initialize object space with object mesh
                gt_img_obj = gt_buffers_obj['shaded'][..., :3]
                gt_mask_obj = gt_buffers_obj['shaded'][..., -1]

            img_hum = buffers['shaded'][..., :3]
            mask_hum = buffers['shaded'][..., -1]
            img_obj = buffers_obj['shaded'][..., :3]
            mask_obj = buffers_obj['shaded'][..., -1]
        
            mask_loss = loss_fn(mask_hum, gt_mask_hum) + loss_fn(mask_obj, gt_mask_obj)
            norm_loss = loss_fn(img_hum * mask_hum.unsqueeze(-1), gt_img_hum * gt_mask_hum.unsqueeze(-1)) + \
                        loss_fn(img_obj * mask_obj.unsqueeze(-1), gt_img_obj * gt_mask_obj.unsqueeze(-1))

            # recon loss
            if self.init_sdf:
                reg_loss += self.decoder.pre_train_sdf(iteration, self.init_scence_vertices)
            else:
                reg_loss += (mask_loss + norm_loss)

            # min surface loss
            reg_loss += torch.exp(-100.0 * torch.abs(sdf_obj)).mean() * self.w_reg_min_surface_init if self.enable_min_surface else 0.0 # 0.2 critical for training

        # save ckpt for initialization wo refinement
        if iteration == self.FLAGS.init_iter:
            if self.FLAGS.save_ckpt:
                torch.save(self.state_dict(), str(self.FLAGS.out_dir / 'init.ckpt'))

        # save ckpt for initialization wo refinement
        if iteration == self.FLAGS.iter:
            torch.save(self.state_dict(), str(self.FLAGS.out_dir / 'final.ckpt'))

        # get obj bbox for zoom in, zoom in after refining the object canonical space (get rid of floaters) via SDS loss in composition space
        if iteration == self.FLAGS.init_refine_iter:
            if self.enable_zoom_in:
                if self.enable_canonical:
                    bbox_min, bbox_max = self.get_obj_bbox(self.getMesh(opt_material, mesh_type='obj')['mesh_obj_cano']) 
                else:
                    bbox_min, bbox_max = self.get_obj_bbox(self.getMesh(opt_material, mesh_type='obj')['mesh_obj_posed'])
                zoom_in_params = {}
                zoom_in_params['scale_start'] = (max((bbox_max[0] - bbox_min[0]) / 2 * 1.2)) # largest of xyz scales, gets smaller if multiplied by larger value at the end
                zoom_in_params['scale_end'] = (max((bbox_max[0] - bbox_min[0]) / 2 * 0.6))
                zoom_in_params['transl_start'] = ((bbox_max[0] - bbox_min[0]) / 4) + bbox_min[0]
                zoom_in_params['transl_end'] = bbox_max[0] - ((bbox_max[0] - bbox_min[0]) / 4)
                self.zoom_in_params = zoom_in_params
            
        ################################################################################
        # Decomposition, Reconstruction & Generation
        ################################################################################ 
        
        if iteration >= self.FLAGS.init_iter:
            ################################################################################
            # 1. Render GTs for reconstruction loss
            ################################################################################
            if self.gt_mesh is None:
                gt_mesh = mesh.Mesh(scene_and_vertices[2], scene_and_vertices[3], material=opt_material, f_seg=scene_and_vertices[4])
                gt_mesh = mesh.auto_normals(gt_mesh)
                self.gt_mesh = gt_mesh
            gt_buffers= render.render_mesh(glctx, self.gt_mesh.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                            background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                            if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
            
            if self.is_obj_most_outer:
                if self.gt_mesh_obj is None:
                    gt_mesh_obj = scene_and_vertices[6]
                    gt_mesh_obj.material = opt_material
                    gt_mesh_obj = mesh.auto_normals(gt_mesh_obj)
                    self.gt_mesh_obj = gt_mesh_obj
                gt_buffers_obj = render.render_mesh(glctx, self.gt_mesh_obj.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
                gt_img_obj = gt_buffers_obj['shaded'][..., :3]
                gt_mask_obj = gt_buffers_obj['shaded'][..., -1]
            
            gt_img = gt_buffers['shaded'][..., :3]
            gt_mask = gt_buffers['shaded'][..., -1]
            gt_seg = torch.cat([(gt_buffers['segm'][..., 0]==1).float()[..., None], 
                                (gt_buffers['segm'][..., 0]==0).float()[..., None]], axis=-1)
            
    
            ################################################################################
            # 2. Get meshes via marching tets
            ################################################################################   
        
            if self.enable_canonical:
                # meshes with gradients
                mesh_dict = self.getMesh(opt_material, mesh_type='comp', 
                                         detach_hum=False, detach_obj=False) # detach obj if self.detach_obj is true
                mesh_hum = mesh_dict['mesh_hum_posed']
                mesh_obj = mesh_dict['mesh_obj_posed']
                mesh_comp = mesh_dict['mesh_comp_posed']
                mesh_hum_cano = mesh_dict['mesh_hum_cano']
                mesh_obj_cano = mesh_dict['mesh_obj_cano']
                mesh_comp_cano = mesh_dict['mesh_comp_cano']
                sdf_hum_cano = mesh_dict['sdf_hum']
                sdf_obj_cano = mesh_dict['sdf_obj'] 
            
            else:
                # meshes with gradients
                mesh_dict = self.getMesh(opt_material, mesh_type='comp', 
                                         detach_hum=False, detach_obj=False) # detach obj if self.detach_obj is true
                mesh_hum = mesh_dict['mesh_hum_posed']
                mesh_obj = mesh_dict['mesh_obj_posed']
                mesh_comp = mesh_dict['mesh_comp_posed']
                sdf_hum = mesh_dict['sdf_hum']
                sdf_obj = mesh_dict['sdf_obj']
                mesh_hum_cano, mesh_obj_cano, mesh_comp_cano = None, None, None


            ################################################################################
            # 4. Render posed space meshes for reconstruction
            ################################################################################

            buffers_comp = self.render('comp', glctx, target, lgt, opt_material, opt_mesh=mesh_comp, if_normal=if_normal, 
                                       mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)

            buffers_hum = self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal=if_normal, 
                                 mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)
            
            buffers_obj= self.render('obj', glctx, target, lgt, opt_material, opt_mesh=mesh_obj, if_normal= if_normal,
                                        mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)
            
            self.gt_smplx_mesh = mesh.auto_normals(self.gt_smplx_mesh)
            self.gt_smplx_mesh.material = opt_material
            buffers_smpl = render.render_mesh(glctx, self.gt_smplx_mesh.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                            background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                            if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)

            # reconstruction loss in human posed space
            img_hum = buffers_hum['shaded'][..., :3]
            mask_hum = buffers_hum['shaded'][..., -1]
            gt_seg_hum = gt_seg[..., 1] if not self.dilate_seg_mask else -self.dilate_seg(-gt_seg[..., 1])

            if (not self.detach_hum or iteration % 2 == 0 and self.take_turns) or not self.use_canonical_sds:
                reg_loss += loss_fn(img_hum * mask_hum.unsqueeze(-1) * gt_seg_hum.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_hum.unsqueeze(-1)) * self.w_recon_full
                smpl_mask = buffers_smpl['shaded'][..., -1]
                erode_size = 5
                erode_filter = torch.nn.MaxPool2d(kernel_size=erode_size, stride=1, padding=erode_size // 2)
                gt_mask_erode = -erode_filter(-gt_mask)
                overlap_mask = smpl_mask * gt_mask_erode
                reg_loss += 1e5 * loss_fn(mask_hum * overlap_mask, overlap_mask)  # it only cares whether foreground mask includes the overlap region of gt human and smpl mesh

            # reconstruction loss in object posed space
            img_obj = buffers_obj['shaded'][..., :3]
            mask_obj = buffers_obj['shaded'][..., -1]
            gt_seg_obj = gt_seg[..., 0] if not self.dilate_seg_mask else self.dilate_seg(gt_seg[..., 0])

            if (not self.detach_obj and iteration % 2 == 1 and self.take_turns) and self.use_canonical_sds:      
                if self.is_obj_most_outer:
                    reg_loss += (loss_fn(mask_obj, gt_mask_obj) * 50 + loss_fn(img_obj * mask_obj.unsqueeze(-1), gt_img_obj * gt_mask_obj.unsqueeze(-1))) * self.w_recon_full
                else:
                    reg_loss += loss_fn(img_obj * mask_obj.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1)) * self.w_recon_full 


            # reconstruction loss in comp posed space
            img_comp = buffers_comp['shaded'][..., :3]
            mask_comp = buffers_comp['shaded'][..., -1]
            segm_comp = buffers_comp['segm'][..., :2]

            mask_loss = loss_fn(mask_comp, gt_mask) # prevents floaters -> replace with min surface loss
            segm_loss = loss_fn(segm_comp * mask_comp.unsqueeze(-1), gt_seg * gt_mask.unsqueeze(-1))  # segm loss should be larger than mask loss to keep hum and obj separated
            # norm_loss = loss_fn(img_comp * mask_comp.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1)) # no normal loss

            reg_loss += (segm_loss * self.w_recon_segm) # segm loss should be large enough

            # bird eye view recon for object layer
            if self.is_obj_most_outer:
                if self.gt_mesh_obj is None:
                    gt_mesh_obj = scene_and_vertices[6]
                    gt_mesh_obj.material = opt_material
                    gt_mesh_obj = mesh.auto_normals(gt_mesh_obj)
                    self.gt_mesh_obj = gt_mesh_obj
                gt_buffers_obj = render.render_mesh(glctx, self.gt_mesh_obj.clone(), self.target_obj_views['mvp'], self.target_obj_views['campos'], lgt, self.target_obj_views['resolution'], spp=self.target_obj_views['spp'], msaa= True,
                                                background= self.target_obj_views['background'],bsdf= None, if_normal= if_normal,normal_rotate= self.target_obj_views['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
                gt_img_obj = gt_buffers_obj['shaded'][..., :3]
                gt_mask_obj = gt_buffers_obj['shaded'][..., -1]

                buffers_obj= self.render('obj', glctx, self.target_obj_views, lgt, opt_material, opt_mesh=mesh_obj, if_normal= if_normal,
                                        mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)

                # reconstruction loss in object posed space
                img_obj = buffers_obj['shaded'][..., :3]
                mask_obj = buffers_obj['shaded'][..., -1]
                gt_seg_obj = gt_seg[..., 0] if not self.dilate_seg_mask else self.dilate_seg(gt_seg[..., 0])

                if (not self.detach_obj and iteration % 2 == 1 and self.take_turns) and self.use_canonical_sds:      
                    if self.is_obj_most_outer:
                        reg_loss += (loss_fn(mask_obj, gt_mask_obj) * 50 + loss_fn(img_obj * mask_obj.unsqueeze(-1), gt_img_obj * gt_mask_obj.unsqueeze(-1))) * self.w_recon_full
                    else:
                        reg_loss += loss_fn(img_obj * mask_obj.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1)) * self.w_recon_full 


            ################################################################################
            # 5. body recon loss with the zoomed in images if enabled
            ################################################################################ 
            if self.zoom_in_body:
                start, end = torch.zeros(3, device='cuda'), torch.zeros(3, device='cuda')
                start[..., 1] -= 0.8
                end[..., 1] += 0.6
                zoom_in_params_body = {
                    'scale': torch.tensor(0.25).cuda(),             
                    'transl': start + torch.rand_like(start) * (end - start)
                }
                gt_buffers = render.render_mesh(glctx, self.gt_mesh.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_body)
                gt_img = gt_buffers['shaded'][..., :3]
                gt_mask = gt_buffers['shaded'][..., -1]
                gt_seg = torch.cat([(gt_buffers['segm'][..., 0]==1).float()[..., None], 
                                    (gt_buffers['segm'][..., 0]==0).float()[..., None]], axis=-1)
                
                gt_buffers_obj = render.render_mesh(glctx, self.gt_mesh_obj.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_body)
                gt_img_obj = gt_buffers_obj['shaded'][..., :3]
                gt_mask_obj = gt_buffers_obj['shaded'][..., -1]
            
                buffers_hum = render.render_mesh(glctx, mesh_hum, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_body)
                
                buffers_obj = render.render_mesh(glctx, mesh_obj, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_body)
                
                buffers_comp = render.render_mesh(glctx, mesh_comp, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_body)
                
                # reconstruction loss in human posed space
                img_hum = buffers_hum['shaded'][..., :3]
                mask_hum = buffers_hum['shaded'][..., -1]
                gt_seg_hum = gt_seg[..., 1] if not self.dilate_seg_mask else -self.dilate_seg(-gt_seg[..., 1])

                if (not self.detach_hum or iteration % 2 == 0 and self.take_turns) or not self.use_canonical_sds:
                    reg_loss += loss_fn(img_hum * mask_hum.unsqueeze(-1) * gt_seg_hum.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_hum.unsqueeze(-1)) * self.w_recon_full

                # reconstruction loss in object posed space
                img_obj = buffers_obj['shaded'][..., :3]
                mask_obj = buffers_obj['shaded'][..., -1]
                gt_seg_obj = gt_seg[..., 0] if not self.dilate_seg_mask else self.dilate_seg(gt_seg[..., 0])

                if (not self.detach_obj and iteration % 2 == 1 and self.take_turns):
                    if self.is_obj_most_outer:
                        reg_loss += (loss_fn(mask_obj, gt_mask_obj) * 50 + loss_fn(img_obj * mask_obj.unsqueeze(-1), gt_img_obj * gt_mask_obj.unsqueeze(-1))) * self.w_recon_full
                    else:
                        reg_loss += loss_fn(img_obj * mask_obj.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1)) * self.w_recon_full 

                # reconstruction loss in comp posed space
                img_comp = buffers_comp['shaded'][..., :3]
                mask_comp = buffers_comp['shaded'][..., -1]
                segm_comp = buffers_comp['segm'][..., :2]

                mask_loss = loss_fn(mask_comp, gt_mask) # prevents floaters -> replace with min surface loss
                segm_loss = loss_fn(segm_comp * mask_comp.unsqueeze(-1), gt_seg * gt_mask.unsqueeze(-1))  # segm loss should be larger than mask loss to keep hum and obj separated
                # norm_loss = loss_fn(img_comp * mask_comp.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1)) # no normal loss

                reg_loss += (segm_loss * self.w_recon_segm) # segm loss should be large enough


            ################################################################################
            # 5. head and hand recon loss with the zoomed in images if enabled
            ################################################################################ 
            if self.zoom_in_head_hands:
                # set zoom in params based on joint positions
                zoom_in_params_head = {
                    'scale': torch.tensor(0.2).cuda(),
                    'transl': self.gt_smplx_verts[self.part_idx_dict['face']].mean(axis=0).cuda()
                }
                zoom_in_params_left_hand = {
                    'scale': torch.tensor(0.1).cuda(),
                    'transl': self.gt_smplx_verts[self.part_idx_dict['left_hand']].mean(axis=0).cuda()
                }
                zoom_in_params_right_hand = {
                    'scale': torch.tensor(0.1).cuda(),
                    'transl': self.gt_smplx_verts[self.part_idx_dict['right_hand']].mean(axis=0).cuda()
                }
                
                # head
                gt_buffers_head = render.render_mesh(glctx, self.gt_mesh.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_head)
                buffers_head = self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal= if_normal, mode = mode, 
                                    if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_head)
                gt_img_head = gt_buffers_head['shaded'][..., :3]
                gt_mask_head = gt_buffers_head['shaded'][..., -1]
                gt_seg_hum_head = torch.cat([(gt_buffers_head['segm'][..., 0]==1).float()[..., None], 
                                        (gt_buffers_head['segm'][..., 0]==0).float()[..., None]], axis=-1)[..., 1]
                if self.dilate_seg_mask: gt_seg_hum_head = -self.dilate_seg(-gt_seg_hum_head)
                img_head = buffers_head['shaded'][..., :3]
                mask_head = buffers_head['shaded'][..., -1]    
                mask_loss_head = loss_fn(mask_head * gt_seg_hum_head, gt_mask_head * gt_seg_hum_head)
                norm_loss_head = loss_fn(img_head * mask_head.unsqueeze(-1) * gt_seg_hum_head.unsqueeze(-1), gt_img_head * gt_mask_head.unsqueeze(-1) * gt_seg_hum_head.unsqueeze(-1))
                reg_loss += (mask_loss_head + norm_loss_head) * self.w_recon_full

                # left hand
                gt_buffers_lh = render.render_mesh(glctx, self.gt_mesh.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_left_hand)
                buffers_lh = self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal= if_normal, mode = mode, 
                                    if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_left_hand)
                gt_img_lh = gt_buffers_lh['shaded'][..., :3]
                gt_mask_lh = gt_buffers_lh['shaded'][..., -1]
                gt_seg_hum_lh = torch.cat([(gt_buffers_lh['segm'][..., 0]==1).float()[..., None], 
                                        (gt_buffers_lh['segm'][..., 0]==0).float()[..., None]], axis=-1)[..., 1]
                if self.dilate_seg_mask: gt_seg_hum_lh = -self.dilate_seg(-gt_seg_hum_lh)
                img_lh = buffers_lh['shaded'][..., :3]
                mask_lh = buffers_lh['shaded'][..., -1]
                mask_loss_lh = loss_fn(mask_lh * gt_seg_hum_lh, gt_mask_lh * gt_seg_hum_lh)
                norm_loss_lh = loss_fn(img_lh * mask_lh.unsqueeze(-1) * gt_seg_hum_lh.unsqueeze(-1), gt_img_lh * gt_mask_lh.unsqueeze(-1) * gt_seg_hum_lh.unsqueeze(-1))
                reg_loss += (mask_loss_lh + norm_loss_lh) * self.w_recon_full

                # right hand
                gt_buffers_rh = render.render_mesh(glctx, self.gt_mesh.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                                background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
                                                if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_right_hand)
                buffers_rh = self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal= if_normal, mode = mode, 
                                    if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_right_hand)
                gt_img_rh = gt_buffers_rh['shaded'][..., :3]
                gt_mask_rh = gt_buffers_rh['shaded'][..., -1]
                gt_seg_hum_rh = torch.cat([(gt_buffers_rh['segm'][..., 0]==1).float()[..., None], 
                                        (gt_buffers_rh['segm'][..., 0]==0).float()[..., None]], axis=-1)[..., 1]
                if self.dilate_seg_mask: gt_seg_hum_rh = -self.dilate_seg(-gt_seg_hum_rh)
                img_rh = buffers_rh['shaded'][..., :3]
                mask_rh = buffers_rh['shaded'][..., -1]
                mask_loss_rh = loss_fn(mask_rh * gt_seg_hum_rh, gt_mask_rh * gt_seg_hum_rh)
                norm_loss_rh = loss_fn(img_rh * mask_rh.unsqueeze(-1) * gt_seg_hum_rh.unsqueeze(-1), gt_img_rh * gt_mask_rh.unsqueeze(-1) * gt_seg_hum_rh.unsqueeze(-1))
                reg_loss += (mask_loss_rh + norm_loss_rh) * self.w_recon_full

            ################################################################################
            # 6. Regularizations
            ################################################################################ 

            # laplacian + consistency loss
            reg_loss += regularizer.laplace_regularizer_const(mesh_hum.v_pos, mesh_hum.t_pos_idx) * self.w_reg_lap if self.enable_lap else 0.0
            if self.use_canonical_sds:
                reg_loss += regularizer.laplace_regularizer_const(mesh_obj_cano.v_pos, mesh_obj_cano.t_pos_idx) * self.w_reg_lap if self.enable_lap else 0.0
            reg_loss += regularizer.normal_consistency(mesh_hum.v_pos, mesh_hum.t_pos_idx) * self.w_reg_norm_smooth if self.enable_norm_smooth else 0.0

            # minimun surface loss
            if self.enable_canonical:
                reg_loss += (torch.exp(-100.0*torch.abs(sdf_hum_cano)).mean() + torch.exp(-100.0*torch.abs(sdf_obj_cano)).mean()) * self.w_reg_min_surface if self.enable_min_surface else 0.0

            # 3D semgmentation loss for preventing human shapes in object space
            # if self.enable_3d_segm and iteration >= self.FLAGS.init_refine_iter:
            #     if self.enable_canonical:
            #         occ = kaolin.ops.mesh.check_sign(mesh_hum_cano.v_pos[None].clone().detach(), mesh_hum_cano.t_pos_idx, mesh_obj_cano.v_pos[None]).float()
            #     else:
            #         occ = kaolin.ops.mesh.check_sign(mesh_hum.v_pos[None].clone().detach(), mesh_hum.t_pos_idx, mesh_obj.v_pos[None]).float()
            #     segm_loss_3d = occ.sum()
            #     reg_loss += segm_loss_3d * self.w_reg_3d

            # r1 reg for obj # seems better when not used for now. harms topology + doesn't remove artifacts inside object layer
            # sdf_weight = 0.3 - (0.3 - 0.01)*min(1.0, 4.0 * iteration/self.FLAGS.iter)
            # reg_loss += sdf_reg_loss(sdf_obj, self.all_edges).mean() * sdf_weight

            ################################################################################
            # 4. Zoom in head + hand for SDS
            ################################################################################


            ################################################################################
            # 5. Pose the canonical meshes into different pose if using multi-pose SDS
            ################################################################################
            if self.multi_pose:
                # deform cano hum mesh, cano obj mesh into desired pose
                sampled_tfs = self.refine_tfs[iteration % len(self.refine_poses)]
                mesh_hum_cano.v_pos = self.cano_to_posed(mesh_hum_cano.v_pos, sampled_tfs)
                mesh_hum_cano = mesh.auto_normals(mesh_hum_cano)
                mesh_obj_cano.v_pos = self.cano_to_posed(mesh_obj_cano.v_pos, sampled_tfs)
                mesh_obj_cano = mesh.auto_normals(mesh_obj_cano)
                mesh_comp_cano = mesh.combine(mesh_hum_cano, mesh_obj_cano)
                # update openpose
                self.op_3d_cano = self.refine_op3ds[iteration % len(self.refine_op3ds)]
        
            ################################################################################
            # 6. SDS loss for human
            ################################################################################

            # if sds used in both spaces, apply sds loss in each space one at a time
            if (not self.detach_hum or iteration % 2 == 0 and self.take_turns) or not self.use_canonical_sds:
                ### SDS loss human - no zoom
                # Choose buffers that are to be used for SDS loss for human
                buffers_sds_hum = self.choose_buffer('human', mesh_hum, mesh_hum_cano, glctx, target, lgt, opt_material, if_normal, mode, if_flip_the_normal,\
                                                     if_use_bump, zoom_in=None)
                if self.FLAGS.add_directional_text:
                    text_embeddings = torch.cat([guidance.uncond_z[target['prompt_index']], guidance.text_z[target['prompt_index']]]) # [B*2, 77, 1024]
                else:
                    text_embeddings = torch.cat([guidance.uncond_z, guidance.text_z])  # [B * 2, 77, 1024]
                sds_loss += self.apply_SDS(buffers=buffers_sds_hum, text_embeddings=text_embeddings, guidance=guidance, iteration=iteration) * self.w_sds   
                
                ### SDS loss human - zoom
                # Zoom in (if enabled) after the initialization refinement stage
                if self.enable_zoom_in and iteration >= self.FLAGS.init_refine_iter:
                    curr_zoom_in_params = {
                        'scale': self.zoom_in_params['scale_start'] + torch.rand_like(self.zoom_in_params['scale_start']) * (self.zoom_in_params['scale_end'] - self.zoom_in_params['scale_start']), # scale between no-zoom and scale-start
                        'transl': self.zoom_in_params['transl_start'] + torch.rand_like(self.zoom_in_params['transl_start']) * (self.zoom_in_params['transl_end'] - self.zoom_in_params['transl_start'])
                    }
                    buffers_sds_hum_zoom = self.choose_buffer('human', mesh_hum, mesh_hum_cano, glctx, target, lgt, opt_material, if_normal, mode, if_flip_the_normal,\
                                                            if_use_bump, zoom_in=curr_zoom_in_params)
                    sds_loss += self.apply_SDS(buffers=buffers_sds_hum_zoom, text_embeddings=text_embeddings, guidance=guidance, iteration=iteration) * self.w_sds   

                # SDS for hands
                if self.sds_for_hands and self.enable_zoom_in:
                    zoom_in_params_left_hand = {
                        'scale': torch.tensor(0.2).cuda(),
                        'transl': self.smplx_verts_cano[self.part_idx_dict['left_hand']].mean(axis=0).cuda()
                    }
                    zoom_in_params_right_hand = {
                        'scale': torch.tensor(0.2).cuda(),
                        'transl': self.smplx_verts_cano[self.part_idx_dict['right_hand']].mean(axis=0).cuda()
                    }

                    buffers_sds_hum_zoom = self.choose_buffer('human', mesh_hum, mesh_hum_cano, glctx, target, lgt, opt_material, if_normal, mode, if_flip_the_normal,\
                                                            if_use_bump, zoom_in=zoom_in_params_left_hand)
                    sds_loss += self.apply_SDS(buffers=buffers_sds_hum_zoom, text_embeddings=text_embeddings, guidance=guidance, iteration=iteration) * self.w_sds   

                    buffers_sds_hum_zoom = self.choose_buffer('human', mesh_hum, mesh_hum_cano, glctx, target, lgt, opt_material, if_normal, mode, if_flip_the_normal,\
                                                            if_use_bump, zoom_in=zoom_in_params_right_hand)
                    sds_loss += self.apply_SDS(buffers=buffers_sds_hum_zoom, text_embeddings=text_embeddings, guidance=guidance, iteration=iteration) * self.w_sds   



            ################################################################################
            # 7. SDS loss for object in composition space
            ################################################################################

            # if not self.use_canonical_sds : we don't need sds loss in comp space for object
            if (not self.detach_obj and iteration % 2 == 1 and self.take_turns) and self.use_canonical_sds:      
                # create composition mesh with detached human mesh and non-detached object mesh to enable changes only in the object space
                mesh_comp_sds = mesh.combine(mesh_hum, mesh_obj, detach_mesh1=self.take_turns)
                mesh_comp_sds.material = opt_material

                if self.enable_canonical:
                    mesh_comp_sds_cano = mesh.combine(mesh_hum_cano, mesh_obj_cano, detach_mesh1=self.take_turns)
                    mesh_comp_sds_cano.material = opt_material

                ### SDS loss object - no zoom
                # Choose buffers that are to be used for SDS loss in composition space for object 
                buffers_sds_comp = self.choose_buffer('comp', mesh_comp_sds, mesh_comp_sds_cano, glctx, target, lgt, opt_material, if_normal, mode, \
                                                       if_flip_the_normal, if_use_bump, zoom_in=None)
                if self.FLAGS.add_directional_text:
                    text_embeddings_comp = torch.cat([guidance.uncond_z_comp[target['prompt_index']], guidance.text_z_comp[target['prompt_index']]]) # [B*2, 77, 1024]
                else:
                    text_embeddings_comp = torch.cat([guidance.uncond_z_comp, guidance.text_z_comp])  # [B * 2, 77, 1024]
                
                sds_loss += self.apply_SDS(buffers=buffers_sds_comp, text_embeddings=text_embeddings_comp, guidance=guidance, iteration=iteration) * self.w_sds   
        
                 ### SDS loss object - zoom
                # Zoom in (if enabled) after the initialization refinement stage
                if self.enable_zoom_in and iteration >= self.FLAGS.init_refine_iter:
                    curr_zoom_in_params = {
                        'scale': self.zoom_in_params['scale_start'] + torch.rand_like(self.zoom_in_params['scale_start']) * (self.zoom_in_params['scale_end'] - self.zoom_in_params['scale_start']), # scale between no-zoom and scale-start
                        'transl': self.zoom_in_params['transl_start'] + torch.rand_like(self.zoom_in_params['transl_start']) * (self.zoom_in_params['transl_end'] - self.zoom_in_params['transl_start'])
                    }
                    buffers_sds_comp_zoom = self.choose_buffer('comp', mesh_comp_sds, mesh_comp_sds_cano, glctx, target, lgt, opt_material, if_normal, mode, \
                                                       if_flip_the_normal, if_use_bump, zoom_in=curr_zoom_in_params)
                    sds_loss += self.apply_SDS(buffers=buffers_sds_comp_zoom, text_embeddings=text_embeddings_comp, guidance=guidance, iteration=iteration) * self.w_sds   

        return sds_loss, img_loss, reg_loss
        

################################################################################
# 7. Reconstruction loss for zoomed in images for more details
################################################################################
# if self.subdivision and iteration > self.FLAGS.subdiv_iter:
#     zoom_in_params_body = {
#         'scale': torch.tensor(0.2).cuda(),
#         'transl': self.gt_smplx_verts[self.part_idx_dict['head']].mean(axis=0).cuda()
#     }
#     gt_buffers = render.render_mesh(glctx, self.gt_mesh.clone(), self.target_body_views['mvp'], self.target_body_views['campos'], lgt, self.target_body_views['resolution'], \
#                                     spp=self.target_body_views['spp'], msaa= True, background= self.target_body_views['background'],bsdf= None, if_normal= if_normal, \
#                                     zoom_in_params=zoom_in_params_body, normal_rotate= self.target_body_views['normal_rotate'],mode = mode, if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
#     gt_img = gt_buffers['shaded'][..., :3]
#     gt_mask = gt_buffers['shaded'][..., -1]
#     gt_seg = torch.cat([(gt_buffers['segm'][..., 0]==1).float()[..., None], 
#                         (gt_buffers['segm'][..., 0]==0).float()[..., None]], axis=-1)
    
#     gt_buffers_obj = render.render_mesh(glctx, self.gt_mesh_obj.clone(), self.target_body_views['mvp'], self.target_body_views['campos'], lgt, self.target_body_views['resolution'], \
#                                     spp=self.target_body_views['spp'], msaa= True, background= self.target_body_views['background'],bsdf= None, if_normal= if_normal,
#                                     zoom_in_params=zoom_in_params_body, normal_rotate= self.target_body_views['normal_rotate'],mode = mode, if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
#     gt_img_obj = gt_buffers_obj['shaded'][..., :3]
#     gt_mask_obj = gt_buffers_obj['shaded'][..., -1]

#     buffers_hum = render.render_mesh(glctx, mesh_hum, self.target_body_views['mvp'], self.target_body_views['campos'], lgt, self.target_body_views['resolution'], \
#                                     spp=self.target_body_views['spp'], msaa= True, background= self.target_body_views['background'],bsdf= None, if_normal= if_normal,
#                                     zoom_in_params=zoom_in_params_body, normal_rotate= self.target_body_views['normal_rotate'],mode = mode, if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
    
#     buffers_obj = render.render_mesh(glctx, mesh_obj, self.target_body_views['mvp'], self.target_body_views['campos'], lgt, self.target_body_views['resolution'], \
#                                     spp=self.target_body_views['spp'], msaa= True, background= self.target_body_views['background'],bsdf= None, if_normal= if_normal,
#                                     zoom_in_params=zoom_in_params_body, normal_rotate= self.target_body_views['normal_rotate'],mode = mode, if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
    
#     buffers_comp = render.render_mesh(glctx, mesh_comp, self.target_body_views['mvp'], self.target_body_views['campos'], lgt, self.target_body_views['resolution'], \
#                                     spp=self.target_body_views['spp'], msaa= True, background= self.target_body_views['background'],bsdf= None, if_normal= if_normal,
#                                     zoom_in_params=zoom_in_params_body, normal_rotate= self.target_body_views['normal_rotate'],mode = mode, if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
    
#     # reconstruction loss in human posed space
#     img_hum = buffers_hum['shaded'][..., :3]
#     mask_hum = buffers_hum['shaded'][..., -1]
#     gt_seg_hum = gt_seg[..., 1] if not self.dilate_seg_mask else -self.dilate_seg(-gt_seg[..., 1])

#     if (not self.detach_hum or iteration % 2 == 0 and self.take_turns) or not self.use_canonical_sds:
#         reg_loss += loss_fn(img_hum * mask_hum.unsqueeze(-1) * gt_seg_hum.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_hum.unsqueeze(-1)) * self.w_recon_full

#     # reconstruction loss in object posed space
#     img_obj = buffers_obj['shaded'][..., :3]
#     mask_obj = buffers_obj['shaded'][..., -1]
#     gt_seg_obj = gt_seg[..., 0] if not self.dilate_seg_mask else self.dilate_seg(gt_seg[..., 0])

#     if (not self.detach_obj and iteration % 2 == 1 and self.take_turns):
#         if self.is_obj_most_outer:
#             reg_loss += (loss_fn(mask_obj, gt_mask_obj) * 10 + loss_fn(img_obj * mask_obj.unsqueeze(-1), gt_img_obj * gt_mask_obj.unsqueeze(-1))) * self.w_recon_full
#         else:
#             reg_loss += loss_fn(img_obj * mask_obj.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1)) * self.w_recon_full 

#     # reconstruction loss in comp posed space
#     img_comp = buffers_comp['shaded'][..., :3]
#     mask_comp = buffers_comp['shaded'][..., -1]
#     segm_comp = buffers_comp['segm'][..., :2]

#     mask_loss = loss_fn(mask_comp, gt_mask) # prevents floaters -> replace with min surface loss
#     segm_loss = loss_fn(segm_comp * mask_comp.unsqueeze(-1), gt_seg * gt_mask.unsqueeze(-1))  # segm loss should be larger than mask loss to keep hum and obj separated
#     # norm_loss = loss_fn(img_comp * mask_comp.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1)) # no normal loss

#     reg_loss += (segm_loss * self.w_recon_segm) # segm loss should be large enough

################################################################################
# 7. Reconstruction loss for zoomed in images for more details
################################################################################
# Render gts for zoomed in images if enabled. Zoom in after the initialization refinement stage
# if self.enable_zoom_in and iteration >= self.FLAGS.init_refine_iter:
#     # gradual zoom in if enabled
#     if iteration % 2 == 0:
#         curr_zoom_in_params = {
#             'scale': 1, # scale between no-zoom and scale-start
#             'transl': 0 # translate between 0 and transl
#         }
#     else:
#         curr_zoom_in_params = {
#             'scale': self.zoom_in_params['scale_start'],
#             'transl': self.zoom_in_params['transl'] # translate between 0 and transl
#         }

#     gt_buffers_zoom = render.render_mesh(glctx, self.gt_mesh.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
#                                 background= target['background'],bsdf= None, if_normal=if_normal, normal_rotate= target['normal_rotate'],mode = mode,
#                                 if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, zoom_in_params=curr_zoom_in_params)               
#     gt_img = gt_buffers_zoom['shaded'][..., :3]
#     gt_mask = gt_buffers_zoom['shaded'][..., -1]
#     gt_seg = torch.cat([(gt_buffers_zoom['segm'][..., 0]==1).float()[..., None], 
#                         (gt_buffers_zoom['segm'][..., 0]==0).float()[..., None]], axis=-1)

#     # composition space segmentation loss with the zoomed in images
#     buffers_comp_zoom = self.render('comp', glctx, target, lgt, opt_material, opt_mesh=mesh_comp, if_normal=if_normal, zoom_in_params=curr_zoom_in_params,
#                         mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)

#     img_comp_zoom = buffers_comp_zoom['shaded'][..., :3]
#     mask_comp_zoom = buffers_comp_zoom['shaded'][..., -1]
#     segm_comp_zoom = buffers_comp_zoom['segm'][..., :2]
    
#     segm_loss_zoom = loss_fn(segm_comp_zoom * mask_comp_zoom.unsqueeze(-1), gt_seg * gt_mask.unsqueeze(-1))
#     reg_loss += (segm_loss_zoom * self.w_recon_segm) # segm loss should be large enough

#     # object recon loss with the zoomed in images if enabled
#     if self.is_obj_most_outer:
#         if self.gt_mesh_obj is None:
#             gt_mesh_obj = scene_and_vertices[6]
#             gt_mesh_obj.material = opt_material
#             gt_mesh_obj = mesh.auto_normals(gt_mesh_obj)
#             self.gt_mesh_obj = gt_mesh_obj
#         gt_buffers_obj_zoom = render.render_mesh(glctx, self.gt_mesh_obj.clone(), target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
#                                         background= target['background'],bsdf= None, if_normal= if_normal,normal_rotate= target['normal_rotate'],mode = mode,
#                                         if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=curr_zoom_in_params)
        
#         buffers_obj_zoom = self.render('obj', glctx, target, lgt, opt_material, opt_mesh=mesh_obj, if_normal= if_normal, mode = mode, 
#                                     if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=curr_zoom_in_params)
    
#         gt_img_obj_zoom = gt_buffers_obj_zoom['shaded'][..., :3]
#         gt_mask_obj_zoom = gt_buffers_obj_zoom['shaded'][..., -1]

#         img_obj_zoom = buffers_obj_zoom['shaded'][..., :3]
#         mask_obj_zoom = buffers_obj_zoom['shaded'][..., -1]
    
#         mask_loss_zoom = loss_fn(mask_obj_zoom, gt_mask_obj_zoom)
#         norm_loss_zoom = loss_fn(img_obj_zoom * mask_obj_zoom.unsqueeze(-1), gt_img_obj_zoom * gt_mask_obj_zoom.unsqueeze(-1))

#         # recon loss
#         reg_loss += (mask_loss_zoom + norm_loss_zoom) * self.w_recon_full



################################################################################
# 7. Reconstruction loss to regularize SDS loss for human
################################################################################

# render human images for regularizing SDS loss with reconstruction loss
# img_hum_recon = buffers_recon_hum['shaded'][..., :3]
# mask_hum_recon = buffers_recon_hum['shaded'][..., -1]
# gt_seg_hum_recon = gt_seg[..., 1] if not self.dilate_seg_mask else -self.dilate_seg(-gt_seg[..., 1])

# reg_loss += loss_fn(img_hum_recon * mask_hum_recon.unsqueeze(-1) * gt_seg_hum_recon.unsqueeze(-1), \
#                     gt_img * gt_mask.unsqueeze(-1) * gt_seg_hum_recon.unsqueeze(-1)) * self.w_sds_reg


################################################################################
# 10. Reconstruction loss to regularize SDS loss in composition space for human and object
################################################################################

# img_obj_recon = buffers_recon_obj['shaded'][..., :3]
# mask_obj_recon = buffers_recon_obj['shaded'][..., -1]
# gt_seg_obj_recon = gt_seg[..., 0] if not self.dilate_seg_mask else self.dilate_seg(gt_seg[..., 0])
# reg_loss += loss_fn(img_obj_recon * mask_obj_recon.unsqueeze(-1) * gt_seg_obj_recon.unsqueeze(-1), \
#                     gt_img * gt_mask.unsqueeze(-1) * gt_seg_obj_recon.unsqueeze(-1)) * self.w_sds_reg


################################################################################
# 3. Apply face segm infos to mesh_hum and mesh_obj to get seg masks in canonical space for inpainting
################################################################################
# if self.use_canonical_sds:
#     if self.gt_segs_posed is None:
#         target_segms = {'mv': [], 'mvp': [], 'campos': [], 'background': [], 'normal_rotate': []}
#         for i in range(18):
#             target_segms['mv'].append(DatasetMesh.train_scene_segm(i)['mv'])
#             target_segms['mvp'].append(DatasetMesh.train_scene_segm(i)['mvp'])
#             target_segms['campos'].append(DatasetMesh.train_scene_segm(i)['campos'])
#             target_segms['background'].append(DatasetMesh.train_scene_segm(i)['background'])
#             target_segms['normal_rotate'].append(DatasetMesh.train_scene_segm(i)['normal_rotate'])
#         self.target_segms_dict = {'mv': torch.cat(target_segms['mv'], dim=0), 'mvp': torch.cat(target_segms['mvp'], dim=0), 'campos': torch.cat(target_segms['campos'], dim=0), \
#                             'background': torch.cat(target_segms['background'], dim=0), 'normal_rotate': torch.cat(target_segms['normal_rotate'], dim=0),
#                             'resolution' : torch.tensor([512, 512]), 'spp' : 1, 'prompt_index' : 0}

#         gt_buffers_for_cano_mask = render.render_mesh(glctx, self.gt_mesh.clone(), self.target_segms_dict['mvp'], self.target_segms_dict['campos'], lgt, self.target_segms_dict['resolution'], spp=self.target_segms_dict['spp'], msaa= True,
#                                     background= self.target_segms_dict['background'],bsdf= None, if_normal= if_normal,normal_rotate= self.target_segms_dict['normal_rotate'],mode = mode,
#                                     if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump)
#         gt_seg_for_cano_mask = torch.cat([(gt_buffers_for_cano_mask['segm'][..., 0]==1).float()[..., None], 
#                         (gt_buffers_for_cano_mask['segm'][..., 0]==0).float()[..., None]], axis=-1)
#         self.gt_masks_posed = gt_buffers_for_cano_mask['shaded'][..., -1]
#         self.gt_segs_posed = gt_seg_for_cano_mask

    
#     rast_hum = render.render_mesh(glctx, mesh_hum, self.target_segms_dict['mvp'], self.target_segms_dict['campos'], lgt, self.target_segms_dict['resolution'], spp=self.target_segms_dict['spp'], msaa= True,
#                                     background= self.target_segms_dict['background'],bsdf= None, if_normal= if_normal,normal_rotate= self.target_segms_dict['normal_rotate'],mode = mode,
#                                     if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, return_rast=True)
    

#     mesh_comp_vis = mesh_comp.clone()
#     mesh_comp_vis.f_seg = None
#     mesh_comp_vis_cano = mesh_comp_cano.clone()
#     mesh_comp_vis_cano.f_seg = None
#     rast_comp_vis = render.render_mesh(glctx, mesh_comp_vis, self.target_segms_dict['mvp'], self.target_segms_dict['campos'], lgt, self.target_segms_dict['resolution'], spp=self.target_segms_dict['spp'], msaa= True,
#                                     background= self.target_segms_dict['background'],bsdf= None, if_normal= if_normal,normal_rotate= self.target_segms_dict['normal_rotate'],mode = mode,
#                                     if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, return_rast=True)
    
#     is_human = -self.dilate_seg(-self.gt_segs_posed[..., 1]) * self.gt_masks_posed
#     is_obj = self.dilate_seg(self.gt_segs_posed[..., 0]) * self.gt_masks_posed
#     is_mask = self.gt_masks_posed
    
#     # for the human mesh, face_seg_label == 1 means the region is not considered as human
#     hum_seg = torch.ones((mesh_hum.t_pos_idx.shape[0], 1), device='cuda')
#     hum_seg[torch.unique(rast_hum * is_human).long() - 1] = 0

#     # hum_seg = torch.zeros((mesh_hum.t_pos_idx.shape[0], 1), device='cuda')
#     # hum_seg[torch.unique(rast_hum * is_obj).long() - 1] = 1
#     mesh_hum.f_seg = hum_seg.clone().detach()
#     mesh_hum_cano.f_seg = hum_seg.clone().detach()

#     comp_vis = torch.zeros((mesh_comp_vis.t_pos_idx.shape[0], 1), device='cuda')
#     comp_vis[torch.unique(rast_comp_vis * is_mask).long() - 1] = 1
#     mesh_comp_vis.f_seg = comp_vis.clone().detach()
#     mesh_comp_vis_cano.f_seg = comp_vis.clone().detach()

#     # for the composition mesh, segm_comp_vis == 1 means it's visible from posed scan
#     buffers_comp_vis = self.render('comp', glctx, target, lgt, opt_material, opt_mesh=mesh_comp_vis, if_normal=if_normal, 
#                            mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump)
#     segm_comp_vis = buffers_comp_vis['segm']
#     buffers_comp_vis_cano = self.render('comp', glctx, target, lgt, opt_material, opt_mesh=mesh_comp_vis_cano, if_normal=if_normal, 
#                            mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump)
#     segm_comp_vis_cano = buffers_comp_vis_cano['segm']
