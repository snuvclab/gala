# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import pickle

from render import mesh
from render import render
from render import mlptexture
from dataset.dataset_mesh import DatasetMesh

from torch.cuda.amp import custom_bwd, custom_fwd 
import numpy as np
from pytorch3d.io import load_obj
from deformer.lib import rotation_converter, helpers
from deformer.smplx import SMPLX
from pytorch3d.ops import knn_points
from render.regularizer import avg_edge_length, laplace_regularizer_const, normal_consistency

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
    
###############################################################################
#  Geometry interface
###############################################################################
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class DLMesh(torch.nn.Module):
    def __init__(self, initial_guess, initial_guess_obj, gt_mesh, FLAGS):
        super(DLMesh, self).__init__()

        self.FLAGS = FLAGS

        self.mesh = initial_guess
        self.mesh_obj = initial_guess_obj
        self.mesh_gt = gt_mesh

        self.mesh_detach = initial_guess.clone()
        self.mesh_obj_detach = initial_guess_obj.clone()
        self.mesh_gt_detach = gt_mesh.clone()
        print("Base human mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))
        print("Base object mesh has %d triangles and %d vertices." % (self.mesh_obj.t_pos_idx.shape[0], self.mesh_obj.v_pos.shape[0]))
        print("GT mesh has %d triangles and %d vertices." % (self.mesh_gt.t_pos_idx.shape[0], self.mesh_gt.v_pos.shape[0]))
        
        # self.mesh.v_pos = torch.nn.Parameter(self.mesh.v_pos, requires_grad= True)
        # self.register_parameter('mesh_vertex', self.mesh.v_pos)
        # self.register_parameter('mesh_obj_vertex', self.mesh_obj.v_pos)

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

        self.use_op_control = self.FLAGS.enable_controlnet and self.FLAGS.controlnet_type == 'op'
        self.op_3d = None
        if self.use_op_control:
            self.op_3d_mapping = np.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 75, 82, 106, 125], dtype=np.int32)
            self.op_3d = self.gt_joints[self.op_3d_mapping] # get from gt SMPLX joints

        self.enable_canonical = FLAGS.enable_canonical
        self.use_canonical_sds = self.enable_canonical and FLAGS.use_canonical_sds
        self.multi_pose = self.enable_canonical and FLAGS.multi_pose
        if self.enable_canonical:
            # precompute lbs grid
            d, h, w = 64, 256, 256
            grid = helpers.create_voxel_grid(d, h, w, device='cuda')
            shape_pose_offsets_grid = helpers.query_weights_smpl(grid, self.smplx_verts_cano.cuda(), self.gt_smplx_offsets[0].cuda()).permute(0,2,1).reshape(1,-1,d,h,w)
            lbs_weights_grid = helpers.query_weights_smpl(grid, self.smplx_verts_cano.cuda(), self.smplx.lbs_weights.cuda()).permute(0,2,1).reshape(1,-1,d,h,w)
            
            v_pos = self.mesh.v_pos
            offsets = torch.nn.functional.grid_sample(shape_pose_offsets_grid , v_pos[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
            lbs_weights = torch.nn.functional.grid_sample(lbs_weights_grid, v_pos[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
            v_pos = v_pos + offsets[0]
            v_pos = helpers.skinning(v_pos, lbs_weights, self.gt_smplx_tfs)[0]

            v_pos_obj = self.mesh_obj.v_pos
            offsets = torch.nn.functional.grid_sample(shape_pose_offsets_grid , v_pos_obj[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
            lbs_weights = torch.nn.functional.grid_sample(lbs_weights_grid, v_pos_obj[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
            v_pos_obj = v_pos_obj + offsets[0]
            v_pos_obj = helpers.skinning(v_pos_obj, lbs_weights, self.gt_smplx_tfs)[0]

            # fixed posed shape
            self.mesh_posed = mesh.Mesh(base=self.mesh, v_pos=v_pos)
            self.mesh_obj_posed = mesh.Mesh(base=self.mesh_obj, v_pos=v_pos_obj)
            
            if self.use_op_control:
                self.op_3d_cano = self.canonical_joints[self.op_3d_mapping] # get from canonical SMPLX joints
        
        # zoom in
        self.enable_zoom_in = self.FLAGS.enable_zoom_in
        self.zoom_in_params = None
        self.zoom_in_body = self.FLAGS.zoom_in_body
        self.zoom_in_head_hands = self.FLAGS.zoom_in_head_hands

        # optimization
        self.take_turns = self.FLAGS.take_turns
        self.detach_hum = self.FLAGS.detach_hum
        self.detach_obj = self.FLAGS.detach_obj

        self.dilate_seg_mask = self.FLAGS.dilate_seg_mask
        if self.dilate_seg_mask:
            self.dilation_size = 9
            self.dilate_seg = torch.nn.MaxPool2d(kernel_size=self.dilation_size, stride=1, padding=self.dilation_size // 2)

        self.net = mlptexture.MLPTexture3D(self.getAABB(), channels=3, min_max=[0, 1]) # min_max not used
        self.net_obj = mlptexture.MLPTexture3D(self.getAABBObj(), channels=3, min_max=[0, 1])

        self.tick_count = 0

        # refine
        if self.FLAGS.mode == 'refine':
            
            self.initial_hum_norm = self.mesh.v_nrm.clone().detach()
            self.initial_obj_norm = self.mesh_obj.v_nrm.clone().detach()
            dists = knn_points(self.mesh.v_pos[None], self.mesh_obj.v_pos[None]).dists[0]
            disp_norm_hum = -torch.rand_like(self.mesh.v_pos.clone().detach())[:,-1:] * 1e-12
            disp_norm_hum[dists > 0.05] = 0
            # disp_norm_obj = torch.rand_like(self.mesh_obj.v_pos.clone().detach())[:,-1:] * 0.000001
            self.disp_norm_hum = torch.nn.Parameter(disp_norm_hum.clone().detach(), requires_grad=True)
            # self.disp_norm_obj = torch.nn.Parameter(disp_norm_obj.clone().detach(), requires_grad=True)
            self.register_parameter('disp_norm_hum', self.disp_norm_hum)
            # self.register_parameter('disp_norm_obj', self.disp_norm_obj)
            self.visible_verts = None

            self.mesh_comp = None
            self.mesh_comp_posed = None
            self.dist_hum2obj = None
            self.dist_obj2hum = None
            self.shape_pose_offsets_grid = helpers.query_weights_smpl(grid, self.smplx_verts_cano.cuda(), self.gt_smplx_offsets[0].cuda()).permute(0,2,1).reshape(1,-1,d,h,w)
            self.lbs_weights_grid = helpers.query_weights_smpl(grid, self.smplx_verts_cano.cuda(), self.smplx.lbs_weights.cuda()).permute(0,2,1).reshape(1,-1,d,h,w)

            self.w_mask = FLAGS.w_mask
            self.w_reg_offset = FLAGS.w_reg_offset
            self.w_reg_edge = FLAGS.w_reg_edge
            self.w_reg_lap = FLAGS.w_reg_lap
            self.w_reg_norm_smooth = FLAGS.w_reg_norm_smooth

        # loss weights
        self.w_recon_rgb = FLAGS.w_recon_rgb
        self.w_sds = FLAGS.w_sds
        
        # whether second++ decomposition
        self.layer = self.FLAGS.layer

        self.rig = self.FLAGS.rig
        if self.rig: self.op_3d = self.canonical_joints[self.op_3d_mapping]


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

        # set canonical space
        self.smplx = SMPLX(self.smplx_config)
        pose = torch.zeros([55,3], dtype=torch.float32, ) # 55
        pose_axispca = torch.zeros([29,3], dtype=torch.float32, )
        angle = 15*np.pi/180.
        pose[1, 2] = angle
        pose[2, 2] = -angle
        # pose_axispca[1, 2] = angle
        # pose_axispca[2, 2] = -angle
        pose_euler = pose.clone()
        pose = rotation_converter.batch_euler2matrix(pose)
        pose = pose[None,...]
        xyz_c, _, joints_c, A, T, shape_offsets, pose_offsets = self.smplx(full_pose = pose, return_T=True, transl=torch.tensor([0, 0.4, 0],dtype=torch.float32, ))
        A_inv = A.squeeze(0).inverse()
        self.A_inv = A_inv
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

        # posed smplx
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
        self.gt_exp = exp
        self.gt_smplx_verts = xyz.squeeze()
        self.gt_joints = joints.squeeze()
        self.gt_smplx_tfs = torch.einsum('bnij,njk->bnik', A, A_inv).cuda()
        self.gt_smplx_offsets = shape_offsets + pose_offsets
        self.gt_smplx_mesh = mesh.Mesh(self.gt_smplx_verts.cuda(), self.smplx_faces.cuda())
        self.smplx_mesh_cano = mesh.Mesh(self.smplx_verts_cano.cuda(), self.smplx_faces.cuda())

    # transforms verts in canonical space to posed space using NN smplx vertex (incorporates shape param offsets)
    def cano_to_posed(self, verts, tfs):
        offsets = torch.nn.functional.grid_sample(self.shape_pose_offsets_grid , verts[None, None, None, :, :]).reshape(1, 3, -1).permute(0, 2, 1)
        lbs_weights = torch.nn.functional.grid_sample(self.lbs_weights_grid, verts[None, None, None, :, :]).reshape(1, 55, -1).permute(0, 2, 1)
        verts = verts + offsets[0]  # shape pose blend shapes
        verts = helpers.skinning(verts, lbs_weights, tfs)[0]  # lbs
        return verts
    
    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)
    
    @torch.no_grad()
    def getAABBObj(self):
        return mesh.aabb(self.mesh_obj)
    
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
    
    def getMesh(self, material, mesh_type='comp', detach_hum=False, detach_obj=False, canonical=False):
        '''
            mesh_type: [human, obj, comp]
        '''
        
        if mesh_type == 'human':
            self.mesh.material = self.material
            vertex_color = self.net.sample(self.mesh.v_pos) # query color for every vertex
            self.mesh.v_color = vertex_color

            imesh_hum = mesh.Mesh(base=self.mesh)
            imesh_hum = mesh.auto_normals(imesh_hum)
            
            if self.enable_canonical:
                self.mesh_posed.material = self.material
                self.mesh_posed.v_color = vertex_color

            if self.enable_canonical and not canonical:
                imesh_hum_posed = mesh.Mesh(base=self.mesh_posed)
                imesh_hum_posed = mesh.auto_normals(imesh_hum_posed)

        elif mesh_type == 'obj':
            self.mesh_obj.material = self.material
            vertex_color = self.net_obj.sample(self.mesh_obj.v_pos) # query color for every vertex
            self.mesh_obj.v_color = vertex_color

            imesh_obj = mesh.Mesh(base=self.mesh_obj)
            imesh_obj = mesh.auto_normals(imesh_obj)
            
            if self.enable_canonical:
                self.mesh_obj_posed.material = self.material
                self.mesh_obj_posed.v_color = vertex_color

            if self.enable_canonical and not canonical:
                imesh_obj_posed = mesh.Mesh(base=self.mesh_obj_posed)
                imesh_obj_posed = mesh.auto_normals(imesh_obj_posed)

        elif mesh_type == 'comp':
            self.mesh.material = self.material
            self.mesh_obj.material = self.material
            vertex_color = self.net.sample(self.mesh.v_pos)
            vertex_color_obj = self.net_obj.sample(self.mesh_obj.v_pos)
            self.mesh.v_color = vertex_color
            self.mesh_obj.v_color = vertex_color_obj

            imesh_comp = mesh.combine_texture(self.mesh, self.mesh_obj, vertex_color, vertex_color_obj, self.material, detach_mesh1=detach_hum)
            self.mesh_comp = imesh_comp

            if self.enable_canonical:
                self.mesh_posed.v_color = vertex_color
                self.mesh_obj_posed.material = self.material
                self.mesh_posed.material = self.material
                self.mesh_obj_posed.v_color = vertex_color_obj

            if self.enable_canonical and not canonical:
                imesh_comp_posed = mesh.combine_texture(self.mesh_posed, self.mesh_obj_posed, vertex_color, vertex_color_obj, self.material, detach_mesh1=detach_hum)
                self.mesh_comp_posed = imesh_comp_posed
            
        else:
            raise NotImplementedError
        
        mesh_dict = {}
        if self.enable_canonical and canonical:
            if mesh_type == 'comp': 
                mesh_dict['mesh_comp_cano'] = imesh_comp
            elif mesh_type == 'human': 
                mesh_dict['mesh_hum_cano'] = imesh_hum
            elif mesh_type == 'obj': 
                mesh_dict['mesh_obj_cano'] = imesh_obj
        elif self.enable_canonical and not canonical:
            if mesh_type == 'comp': 
                mesh_dict['mesh_comp_posed'] = imesh_comp_posed
                mesh_dict['mesh_comp_cano'] = imesh_comp
            elif mesh_type == 'human': 
                mesh_dict['mesh_hum_posed'] = imesh_hum_posed
                mesh_dict['mesh_hum_cano'] = imesh_hum
            elif mesh_type == 'obj': 
                mesh_dict['mesh_obj_posed'] = imesh_obj_posed
                mesh_dict['mesh_obj_cano'] = imesh_obj
        
        else:  # no canonical modeling
            if mesh_type == 'comp': 
                mesh_dict['mesh_comp_posed'] = imesh_comp
            elif mesh_type == 'human': 
                mesh_dict['mesh_hum_posed'] = imesh_hum
            elif mesh_type == 'obj': 
                mesh_dict['mesh_obj_posed'] = imesh_obj

        return mesh_dict

        # if self.enable_canonical and not canonical:
        #     if mesh_type == 'comp': return imesh_posed, imesh
        #     elif mesh_type == 'human': return imesh_posed, imesh
        #     elif mesh_type == 'obj': return imesh_posed, imesh
        # else:
        #     if mesh_type == 'comp': return imesh
        #     elif mesh_type == 'human': return imesh
        #     elif mesh_type == 'obj': return imesh

    def getMeshRefine(self, material, mesh_type='comp', detach_hum=False, detach_obj=False, canonical=False):
        '''
            mesh_type: [human, obj, comp]
        '''
        
        if mesh_type == 'human':
            self.mesh.material = self.mesh_gt.material

            imesh = mesh.Mesh(base=self.mesh)
            imesh = mesh.auto_normals(imesh)
            
            if self.enable_canonical:
                self.mesh_posed.material = self.mesh_gt.material
                self.mesh_posed.v_color = self.mesh.v_color

            if self.enable_canonical and not canonical:
                imesh_posed = mesh.Mesh(base=self.mesh_posed)
                imesh_posed = mesh.auto_normals(imesh_posed)

        elif mesh_type == 'obj':
            self.mesh_obj.material = self.mesh_gt.material

            imesh = mesh.Mesh(base=self.mesh_obj)
            imesh = mesh.auto_normals(imesh)
            
            if self.enable_canonical:
                self.mesh_obj_posed.material = self.mesh_gt.material
                self.mesh_obj_posed.v_color = self.mesh_obj.v_color

            if self.enable_canonical and not canonical:
                imesh_posed = mesh.Mesh(base=self.mesh_obj_posed)
                imesh_posed = mesh.auto_normals(imesh_posed)

        elif mesh_type == 'comp':
            self.mesh.material = self.mesh_gt.material
            self.mesh_obj.material = self.mesh_gt.material
            if self.mesh_comp is None:
                imesh = mesh.combine_texture(self.mesh, self.mesh_obj, self.mesh.v_color, self.mesh_obj.v_color, self.mesh_gt.material, detach_mesh1=detach_hum)
            else:
                imesh = self.mesh_comp
                imesh = mesh.auto_normals(imesh)

            # if self.enable_canonical:
            #     self.mesh_posed.v_color = self.mesh.v_color
            #     self.mesh_posed.material = self.mesh_gt.material
            #     self.mesh_obj_posed.v_color = self.mesh_obj.v_color
            #     self.mesh_obj_posed.material = self.mesh_gt.material

            if self.enable_canonical and not canonical:
                verts = self.cano_to_posed(imesh.v_pos, self.gt_smplx_tfs)
                imesh_posed = mesh.Mesh(verts, imesh.t_pos_idx, v_color=imesh.v_color, material=material)
                imesh_posed = mesh.auto_normals(imesh_posed)
                self.mesh_comp_posed = imesh_posed
            
        else:
            raise NotImplementedError

        mesh_dict = {}
        if self.enable_canonical and canonical:
            if mesh_type == 'comp':
                mesh_dict['mesh_comp_cano'] = imesh
            elif mesh_type == 'human':
                mesh_dict['mesh_hum_cano'] = imesh
            elif mesh_type == 'obj':
                mesh_dict['mesh_obj_cano'] = imesh
        elif self.enable_canonical and not canonical:
            if mesh_type == 'comp': 
                mesh_dict['mesh_comp_cano'] = imesh
                mesh_dict['mesh_comp_posed'] = imesh_posed
            elif mesh_type == 'human': 
                mesh_dict['mesh_hum_cano'] = imesh
                mesh_dict['mesh_hum_posed'] = imesh_posed
            elif mesh_type == 'obj': 
                mesh_dict['mesh_obj_cano'] = imesh
                mesh_dict['mesh_obj_posed'] = imesh_posed
        else:
            if mesh_type == 'comp':
                mesh_dict['mesh_comp_posed'] = imesh
            elif mesh_type == 'human':
                mesh_dict['mesh_hum_posed'] = imesh
            elif mesh_type == 'obj':
                mesh_dict['mesh_obj_posed'] = imesh

        return mesh_dict

    # @profile
    def render(self, mesh_type, glctx, target, lgt, opt_material, opt_mesh=None, bsdf=None, if_normal=False, mode='appearance_modeling', 
               if_flip_the_normal = False, if_use_bump = False, zoom_in_params=None, detach_obj=False, render_control_op=False, canonical=False, return_rast=False, refine=False):
        if opt_mesh is None:
            mesh_dict = self.getMesh(opt_material, mesh_type=mesh_type, detach_obj=detach_obj, canonical=canonical)
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
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], msaa= True,
                                  background= target['background'], bsdf= bsdf, if_normal= if_normal, normal_rotate= target['normal_rotate'], mode = mode,
                                  if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params, 
                                  op_3d = self.op_3d if render_control_op else None, return_rast=return_rast)
    
    def getMeshGT(self):
        imesh = mesh.Mesh(base=self.mesh_gt)
        imesh = mesh.auto_normals(imesh)
        if not self.layer: imesh = mesh.compute_tangents(imesh)
        return imesh
    
    def renderGT(self, glctx, target, lgt, opt_material, bsdf=None,if_normal=False, mode = 'appearance_modeling', if_flip_the_normal = False, if_use_bump = False, zoom_in_params=None):
        opt_mesh = self.getMeshGT()
        if self.layer: opt_mesh.material = opt_material
        return render.render_mesh(glctx, opt_mesh,target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'],  msaa=True, background= target['background'] ,
                                  bsdf= bsdf, if_normal=if_normal, normal_rotate=target['normal_rotate'], mode = mode, if_flip_the_normal = if_flip_the_normal,
                                  if_use_bump = if_use_bump, zoom_in_params=zoom_in_params)

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
            srgb =  buffers['v_color'][...,0:3]
            mask_hum = buffers['v_color'][..., -1]
            # srgb = util.rgb_to_srgb(srgb)
            # t = torch.randint( guidance.min_step_early, guidance.max_step_early, [1], dtype=torch.long, device='cuda')
            t = torch.randint( guidance.min_step_early, guidance.max_step_early+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]
        else:
            srgb =   buffers['v_color'][...,0:3]
            mask_hum = buffers['v_color'][..., -1]
            # srgb = util.rgb_to_srgb(srgb)
            # t = torch.randint(guidance.min_step_late, guidance.max_step_late, [1], dtype=torch.long, device='cuda')
            t = torch.randint( guidance.min_step_late, guidance.max_step_late+1, [self.FLAGS.batch], dtype=torch.long, device='cuda') # [B]

        pred_rgb_512 = srgb.permute(0, 3, 1, 2).contiguous().to(torch.float16) # [1, 3, H, W]
        if guidance.use_legacy:
            latents = guidance.encode_imgs(pred_rgb_512)

        # prepare inpaint mask
        mask_inpaint = None
        # latents_masked = None
        if guidance.use_inpaint:
            if self.use_canonical_sds:
                # hum_inpaint_mask
                mask_inpaint = (buffers['segm'][..., 0]) * buffers['shaded'][..., -1]
            else:
                # hum_inpaint_mask
                mask_inpaint = -self.dilate_seg(-gt_seg[..., 1]) if self.dilate_seg_mask else gt_seg[..., 1]
                mask_inpaint = 1 - mask_inpaint
            
        # prepare control images
        control_images = None
        if guidance.enable_controlnet:
            if guidance.control_type == 'op':
                control_images = buffers['op_img']
            else:
                raise NotImplementedError
                            
        # # test strength example : int(t[0] / guidance.num_train_timesteps)
        # num_inference_steps_test = 50
        # strength = t[0] / guidance.num_train_timesteps
        # image_pil = guidance.test_diffusion(latents, noise, num_inference_steps_test, text_embeddings, strength=strength,
        #                                     mask=mask_inpaint, latents_masked=latents_masked, control_images=control_images)
        # image_pil[0].show()

        # # diffusers 
        # strength = t[0] / guidance.num_train_timesteps
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        # )
        # pipe = pipe.to("cuda")
        # mask_inpaint_resized = torch.nn.functional.interpolate(mask_inpaint, size=latents.shape[-2:])
        # negative_prompt_embeds, prompt_embeds = text_embeddings.chunk(2)
    
        # # these works well
        # from_diffusers = pipe(image=pred_rgb_512, strength=strength, mask_image=mask_inpaint, 
        #                     prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images

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
                                image=torch.clamp(pred_rgb_512, 0, 1), 
                                mask_image=mask_inpaint, 
                                prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                control_image=control_images)

        # w = (1 - guidance.alphas[t]) # [B]
        # w = self.guidance.alphas[t]
        # w = 1 / (1 - guidance.alphas[t])
        # w = 1 / torch.sqrt(1 - guidance.alphas[t])
        if guidance.sds_weight_strategy == 0:
            w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
        elif guidance.sds_weight_strategy == 1:
            # w = 1 / torch.sqrt(1 - guidance.alphas[t])
            # w = (1 - guidance.alphas[t]) 
            w = 1 / (1 - guidance.alphas[t])
        elif guidance.sds_weight_strategy == 2:
            if iteration <= self.FLAGS.coarse_iter:
                w = guidance.alphas[t] ** 0.5 * (1 - guidance.alphas[t])
            else:
                w = 1 / (1 - guidance.alphas[t])
                
        w = w[:, None, None, None] # [B, 1, 1, 1]
        if guidance.use_inpaint and guidance.repaint:
            mask_interp = torch.nn.functional.interpolate(1 - mask_inpaint, size=latents.shape[-2:])  # B x 1 x H x W
            noise_pred = noise_pred * (1 - mask_interp) + noise * mask_interp
        grad =  w * (noise_pred - noise) #*w1 s
        grad = torch.nan_to_num(grad)
        return SpecifyGradient.apply(latents, grad)
    

    def tick(self, glctx, target, lgt, opt_material, iteration, if_normal, guidance, scene_and_vertices, mode, if_flip_the_normal, if_use_bump):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        self.tick_count += 1
        sds_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        loss_fn = torch.nn.MSELoss()
        self.material = opt_material

        # color reconstruction
        gt_buffers = self.renderGT(glctx, target, lgt, opt_material, if_normal=if_normal, mode=mode, 
                                   if_flip_the_normal=if_flip_the_normal, if_use_bump = if_use_bump)
        gt_img = gt_buffers['shaded'][..., :3]

        if self.layer: gt_img = gt_buffers['v_color'][..., :3]

        gt_mask = gt_buffers['shaded'][..., -1]
        gt_seg = torch.cat([(gt_buffers['segm'][..., 0]==1).float()[..., None],
                            (gt_buffers['segm'][..., 0]==0).float()[..., None]], axis=-1)
        
        if self.enable_canonical:
            mesh_dict_hum = self.getMesh(opt_material, mesh_type='human')
            mesh_hum = mesh_dict_hum['mesh_hum_posed']
            mesh_hum_cano = mesh_dict_hum['mesh_hum_cano']
            mesh_dict_obj = self.getMesh(opt_material, mesh_type='obj')
            mesh_obj = mesh_dict_obj['mesh_obj_posed']
            mesh_obj_cano = mesh_dict_obj['mesh_obj_cano']
        else:
            mesh_hum = self.getMesh(opt_material, mesh_type='human')['mesh_hum_posed']
            mesh_obj = self.getMesh(opt_material, mesh_type='obj')['mesh_obj_posed']
        

        buffers_hum= self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal=if_normal, 
                                 mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, 
                                 render_control_op=self.use_op_control)
        
        buffers_obj= self.render('obj', glctx, target, lgt, opt_material, opt_mesh=mesh_obj, if_normal= if_normal,
                                    mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, 
                                    render_control_op=self.use_op_control)

        img_hum = buffers_hum['v_color'][..., :3]
        mask_hum = buffers_hum['v_color'][..., -1]
        seg_hum = -self.dilate_seg(-gt_seg[..., 1]) if self.dilate_seg_mask else gt_seg[..., 1]

        img_obj = buffers_obj['v_color'][..., :3]
        mask_obj = buffers_obj['v_color'][..., -1]
        seg_obj = self.dilate_seg(gt_seg[..., 0]) if self.dilate_seg_mask else gt_seg[..., 0]

        rgb_loss = loss_fn(img_hum * mask_hum.unsqueeze(-1) * seg_hum.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * seg_hum.unsqueeze(-1)) + \
                    loss_fn(img_obj * mask_obj.unsqueeze(-1) * seg_obj.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * seg_obj.unsqueeze(-1))

        reg_loss += rgb_loss * self.w_recon_rgb


        if self.zoom_in_body:
            start, end = torch.zeros(3, device='cuda'), torch.zeros(3, device='cuda')
            start[..., 1] -= 0.8
            end[..., 1] += 0.6
            zoom_in_params_body = {
                'scale': torch.tensor(0.25).cuda(),             
                'transl': start + torch.rand_like(start) * (end - start)
            }
            gt_buffers = self.renderGT(glctx, target, lgt, opt_material, if_normal=if_normal, mode=mode, 
                                   if_flip_the_normal=if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_body)
            gt_img = gt_buffers['shaded'][..., :3]
            gt_mask = gt_buffers['shaded'][..., -1]
            gt_seg = torch.cat([(gt_buffers['segm'][..., 0]==1).float()[..., None], 
                                (gt_buffers['segm'][..., 0]==0).float()[..., None]], axis=-1)
        
            buffers_hum= self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal=if_normal, 
                                 mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, 
                                 render_control_op=self.use_op_control, zoom_in_params=zoom_in_params_body)
        
            buffers_obj= self.render('obj', glctx, target, lgt, opt_material, opt_mesh=mesh_obj, if_normal= if_normal,
                                        mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, 
                                        render_control_op=self.use_op_control, zoom_in_params=zoom_in_params_body)

            img_hum = buffers_hum['v_color'][..., :3]
            mask_hum = buffers_hum['v_color'][..., -1]
            seg_hum = -self.dilate_seg(-gt_seg[..., 1]) if self.dilate_seg_mask else gt_seg[..., 1]

            img_obj = buffers_obj['v_color'][..., :3]
            mask_obj = buffers_obj['v_color'][..., -1]
            seg_obj = self.dilate_seg(gt_seg[..., 0]) if self.dilate_seg_mask else gt_seg[..., 0]

            rgb_loss = loss_fn(img_hum * mask_hum.unsqueeze(-1) * seg_hum.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * seg_hum.unsqueeze(-1)) + \
                        loss_fn(img_obj * mask_obj.unsqueeze(-1) * seg_obj.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * seg_obj.unsqueeze(-1))

            reg_loss += rgb_loss * self.w_recon_rgb

            # if (not self.detach_obj and iteration % 2 == 1 and self.take_turns):
            #     if self.is_obj_most_outer:
            #         reg_loss += (loss_fn(mask_obj, gt_mask_obj) * 10 + loss_fn(img_obj * mask_obj.unsqueeze(-1), gt_img_obj * gt_mask_obj.unsqueeze(-1))) * self.w_recon_full
            #     else:
            #         reg_loss += loss_fn(img_obj * mask_obj.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1), gt_img * gt_mask.unsqueeze(-1) * gt_seg_obj.unsqueeze(-1)) * self.w_recon_full 

         # head and hand recon loss with the zoomed in images if enabled
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
            gt_buffers_head = self.renderGT(glctx, target, lgt, opt_material, if_normal=if_normal, mode=mode, 
                                   if_flip_the_normal=if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_head)
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
            reg_loss += (mask_loss_head + norm_loss_head) * self.w_recon_rgb

            # left hand
            gt_buffers_lh = self.renderGT(glctx, target, lgt, opt_material, if_normal=if_normal, mode=mode, 
                                   if_flip_the_normal=if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_left_hand)
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
            reg_loss += (mask_loss_lh + norm_loss_lh) * self.w_recon_rgb

            # right hand
            gt_buffers_rh = self.renderGT(glctx, target, lgt, opt_material, if_normal=if_normal, mode=mode, 
                                   if_flip_the_normal=if_flip_the_normal, if_use_bump = if_use_bump, zoom_in_params=zoom_in_params_right_hand)
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
            reg_loss += (mask_loss_rh + norm_loss_rh) * self.w_recon_rgb

        # get obj bbox for zoom in
        if self.tick_count == self.FLAGS.init_iter:
            if self.enable_canonical:
                bbox_min, bbox_max = self.get_obj_bbox(self.getMesh(opt_material, mesh_type='obj')['mesh_obj_cano'])
            else:
                bbox_min, bbox_max = self.get_obj_bbox(self.getMesh(opt_material, mesh_type='obj')['mesh_obj_posed'])
            zoom_in_params = {}
            zoom_in_params['scale_start'] = max(max((bbox_max[0] - bbox_min[0]) / 2 * 1.2), torch.tensor(0.5)) # largest of xyz scales, gets smaller if multiplied by larger value at the end
            zoom_in_params['scale_end'] = max(max((bbox_max[0] - bbox_min[0]) / 2 * 0.6), torch.tensor(0.25))
            zoom_in_params['transl_start'] = ((bbox_max[0] - bbox_min[0]) / 4) + bbox_min[0]
            zoom_in_params['transl_end'] = bbox_max[0] - ((bbox_max[0] - bbox_min[0]) / 4)
            self.zoom_in_params = zoom_in_params


        if self.tick_count > self.FLAGS.init_iter:
            ################################################################################
            # SDS loss for human
            ################################################################################
            if (not self.detach_hum or iteration % 2 == 0 and self.take_turns) or not self.use_canonical_sds:

                buffers_sds_hum = self.choose_buffer('human', mesh_hum, mesh_hum_cano, glctx, target, lgt, opt_material, if_normal, mode, if_flip_the_normal,\
                                                     if_use_bump, zoom_in=None)
                if self.FLAGS.add_directional_text:
                    text_embeddings = torch.cat([guidance.uncond_z[target['prompt_index']], guidance.text_z[target['prompt_index']]])
                else:
                    text_embeddings = torch.cat([guidance.uncond_z, guidance.text_z])
                sds_loss += self.apply_SDS(buffers=buffers_sds_hum, text_embeddings=text_embeddings, guidance=guidance, iteration=iteration) * self.w_sds  

                if self.multi_pose:
                    buffers_sds_hum = self.render('human', glctx, target, lgt, opt_material, opt_mesh=mesh_hum, if_normal= if_normal, mode=mode, zoom_in_params=None,\
                                            if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, render_control_op=self.use_op_control)
                    sds_loss += self.apply_SDS(buffers=buffers_sds_hum, text_embeddings=text_embeddings, guidance=guidance, iteration=iteration) * self.w_sds  

                ### SDS loss human - zoom
                # Zoom in (if enabled) after the initialization refinement stage
                if self.enable_zoom_in and self.FLAGS.init_refine_iter < iteration:
                    curr_zoom_in_params = {
                        'scale': self.zoom_in_params['scale_start'] + torch.rand_like(self.zoom_in_params['scale_start']) * (self.zoom_in_params['scale_end'] - self.zoom_in_params['scale_start']), # scale between no-zoom and scale-start
                        'transl': self.zoom_in_params['transl_start'] + torch.rand_like(self.zoom_in_params['transl_start']) * (self.zoom_in_params['transl_end'] - self.zoom_in_params['transl_start'])
                    }
                    buffers_sds_hum_zoom = self.choose_buffer('human', mesh_hum, mesh_hum_cano, glctx, target, lgt, opt_material, if_normal, mode, if_flip_the_normal,\
                                                            if_use_bump, zoom_in=curr_zoom_in_params)
                    sds_loss += self.apply_SDS(buffers=buffers_sds_hum_zoom, text_embeddings=text_embeddings, guidance=guidance, iteration=iteration) * self.w_sds          


            ################################################################################
            # SDS loss for object in compsition space
            ################################################################################      
            if (not self.detach_obj and iteration % 2 == 1 and self.take_turns) and self.use_canonical_sds:   
                # get comp mesh
                if self.enable_canonical:
                    mesh_dict_comp = self.getMesh(opt_material, mesh_type='comp', detach_hum=self.detach_hum)
                    mesh_comp_sds = mesh_dict_comp['mesh_comp_posed']
                    mesh_comp_cano_sds = mesh_dict_comp['mesh_comp_cano']
                else:
                    mesh_dict_comp = self.getMesh(opt_material, mesh_type='comp', detach_hum=self.detach_hum)
                    mesh_comp_sds = mesh_dict_comp['mesh_comp_posed']
                    mesh_comp_cano_sds = None

                ### SDS loss object - no zoom
                # Choose buffers that are to be used for SDS loss in composition space for object 
                buffers_sds_comp = self.choose_buffer('comp', mesh_comp_sds, mesh_comp_cano_sds, glctx, target, lgt, opt_material, if_normal, mode, \
                                                       if_flip_the_normal, if_use_bump, zoom_in=None)
                if self.FLAGS.add_directional_text:
                    text_embeddings_comp = torch.cat([guidance.uncond_z_comp[target['prompt_index']], guidance.text_z_comp[target['prompt_index']]]) # [B*2, 77, 1024]
                else:
                    text_embeddings_comp = torch.cat([guidance.uncond_z_comp, guidance.text_z_comp])  # [B * 2, 77, 1024]
                
                sds_loss += self.apply_SDS(buffers=buffers_sds_comp, text_embeddings=text_embeddings_comp, guidance=guidance, iteration=iteration) * self.w_sds   
        
                 ### SDS loss object - zoom
                # Zoom in (if enabled) after the initialization refinement stage
                if self.enable_zoom_in and self.FLAGS.init_refine_iter < iteration:
                    curr_zoom_in_params = {
                        'scale': self.zoom_in_params['scale_start'] + torch.rand_like(self.zoom_in_params['scale_start']) * (self.zoom_in_params['scale_end'] - self.zoom_in_params['scale_start']), # scale between no-zoom and scale-start
                        'transl': self.zoom_in_params['transl_start'] + torch.rand_like(self.zoom_in_params['transl_start']) * (self.zoom_in_params['transl_end'] - self.zoom_in_params['transl_start'])
                    }
                    buffers_sds_comp_zoom = self.choose_buffer('comp', mesh_comp_sds, mesh_comp_cano_sds, glctx, target, lgt, opt_material, if_normal, mode, \
                                                       if_flip_the_normal, if_use_bump, zoom_in=curr_zoom_in_params)
                    sds_loss += self.apply_SDS(buffers=buffers_sds_comp_zoom, text_embeddings=text_embeddings_comp, guidance=guidance, iteration=iteration) * self.w_sds   
     
        return sds_loss, img_loss, reg_loss
    
    def tick_refine(self, glctx, target, lgt, opt_material, iteration, if_normal, guidance, scene_and_vertices, mode, if_flip_the_normal, if_use_bump):
        self.tick_count += 1
        sds_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        img_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        loss_fn = torch.nn.MSELoss()

        bbox_min, bbox_max = self.get_obj_bbox(self.mesh_obj) # get object mesh bbox in canonical space
        zoom_in_params = {}
        zoom_in_params['scale_start'] = max(max((bbox_max[0] - bbox_min[0]) / 2 * 1.2), torch.tensor(0.5)) # largest of xyz scales, gets smaller if multiplied by larger value at the end
        zoom_in_params['scale_end'] = max(max((bbox_max[0] - bbox_min[0]) / 2 * 0.6), torch.tensor(0.25))
        zoom_in_params['transl_start'] = ((bbox_max[0] - bbox_min[0]) / 4) + bbox_min[0]
        zoom_in_params['transl_end'] = bbox_max[0] - ((bbox_max[0] - bbox_min[0]) / 4)

        if self.visible_verts is None:
            target_segms = {'mv': [], 'mvp': [], 'campos': [], 'background': [], 'normal_rotate': []}
            for i in range(18):
                target_segms['mv'].append(DatasetMesh.train_scene_segm(i)['mv'])
                target_segms['mvp'].append(DatasetMesh.train_scene_segm(i)['mvp'])
                target_segms['campos'].append(DatasetMesh.train_scene_segm(i)['campos'])
                target_segms['background'].append(DatasetMesh.train_scene_segm(i)['background'])
                target_segms['normal_rotate'].append(DatasetMesh.train_scene_segm(i)['normal_rotate'])
                target_segms['mv'].append(DatasetMesh.train_scene_segm_2(i)['mv'])
                target_segms['mvp'].append(DatasetMesh.train_scene_segm_2(i)['mvp'])
                target_segms['campos'].append(DatasetMesh.train_scene_segm_2(i)['campos'])
                target_segms['background'].append(DatasetMesh.train_scene_segm_2(i)['background'])
                target_segms['normal_rotate'].append(DatasetMesh.train_scene_segm_2(i)['normal_rotate'])

            self.target_segms_dict = {'mv': torch.cat(target_segms['mv'], dim=0), 'mvp': torch.cat(target_segms['mvp'], dim=0), 'campos': torch.cat(target_segms['campos'], dim=0), \
                                'background': torch.cat(target_segms['background'], dim=0), 'normal_rotate': torch.cat(target_segms['normal_rotate'], dim=0),
                                'resolution' : torch.tensor([512, 512]), 'spp' : 1, 'prompt_index' : 0}
            self.mesh_obj = mesh.auto_normals(self.mesh_obj)

            rast = render.render_mesh(glctx, self.mesh_obj.clone(), self.target_segms_dict['mvp'], self.target_segms_dict['campos'], lgt, self.target_segms_dict['resolution'], spp=self.target_segms_dict['spp'], msaa= True,
                                        background= self.target_segms_dict['background'],bsdf= None, if_normal= if_normal,normal_rotate= self.target_segms_dict['normal_rotate'],mode = mode,
                                        if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, return_rast=True)
            
            face_idx = torch.unique(rast).long() - 1
            self.visible_verts = torch.unique(self.mesh_obj.t_pos_idx[face_idx])
        # self.visible_verts = torch.arange(0, len(self.mesh_obj.v_pos))

        mesh_hum = self.mesh.clone()
        mesh_hum = mesh.auto_normals(mesh_hum)      
        mesh_hum.v_pos += self.initial_hum_norm * self.disp_norm_hum


        dists, idx, nn = knn_points(self.mesh_obj.v_pos[self.visible_verts][None], mesh_hum.v_pos[None], return_nn=True)

        vec_obj2hum = nn[0].squeeze(1) - self.mesh_obj.v_pos[self.visible_verts]
        vec_obj2hum = torch.div(vec_obj2hum.T, vec_obj2hum.norm(dim=1)).T
        obj_nn_nrm = self.mesh_obj.v_nrm[self.visible_verts]

        dot = (vec_obj2hum * obj_nn_nrm).sum(axis=1)
        
        reg_loss += dot[dot > 0].sum()


        dists, idx, nn = knn_points(mesh_hum.v_pos[None], self.mesh_obj.v_pos[self.visible_verts][None], return_nn=True)

        dists = dists[0].squeeze(1)
        vec_hum2obj = nn[0].squeeze(1) - mesh_hum.v_pos
        vec_hum2obj = torch.div(vec_hum2obj.T, vec_hum2obj.norm(dim=1)).T
        hum_nn_nrm = self.mesh.v_nrm

        dot_ = (vec_hum2obj[dists < 0.01] * hum_nn_nrm[dists < 0.01]).sum(axis=1)
        
        reg_loss += (-dot_[dot_ < 0]).sum()

        loss_offset = (torch.linalg.norm(self.disp_norm_hum, dim=1).unsqueeze(1) * self.w_reg_offset).sum()
        reg_loss += loss_offset

        loss_offset = (torch.linalg.norm(self.disp_norm_hum, dim=1).unsqueeze(1) * self.w_reg_offset).sum()
        reg_loss += loss_offset

        loss_edge = avg_edge_length(mesh_hum.v_pos, mesh_hum.t_pos_idx) * self.w_reg_edge
        loss_lap = laplace_regularizer_const(mesh_hum.v_pos, mesh_hum.t_pos_idx) * self.w_reg_lap
        loss_norm = normal_consistency(mesh_hum.v_pos, mesh_hum.t_pos_idx) * self.w_reg_norm_smooth
        reg_loss += (loss_edge + loss_lap + loss_norm)


        self.mesh_comp = mesh.combine_texture(mesh_hum, self.mesh_obj, self.mesh.v_color, self.mesh_obj.v_color, opt_material)

        print(f'dot: {dot[dot > 0].sum()}, dot_: {(-dot_[dot_ < 0]).sum()}, offset: {loss_offset}, edge: {loss_edge}, lap: {loss_lap}, norm: {loss_norm}')


        
        # if self.dist_hum2obj is None:
        #     self.dist_hum2obj = knn_points(self.mesh.v_pos[None], self.mesh_obj.v_pos[None]).dists[0]

        # if self.FLAGS.refine_hum:
        #     # v_hum = (mesh_hum_cano.v_pos.view(-1, 3) - self.getAABB()[0][None, ...]) / (self.getAABB()[1][None, ...] - self.getAABB()[0][None, ...])
        #     # v_hum = torch.clamp(v_hum, min=0, max=1)
        #     # v_hum_enc = self.refine_encoder_hum(mesh_hum_cano.v_pos)
        #     # offset_hum = self.refine_deformer_hum(v_hum_enc)
        #     # mesh_hum_cano.v_pos = mesh_hum_cano.v_pos + self.initial_hum_norm * self.disp_norm_hum
            
        #     # dist_hum2obj = knn_points(mesh_hum_cano.v_pos[None], mesh_obj_cano.v_pos[None]).dists[0]
        #     # reg_loss += (torch.linalg.norm(offset_hum, dim=1).unsqueeze(1) * torch.exp(0 * dist_hum2obj) * self.w_reg_offset).sum()

        #     reg_loss += avg_edge_length(self.mesh.v_pos, self.mesh.t_pos_idx) * self.w_reg_edge
        #     reg_loss += laplace_regularizer_const(self.mesh.v_pos, self.mesh.t_pos_idx) * self.w_reg_lap
        #     reg_loss += normal_consistency(self.mesh.v_pos, self.mesh.t_pos_idx) * self.w_reg_norm_smooth

        # if self.FLAGS.refine_obj:
        #     v_obj = (mesh_obj_cano.v_pos.view(-1, 3) - self.getAABBObj()[0][None, ...]) / (self.getAABBObj()[1][None, ...] - self.getAABBObj()[0][None, ...])
        #     v_obj = torch.clamp(v_obj, min=0, max=1)
        #     v_obj_enc = self.refine_encoder_obj(v_obj.contiguous())
        #     offset_obj = self.refine_deformer_obj(v_obj_enc)
        #     mesh_obj_cano.v_pos = mesh_obj_cano.v_pos + offset_obj
        #     reg_loss += loss_fn(offset_obj, torch.zeros_like(offset_obj)) * self.w_reg_offset


        # mesh_comp_cano = mesh.combine_texture(mesh_hum, self.mesh_obj, self.mesh.v_color, self.mesh_obj.v_color, self.mesh_gt.material)
        # self.mesh_comp = mesh_comp_cano

        #  ### SDS loss object - no zoom
        # # Choose buffers that are to be used for SDS loss in composition space for object 
        # buffers_sds_comp = self.choose_buffer('comp', None, mesh_comp_cano, glctx, target, lgt, opt_material, if_normal, mode, \
        #                                         if_flip_the_normal, if_use_bump, zoom_in=None)
        # buffers_obj= self.render('obj', glctx, target, lgt, opt_material, opt_mesh=self.mesh_obj, if_normal= if_normal,
        #                             mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, 
        #                             render_control_op=self.use_op_control)
        
        # mask_obj = buffers_obj['v_color'][..., -1]
        # segm_comp_obj_mask = buffers_sds_comp['segm'][..., 0]
        
        # reg_loss += loss_fn(mask_obj, segm_comp_obj_mask) * self.w_mask
          

        #     ### SDS loss object - zoom
        # # Zoom in (if enabled) after the initialization refinement stage
        # if self.enable_zoom_in:
        #     curr_zoom_in_params = {
        #         'scale': zoom_in_params['scale_start'] + torch.rand_like(zoom_in_params['scale_start']) * (zoom_in_params['scale_end'] - zoom_in_params['scale_start']), # scale between no-zoom and scale-start
        #         'transl': zoom_in_params['transl_start'] + torch.rand_like(zoom_in_params['transl_start']) * (zoom_in_params['transl_end'] - zoom_in_params['transl_start'])
        #     }
        #     buffers_sds_comp_zoom = self.choose_buffer('comp', None, mesh_comp_cano, glctx, target, lgt, opt_material, if_normal, mode, \
        #                                         if_flip_the_normal, if_use_bump, zoom_in=curr_zoom_in_params)
        #     buffers_obj= self.render('obj', glctx, target, lgt, opt_material, opt_mesh=self.mesh_obj, if_normal= if_normal,
        #                             mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump, 
        #                             render_control_op=self.use_op_control, zoom_in_params=curr_zoom_in_params)
        
        #     mask_obj = buffers_obj['v_color'][..., -1]
        #     segm_comp_obj_mask = buffers_sds_comp_zoom['segm'][..., 0]
            
        #     reg_loss += loss_fn(mask_obj, segm_comp_obj_mask) * self.w_mask
            
        
        
        return sds_loss, img_loss, reg_loss

    ################################################################################
    # Apply face segm infos to mesh_hum and mesh_obj to get seg masks in canonical space for inpainting
    ################################################################################
    # if self.use_canonical_sds:
    #     # to get vis mask for inpainting masks
    #     mesh_comp, mesh_comp_cano = self.getMesh(opt_material, mesh_type='comp', detach_hum=self.detach_hum)

    #     target_segms = {'mv': [], 'mvp': [], 'campos': [], 'background': [], 'normal_rotate': []}
    #     for i in range(18):
    #         target_segms['mv'].append(DatasetMesh.train_scene_segm(i)['mv'])
    #         target_segms['mvp'].append(DatasetMesh.train_scene_segm(i)['mvp'])
    #         target_segms['campos'].append(DatasetMesh.train_scene_segm(i)['campos'])
    #         target_segms['background'].append(DatasetMesh.train_scene_segm(i)['background'])
    #         target_segms['normal_rotate'].append(DatasetMesh.train_scene_segm(i)['normal_rotate'])
    #     target_segms_dict = {'mv': torch.cat(target_segms['mv'], dim=0), 'mvp': torch.cat(target_segms['mvp'], dim=0), 'campos': torch.cat(target_segms['campos'], dim=0), \
    #                         'background': torch.cat(target_segms['background'], dim=0), 'normal_rotate': torch.cat(target_segms['normal_rotate'], dim=0),
    #                         'resolution' : torch.tensor([512, 512]), 'spp' : 1, 'prompt_index' : 0}
        
    #     gt_buffers_for_cano_mask = self.renderGT(glctx, target_segms_dict, lgt, opt_material, if_normal=if_normal, mode=mode, 
    #                                             if_flip_the_normal=if_flip_the_normal, if_use_bump = if_use_bump)
    #     gt_seg_for_cano_mask = torch.cat([(gt_buffers_for_cano_mask['segm'][..., 0]==1).float()[..., None], 
    #                     (gt_buffers_for_cano_mask['segm'][..., 0]==0).float()[..., None]], axis=-1)
    #     gt_masks_posed = gt_buffers_for_cano_mask['shaded'][..., -1]
    #     gt_segs_posed = gt_seg_for_cano_mask

    #     rast_hum = self.render('human', glctx, target_segms_dict, lgt, opt_material, opt_mesh=mesh_hum, if_normal= if_normal, mode = mode, 
    #                     if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, return_rast=True)
                        
    #     mesh_comp_vis = mesh_comp.clone()
    #     mesh_comp_vis.f_seg = None
    #     mesh_comp_vis_cano = mesh_comp_cano.clone()
    #     mesh_comp_vis_cano.f_seg = None

    #     rast_comp_vis = self.render('comp', glctx, target_segms_dict, lgt, opt_material, opt_mesh=mesh_comp_vis, if_normal= if_normal, mode = mode, 
    #                     if_flip_the_normal = if_flip_the_normal, if_use_bump = if_use_bump, return_rast=True)
        
    #     is_human = -self.dilate_seg(-gt_segs_posed[..., 1]) * gt_masks_posed 
    #     is_mask = gt_masks_posed
        
    #     # for the human mesh, face_seg_label == 1 means the region is not considered as human
    #     hum_seg = torch.ones((mesh_hum.t_pos_idx.shape[0], 1), device='cuda')
    #     hum_seg[torch.unique(rast_hum * is_human).long() - 1] = 0
    #     self.mesh_posed.f_seg = hum_seg
    #     self.mesh.f_seg = hum_seg

    #     comp_vis = torch.zeros((mesh_comp_vis.t_pos_idx.shape[0], 1), device='cuda')
    #     comp_vis[torch.unique(rast_comp_vis * is_mask).long() - 1] = 1
    #     mesh_comp_vis.f_seg = comp_vis
    #     mesh_comp_vis_cano.f_seg = comp_vis

    #     # for the composition mesh, segm_comp_vis == 1 means it's visible from posed scan
    #     buffers_comp_vis = self.render('comp', glctx, target, lgt, opt_material, opt_mesh=mesh_comp_vis, if_normal=if_normal, 
    #                             mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump)
    #     segm_comp_vis = buffers_comp_vis['segm']
    #     self.buffers_comp_vis_cano = self.render('comp', glctx, target, lgt, opt_material, opt_mesh=mesh_comp_vis_cano, if_normal=if_normal, 
    #                             mode=mode, if_flip_the_normal=if_flip_the_normal, if_use_bump=if_use_bump)
    #     self.segm_comp_vis_cano = self.buffers_comp_vis_cano['segm']