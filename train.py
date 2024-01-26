# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os, sys
import time
import argparse
import json
import math
import numpy as np
import torch
import nvdiffrast.torch as dr
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime


# Import data readers / generators
from dataset.dataset_mesh import DatasetMesh
from dataset.dataset_mesh import get_camera_params

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render
from utils.sd import StableDiffusion
from tqdm import tqdm
import open3d as o3d
import torchvision.transforms as transforms
from render import util
from render.video import Video
import random
import imageio
import os.path as osp

import trimesh

###############################################################################
# Mix background into a dataset image
###############################################################################
@torch.no_grad()
def prepare_batch(target, background= 'black',it = 0,coarse_iter=0):
    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['normal_rotate'] = target['normal_rotate'].cuda()
    # target['prompt_index'] = target['prompt_index'].cuda()
    batch_size = target['mv'].shape[0]
    resolution = target['resolution']
    if background == 'random':
        target['background'] = torch.ones(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device='cuda')
        if random.random() < 0.5:
            target['background'] = target['background'] * torch.rand(target['background'].shape[0], 1, 1, 3).expand_as(target['background']).cuda()
    if background == 'white':
        target['background']= torch.ones(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device='cuda') 
    if background == 'black':
        target['background'] = torch.zeros(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device='cuda')
    
        # if it<=coarse_iter:
        #     target['background'][:,:,:,0:2] -=1
        #     target['background'][:,:,:,2:3] +=1
    return target


###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS, canonical=False):
    if FLAGS.enable_canonical and canonical:
        eval_mesh = geometry.getMesh(mat, mesh_type='human', canonical=canonical, val=FLAGS.subdivision)['mesh_hum_cano']
    elif FLAGS.enable_canonical and not canonical:
        eval_mesh = geometry.getMesh(mat, mesh_type='human', canonical=canonical, val=FLAGS.subdivision)['mesh_hum_posed']
    else:
        eval_mesh = geometry.getMesh(mat, mesh_type='human', canonical=canonical, val=FLAGS.subdivision)['mesh_hum_posed']
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()

    # clean mesh
    # mesh_tri = trimesh.Trimesh(vertices=v_pos, faces=t_pos_idx)
    # cc = mesh_tri.split(only_watertight=False)
    # out_mesh = cc[0]
    # bbox = out_mesh.bounds
    # height = bbox[1,0] - bbox[0,0]
    # for c in cc:
    #     bbox = c.bounds
    #     if height < bbox[1,0] - bbox[0,0]:
    #         height = bbox[1,0] - bbox[0,0]
    #         out_mesh = c
    # v_pos, t_pos_idx = out_mesh.vertices, out_mesh.faces
    
    # vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    # indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    # uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    # faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    # new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    new_mesh = mesh.Mesh(v_pos=torch.tensor(v_pos, device='cuda'), t_pos_idx=torch.tensor(t_pos_idx, device='cuda'))
    new_mesh = mesh.auto_normals(new_mesh)
    
    # mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
    # if FLAGS.layers > 1:
    #     kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    # kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    # ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    # nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    # new_mesh.material = material.Material({
    #     'bsdf'   : mat['bsdf'],
    #     'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
    #     'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
    #     'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    # })

    return new_mesh

@torch.no_grad()
def xatlas_uvmap1(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat, mesh_type='human')
    new_mesh = mesh.Mesh( base=eval_mesh)
    new_mesh = mesh.auto_normals(new_mesh)
    mask, kd, ks, normal = render.render_uv1(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'], FLAGS.uv_padding_block)
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

@torch.no_grad()
def xatlas_uvmap_obj(glctx, geometry, mat, FLAGS, canonical=False):
    if FLAGS.enable_canonical and canonical:
        eval_mesh = geometry.getMesh(mat, mesh_type='obj', canonical=canonical, val=FLAGS.subdivision)['mesh_obj_cano']
    elif FLAGS.enable_canonical and not canonical:
        eval_mesh = geometry.getMesh(mat, mesh_type='obj', canonical=canonical, val=FLAGS.subdivision)['mesh_obj_posed']
    else:
        eval_mesh = geometry.getMesh(mat, mesh_type='obj', canonical=canonical, val=FLAGS.subdivision)['mesh_obj_posed']

    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()

    # clean mesh
    # mesh_tri = trimesh.Trimesh(vertices=v_pos, faces=t_pos_idx)
    # cc = mesh_tri.split(only_watertight=False)
    # out_mesh = cc[0]
    # bbox = out_mesh.bounds
    # height = bbox[1,0] - bbox[0,0]
    # for c in cc:
    #     bbox = c.bounds
    #     if height < bbox[1,0] - bbox[0,0]:
    #         height = bbox[1,0] - bbox[0,0]
    #         out_mesh = c
    # v_pos, t_pos_idx = out_mesh.vertices, out_mesh.faces
    
    # vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    # indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    # uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    # faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    # new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    new_mesh = mesh.Mesh(v_pos=torch.tensor(v_pos, device='cuda'), t_pos_idx=torch.tensor(t_pos_idx, device='cuda'))
    new_mesh = mesh.auto_normals(new_mesh)
    
    # mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
    # if FLAGS.layers > 1:
    #     kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    # kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    # ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    # nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    # new_mesh.material = material.Material({
    #     'bsdf'   : mat['bsdf'],
    #     'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
    #     'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
    #     'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    # })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def get_normalize_mesh(pro_path):
    mesh = o3d.io.read_triangle_mesh(str(pro_path))
    vertices = np.asarray(mesh.vertices)
    # shift = np.mean(vertices,axis=0)
    # scale = np.max(np.linalg.norm(vertices-shift, ord=2, axis=1)) * 0.8
    # vertices = (vertices-shift) / scale
    # vertices *= 2
    # vertices[:, 1] += 0.15
    mesh.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
    return mesh


def initial_guness_material(geometry, mlp, FLAGS, init_mat=None):
    # ipdb.set_trace(())
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat

###############################################################################
# Validation & testing
###############################################################################
# @torch.no_grad()  
def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, relight = None):
    result_dict = {}
    with torch.no_grad():
        # if FLAGS.mode == 'appearance_modeling':
        #     with torch.no_grad():
        #         lgt.build_mips()
        #         if FLAGS.camera_space_light:
        #             lgt.xfm(target['mv'])
        #         if relight != None:
        #             relight.build_mips()

        # hum
        buffers = geometry.render('human', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)
        result_dict['shaded'] =  buffers['shaded'][0, ..., 0:3]
        result_dict['shaded'] = util.rgb_to_srgb(result_dict['shaded'])
        if relight != None:
            result_dict['relight'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
            result_dict['relight'] = util.rgb_to_srgb(result_dict['relight'])
        result_dict['mask'] = (buffers['shaded'][0, ..., 3:4])
        result_image = result_dict['shaded']

        if FLAGS.display is not None :
            # white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)

                elif 'bsdf' in layer:
                    buffers  = geometry.render('human', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    mask = buffers['shaded'][0, ..., -1:]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']] * mask + torch.ones_like(result_dict[layer['bsdf']]) * (1-mask)], axis=1)

        if FLAGS.enable_canonical:
            buffers = geometry.render('human', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, canonical=True, val=FLAGS.vis_subdiv)
            result_dict['shaded_cano'] =  buffers['shaded'][0, ..., 0:3]
            result_dict['shaded_cano'] = util.rgb_to_srgb(result_dict['shaded_cano'])
            if relight != None:
                result_dict['relight_cano'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
                result_dict['relight_cano'] = util.rgb_to_srgb(result_dict['relight_cano'])
            result_dict['mask_cano'] = (buffers['shaded'][0, ..., 3:4])
            result_image = torch.cat([result_image, result_dict['shaded_cano']], axis=1)

            if FLAGS.display is not None :
                # white_bg = torch.ones_like(target['background'])
                for layer in FLAGS.display:
                    if 'latlong' in layer and layer['latlong']:
                        if isinstance(lgt, light.EnvironmentLight):
                            result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                        result_image = torch.cat([result_image, result_dict['light_image']], axis=1)

                    elif 'bsdf' in layer:
                        buffers = geometry.render('human', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump, canonical=True, val=FLAGS.vis_subdiv)
                        if layer['bsdf'] == 'kd':
                            result_dict[layer['bsdf']+'_cano'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                        elif layer['bsdf'] == 'normal':
                            result_dict[layer['bsdf']+'_cano'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                        else:
                            result_dict[layer['bsdf']+'_cano'] = buffers['shaded'][0, ..., 0:3]
                        mask = buffers['shaded'][0, ..., -1:]
                        result_image = torch.cat([result_image, result_dict[layer['bsdf']+'_cano'] * mask + torch.ones_like(result_dict[layer['bsdf']+'_cano']) * (1-mask)], axis=1)

            if FLAGS.pose_dependent_shape:
                buffers = geometry.render_target_pose('human', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump)
                result_dict['shaded_pd'] =  buffers['shaded'][0, ..., 0:3]
                result_dict['shaded_pd'] = util.rgb_to_srgb(result_dict['shaded_pd'])
                if relight != None:
                    result_dict['relight_cano'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
                    result_dict['relight_cano'] = util.rgb_to_srgb(result_dict['relight_cano'])
                result_dict['mask_pd'] = (buffers['shaded'][0, ..., 3:4])
                result_image = torch.cat([result_image, result_dict['shaded_pd']], axis=1)

                if FLAGS.display is not None :
                    # white_bg = torch.ones_like(target['background'])
                    for layer in FLAGS.display:
                        if 'latlong' in layer and layer['latlong']:
                            if isinstance(lgt, light.EnvironmentLight):
                                result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                            result_image = torch.cat([result_image, result_dict['light_image']], axis=1)

                        elif 'bsdf' in layer:
                            buffers = geometry.render_target_pose('human', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump)
                            if layer['bsdf'] == 'kd':
                                result_dict[layer['bsdf']+'_pd'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                            elif layer['bsdf'] == 'normal':
                                result_dict[layer['bsdf']+'_pd'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                            else:
                                result_dict[layer['bsdf']+'_pd'] = buffers['shaded'][0, ..., 0:3]
                            result_image = torch.cat([result_image, result_dict[layer['bsdf']+'_pd']], axis=1)

        # obj
        buffers = geometry.render('obj', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)
        result_dict['shaded_obj'] =  buffers['shaded'][0, ..., 0:3]
        result_dict['shaded_obj'] = util.rgb_to_srgb(result_dict['shaded_obj'])
        if relight != None:
            result_dict['relight_obj'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
            result_dict['relight_obj'] = util.rgb_to_srgb(result_dict['relight_obj'])
        result_dict['mask_obj'] = (buffers['shaded'][0, ..., 3:4])
        result_image_obj = result_dict['shaded_obj']

        if FLAGS.display is not None :
            # white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image_obj'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image_obj = torch.cat([result_image_obj, result_dict['light_image_obj']], axis=1)

                elif 'bsdf' in layer:
                    buffers  = geometry.render('obj', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']+'_obj'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']+'_obj'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']+'_obj'] = buffers['shaded'][0, ..., 0:3]
                    mask = buffers['shaded'][0, ..., -1:]
                    result_image_obj = torch.cat([result_image_obj, result_dict[layer['bsdf']+'_obj'] * mask + torch.ones_like(result_dict[layer['bsdf']+'_obj']) * (1- mask)], axis=1)

        if FLAGS.enable_canonical:
            buffers = geometry.render('obj', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, canonical=True, val=FLAGS.vis_subdiv)
            result_dict['shaded_obj_cano'] =  buffers['shaded'][0, ..., 0:3]
            result_dict['shaded_obj_cano'] = util.rgb_to_srgb(result_dict['shaded_obj_cano'])
            if relight != None:
                result_dict['relight_obj'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
                result_dict['relight_obj'] = util.rgb_to_srgb(result_dict['relight_obj'])
            result_dict['mask_obj_cano'] = (buffers['shaded'][0, ..., 3:4])
            result_image_obj = torch.cat([result_image_obj, result_dict['shaded_obj_cano']], axis=1)

            if FLAGS.display is not None :
                # white_bg = torch.ones_like(target['background'])
                for layer in FLAGS.display:
                    if 'latlong' in layer and layer['latlong']:
                        if isinstance(lgt, light.EnvironmentLight):
                            result_dict['light_image_obj'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                        result_image_obj = torch.cat([result_image_obj, result_dict['light_image_obj']], axis=1)

                    elif 'bsdf' in layer:
                        buffers  = geometry.render('obj', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump, canonical=True, val=FLAGS.vis_subdiv)
                        if layer['bsdf'] == 'kd':
                            result_dict[layer['bsdf']+'_obj_cano'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                        elif layer['bsdf'] == 'normal':
                            result_dict[layer['bsdf']+'_obj_cano'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                        else:
                            result_dict[layer['bsdf']+'_obj_cano'] = buffers['shaded'][0, ..., 0:3]
                        mask = buffers['shaded'][0, ..., -1:]
                        result_image_obj = torch.cat([result_image_obj, result_dict[layer['bsdf']+'_obj_cano'] * mask + torch.ones_like(result_dict[layer['bsdf']+'_obj_cano']) * (1-mask)], axis=1)
            
            if FLAGS.pose_dependent_shape:
                buffers = geometry.render_target_pose('obj', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump)
                result_dict['shaded_obj_pd'] =  buffers['shaded'][0, ..., 0:3]
                result_dict['shaded_obj_pd'] = util.rgb_to_srgb(result_dict['shaded_obj_pd'])
                if relight != None:
                    result_dict['relight_cano'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
                    result_dict['relight_cano'] = util.rgb_to_srgb(result_dict['relight_cano'])
                result_dict['mask_obj_pd'] = (buffers['shaded'][0, ..., 3:4])
                result_image_obj = torch.cat([result_image_obj, result_dict['shaded_obj_pd']], axis=1)

                if FLAGS.display is not None :
                    # white_bg = torch.ones_like(target['background'])
                    for layer in FLAGS.display:
                        if 'latlong' in layer and layer['latlong']:
                            if isinstance(lgt, light.EnvironmentLight):
                                result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                            result_image_obj = torch.cat([result_image_obj, result_dict['light_image']], axis=1)

                        elif 'bsdf' in layer:
                            buffers = geometry.render_target_pose('obj', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump)
                            if layer['bsdf'] == 'kd':
                                result_dict[layer['bsdf']+'_obj_pd'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                            elif layer['bsdf'] == 'normal':
                                result_dict[layer['bsdf']+'_obj_pd'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                            else:
                                result_dict[layer['bsdf']+'_obj_pd'] = buffers['shaded'][0, ..., 0:3]
                            result_image_obj = torch.cat([result_image_obj, result_dict[layer['bsdf']+'_obj_pd']], axis=1)


         # comp
        buffers = geometry.render('comp', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)
        result_dict['shaded_comp'] =  buffers['shaded'][0, ..., 0:3]
        result_dict['shaded_comp'] = util.rgb_to_srgb(result_dict['shaded_comp'])
        if relight != None:
            result_dict['relight_comp'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
            result_dict['relight_comp'] = util.rgb_to_srgb(result_dict['relight_comp'])
        result_dict['mask_comp'] = (buffers['shaded'][0, ..., 3:4])
        result_image_comp = result_dict['shaded_comp']

        if FLAGS.display is not None :
            # white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image_comp'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image_comp = torch.cat([result_image_comp, result_dict['light_image_comp']], axis=1)

                elif 'bsdf' in layer:
                    buffers  = geometry.render('comp', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']+'_comp'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']+'_comp'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']+'_comp'] = buffers['shaded'][0, ..., 0:3]
                    mask = buffers['shaded'][0, ..., -1:]
                    result_image_comp = torch.cat([result_image_comp, result_dict[layer['bsdf']+'_comp'] * mask + torch.ones_like(result_dict[layer['bsdf']+'_comp']) * (1-mask)], axis=1)

        if FLAGS.enable_canonical:
            buffers = geometry.render('comp', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, canonical=True, val=FLAGS.vis_subdiv)
            result_dict['shaded_comp_cano'] =  buffers['shaded'][0, ..., 0:3]
            result_dict['shaded_comp_cano'] = util.rgb_to_srgb(result_dict['shaded_comp_cano'])
            if relight != None:
                result_dict['relight_comp'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
                result_dict['relight_comp'] = util.rgb_to_srgb(result_dict['relight_comp'])
            result_dict['mask_comp_cano'] = (buffers['shaded'][0, ..., 3:4])
            result_image_comp = torch.cat([result_image_comp, result_dict['shaded_comp_cano']], axis=1)

            if FLAGS.display is not None :
                # white_bg = torch.ones_like(target['background'])
                for layer in FLAGS.display:
                    if 'latlong' in layer and layer['latlong']:
                        if isinstance(lgt, light.EnvironmentLight):
                            result_dict['light_image_comp'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                        result_image_comp = torch.cat([result_image_comp, result_dict['light_image_comp']], axis=1)

                    elif 'bsdf' in layer:
                        buffers  = geometry.render('comp', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump, canonical=True, val=FLAGS.vis_subdiv)
                        if layer['bsdf'] == 'kd':
                            result_dict[layer['bsdf']+'_comp_cano'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                        elif layer['bsdf'] == 'normal':
                            result_dict[layer['bsdf']+'_comp_cano'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                        else:
                            result_dict[layer['bsdf']+'_comp_cano'] = buffers['shaded'][0, ..., 0:3]
                        mask = buffers['shaded'][0, ..., -1:]
                        result_image_comp = torch.cat([result_image_comp, result_dict[layer['bsdf']+'_comp_cano'] * mask + torch.ones_like(result_dict[layer['bsdf']+'_comp_cano'] ) * (1 - mask)], axis=1)

            if FLAGS.pose_dependent_shape:
                buffers = geometry.render_target_pose('comp', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump)
                result_dict['shaded_comp_pd'] =  buffers['shaded'][0, ..., 0:3]
                result_dict['shaded_comp_pd'] = util.rgb_to_srgb(result_dict['shaded_comp_pd'])
                if relight != None:
                    result_dict['relight_cano'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump, val=FLAGS.vis_subdiv)['shaded'][0, ..., 0:3]
                    result_dict['relight_cano'] = util.rgb_to_srgb(result_dict['relight_cano'])
                result_dict['mask_obj_pd'] = (buffers['shaded'][0, ..., 3:4])
                result_image_comp = torch.cat([result_image_comp, result_dict['shaded_comp_pd']], axis=1)

                if FLAGS.display is not None :
                    # white_bg = torch.ones_like(target['background'])
                    for layer in FLAGS.display:
                        if 'latlong' in layer and layer['latlong']:
                            if isinstance(lgt, light.EnvironmentLight):
                                result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                            result_image_obj = torch.cat([result_image_obj, result_dict['light_image']], axis=1)

                        elif 'bsdf' in layer:
                            buffers = geometry.render_target_pose('comp', glctx, target, lgt, opt_material, bsdf=layer['bsdf'],  if_use_bump = FLAGS.if_use_bump)
                            if layer['bsdf'] == 'kd':
                                result_dict[layer['bsdf']+'_comp_pd'] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                            elif layer['bsdf'] == 'normal':
                                result_dict[layer['bsdf']+'_comp_pd'] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                            else:
                                result_dict[layer['bsdf']+'_comp_pd'] = buffers['shaded'][0, ..., 0:3]
                            result_image_comp = torch.cat([result_image_comp, result_dict[layer['bsdf']+'_comp_pd']], axis=1)

        result_image = torch.cat([result_image, result_image_obj, result_image_comp], axis=0)

        return result_image, result_dict
    
def validate_itr_appearance(glctx, target, geometry, opt_material, lgt, FLAGS, relight = None):
    result_dict = {}
    with torch.no_grad():
        # if FLAGS.mode == 'appearance_modeling':
        #     with torch.no_grad():
        #         lgt.build_mips()
        #         if FLAGS.camera_space_light:
        #             lgt.xfm(target['mv'])
        #         if relight != None:
        #             relight.build_mips()

        # hum
        buffers = geometry.render('human', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, refine=FLAGS.mode=='refine')
        result_dict['shaded'] =  buffers['v_color'][0, ..., 0:3]
        # result_dict['shaded'] = util.rgb_to_srgb(result_dict['shaded'])
        if relight != None:
            result_dict['relight'] = geometry.render('human', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump)['shaded'][0, ..., 0:3]
            result_dict['relight'] = util.rgb_to_srgb(result_dict['relight'])
        result_dict['mask'] = (buffers['shaded'][0, ..., 3:4])
        result_image = result_dict['shaded'] * result_dict['mask'] + torch.ones_like(result_dict['shaded']) * (1 - result_dict['mask'])

        if FLAGS.enable_canonical:
            buffers = geometry.render('human', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, canonical=True, refine=FLAGS.mode=='refine')
            result_dict['shaded_cano'] =  buffers['v_color'][0, ..., 0:3]
            # result_dict['shaded'] = util.rgb_to_srgb(result_dict['shaded'])
            if relight != None:
                result_dict['relight'] = geometry.render('human', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump)['shaded'][0, ..., 0:3]
                result_dict['relight'] = util.rgb_to_srgb(result_dict['relight'])
            result_dict['mask_cano'] = (buffers['shaded'][0, ..., 3:4])
            result_image = torch.cat([result_image, result_dict['shaded_cano'] * result_dict['mask_cano'] + torch.ones_like(result_dict['shaded_cano']) * (1 - result_dict['mask_cano'])], axis=1)

        # obj
        buffers = geometry.render('obj', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, refine=FLAGS.mode=='refine')
        result_dict['shaded_obj'] =  buffers['v_color'][0, ..., 0:3]
        # result_dict['shaded_obj'] = util.rgb_to_srgb(result_dict['shaded_obj'])
        if relight != None:
            result_dict['relight_obj'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump)['shaded'][0, ..., 0:3]
            result_dict['relight_obj'] = util.rgb_to_srgb(result_dict['relight_obj'])
        result_dict['mask_obj'] = (buffers['shaded'][0, ..., 3:4])
        result_image_obj = result_dict['shaded_obj'] * result_dict['mask_obj'] + torch.ones_like(result_dict['shaded_obj']) * (1 - result_dict['mask_obj'])

        if FLAGS.enable_canonical:
            buffers = geometry.render('obj', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, canonical=True, refine=FLAGS.mode=='refine')
            result_dict['shaded_obj_cano'] =  buffers['v_color'][0, ..., 0:3]
            # result_dict['shaded_obj'] = util.rgb_to_srgb(result_dict['shaded_obj'])
            if relight != None:
                result_dict['relight_obj'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump)['shaded'][0, ..., 0:3]
                result_dict['relight_obj'] = util.rgb_to_srgb(result_dict['relight_obj'])
            result_dict['mask_obj_cano'] = (buffers['shaded'][0, ..., 3:4])
            result_image_obj = torch.cat([result_image_obj, result_dict['shaded_obj_cano'] * result_dict['mask_obj_cano'] + torch.ones_like(result_dict['shaded_obj_cano']) * (1 - result_dict['mask_obj_cano'])], axis=1)

        # comp
        buffers = geometry.render('comp', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, refine=FLAGS.mode=='refine')
        result_dict['shaded_comp'] =  buffers['v_color'][0, ..., 0:3]
        # result_dict['shaded_comp'] = util.rgb_to_srgb(result_dict['shaded_comp'])
        if relight != None:
            result_dict['relight_comp'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump)['shaded'][0, ..., 0:3]
            result_dict['relight_comp'] = util.rgb_to_srgb(result_dict['relight_comp'])
        result_dict['mask_comp'] = (buffers['shaded'][0, ..., 3:4])
        result_image_comp = result_dict['shaded_comp'] * result_dict['mask_comp'] + torch.ones_like(result_dict['shaded_comp']) * (1 - result_dict['mask_comp'])

        if FLAGS.enable_canonical:
            buffers = geometry.render('comp', glctx, target, lgt, opt_material, if_use_bump = FLAGS.if_use_bump, canonical=True, refine=FLAGS.mode=='refine')
            result_dict['shaded_comp_cano'] =  buffers['v_color'][0, ..., 0:3]
            # result_dict['shaded_comp'] = util.rgb_to_srgb(result_dict['shaded_comp'])
            if relight != None:
                result_dict['relight_comp'] = geometry.render('obj', glctx, target, relight, opt_material, if_use_bump = FLAGS.if_use_bump)['shaded'][0, ..., 0:3]
                result_dict['relight_comp'] = util.rgb_to_srgb(result_dict['relight_comp'])
            result_dict['mask_comp_cano'] = (buffers['shaded'][0, ..., 3:4])
            result_image_comp = torch.cat([result_image_comp, result_dict['shaded_comp_cano'] * result_dict['mask_comp_cano'] + torch.ones_like(result_dict['shaded_comp_cano']) * (1 - result_dict['mask_comp_cano'])], axis=1)

        result_image = torch.cat([result_image, result_image_obj, result_image_comp], axis=0)

        return result_image, result_dict

def save_gif(dir,fps):
    imgpath = dir
    frames = []
    for idx in sorted(os.listdir(imgpath)):
        # print(idx)
        img = osp.join(imgpath,idx)
        frames.append(imageio.imread(img))
    imageio.mimsave(os.path.join(dir, 'eval.gif'),frames,'GIF',duration=1/fps)
    
@torch.no_grad()     
def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS, relight= None):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    
    shaded_dir = os.path.join(out_dir, "shaded")
    relight_dir = os.path.join(out_dir, "relight")
    kd_dir = os.path.join(out_dir, "kd")
    ks_dir = os.path.join(out_dir, "ks")
    normal_dir = os.path.join(out_dir, "normal")
    mask_dir = os.path.join(out_dir, "mask")
    
    os.makedirs(shaded_dir, exist_ok=True)
    os.makedirs(relight_dir, exist_ok=True)
    os.makedirs(kd_dir, exist_ok=True)
    os.makedirs(ks_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print("Running validation")
    dataloader_validate = tqdm(dataloader_validate)
    for it, target in enumerate(dataloader_validate):

        # Mix validation background
        target = prepare_batch(target, 'white')

        if FLAGS.mode == 'geometry_modeling':
            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, relight)
        elif FLAGS.mode == 'appearance_modeling' or FLAGS.mode == 'refine':
            result_image, result_dict = validate_itr_appearance(glctx, target, geometry, opt_material, lgt, FLAGS, relight)
        os.makedirs(shaded_dir + '/hum', exist_ok=True)
        os.makedirs(relight_dir + '/hum', exist_ok=True)
        os.makedirs(kd_dir + '/hum', exist_ok=True)
        os.makedirs(ks_dir + '/hum', exist_ok=True)
        os.makedirs(normal_dir + '/hum', exist_ok=True)
        os.makedirs(shaded_dir + '/obj', exist_ok=True)
        os.makedirs(normal_dir + '/obj', exist_ok=True)
        for k in result_dict.keys():
            np_img = result_dict[k].detach().cpu().numpy()
            if k == 'shaded':
                util.save_image(shaded_dir + '/hum/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'relight':
                util.save_image(relight_dir + '/hum/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'kd':
                util.save_image(kd_dir + '/hum/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'ks':
                util.save_image(ks_dir + '/hum/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'normal':
                util.save_image(normal_dir + '/hum/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'shaded_obj':
                util.save_image(shaded_dir + '/obj/' + ('val_%06d_%s.png' % (it, k)), np_img)
            elif k == 'normal_obj':
                util.save_image(normal_dir + '/obj/' + ('val_%06d_%s.png' % (it, k)), np_img)
            # elif k == 'mask':
            #     util.save_image(mask_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)
    if 'shaded' in result_dict.keys():
        save_gif(shaded_dir+'/hum',30)
    if 'relight' in result_dict.keys():
        save_gif(relight_dir+'/hum',30)
    if 'kd' in result_dict.keys():
        save_gif(kd_dir+'/hum',30)
    if 'ks' in result_dict.keys():
        save_gif(ks_dir+'/hum',30)
    if 'normal' in result_dict.keys():
        save_gif(normal_dir+'/hum',30)
    if 'shaded_obj' in result_dict.keys():
        save_gif(shaded_dir+'/obj',30)
    if 'normal_obj' in result_dict.keys():
        save_gif(normal_dir+'/obj',30)
    # if 'mask' in result_dict.keys():
    #     save_gif(mask_dir,30)
    return 0

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, FLAGS, guidance):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.FLAGS = FLAGS
        self.guidance = guidance
        self.if_flip_the_normal = FLAGS.if_flip_the_normal
        self.if_use_bump = FLAGS.if_use_bump
        # if self.FLAGS.mode == 'appearance_modeling':
        #     if not self.optimize_light:
        #         with torch.no_grad():
        #             self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        # self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []
        self.geo_params = list(self.geometry.parameters())

        if self.FLAGS.mode == 'refine':
            self.geo_params = []
            if self.FLAGS.refine_hum:
                self.geo_params += [self.geometry.disp_norm_hum]
                # self.geo_params += list(self.geometry.refine_deformer_hum.parameters()) 
            if self.FLAGS.refine_obj:
                self.geo_params += [self.geometry.disp_norm_obj]
                # self.geo_params += list(self.geometry.refine_encoder_obj.parameters())
                # self.geo_params += list(self.geometry.refine_deformer_obj.parameters()) 
      

    def forward(self, target, it, if_normal, if_pretrain, scene_and_vertices ):
        if self.FLAGS.mode == 'appearance_modeling':
            if self.optimize_light:
                self.light.build_mips()
                if self.FLAGS.camera_space_light:
                    self.light.xfm(target['mv'])
        if self.FLAGS.mode == 'refine':
            return self.geometry.tick_refine(self.glctx, target, self.light, self.material, it , if_normal, self.guidance, 
                                      scene_and_vertices, self.FLAGS.mode, self.if_flip_the_normal, self.if_use_bump)
        else:
            if if_pretrain:        
                return self.geometry.decoder.pre_train_ellipsoid(it, scene_and_vertices)
            else:
                return self.geometry.tick(self.glctx, target, self.light, self.material, it , if_normal, self.guidance, 
                                        scene_and_vertices, self.FLAGS.mode, self.if_flip_the_normal, self.if_use_bump)

def get_o3d_mesh(vertices, faces, vertex_normals=None, vertex_colors=None):
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    if vertex_normals is not None and torch.is_tensor(vertex_normals):
        vertex_normals = vertex_normals.detach().cpu().numpy()
    if vertex_colors is not None and torch.is_tensor(vertex_colors):
        vertex_colors = vertex_colors.detach().cpu().numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if vertex_normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    if vertex_colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return mesh

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    log_interval=10,
    optimize_light=True,
    optimize_geometry=True,
    guidance = None,
    scene_and_vertices = None,
    base_scene_and_vertices = None,
    ):
    
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=False)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)
   
    model = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, FLAGS, guidance)
    # model = model.cuda()
    if optimize_geometry: 
        
        optimizer_mesh = torch.optim.AdamW(model.geo_params, lr=0.001, betas=(0.9, 0.99), eps=1e-15) # doesn't converge if set to 0.01
        # scheduler_mesh = torch.optim.lr_scheduler.MultiStepLR(optimizer_mesh,
        #                                                 [400],
        #                                                 0.1)
    optimizer = torch.optim.AdamW(model.geo_params, lr=0.01, betas=(0.9, 0.99), eps=1e-15)
    if FLAGS.multi_gpu: 
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[FLAGS.local_rank],
                                                        find_unused_parameters= (FLAGS.mode =='geometry_modeling')
                                                        )
        
    if FLAGS.resume:
        ckpt = torch.load(FLAGS.resume_ckpt)
        model.geometry.load_state_dict(ckpt, strict=False)
    
    # if FLAGS.mode == 'refine':
    #     ckpt_hum = torch.load(Path(FLAGS.folder_hum_geo) / f'final.ckpt')
    #     ckpt_obj = torch.load(Path(FLAGS.folder_obj_geo) / f'final.ckpt')

    #     model.geometry.load_state_dict(ckpt_hum, strict=False)
    #     model_dict = model.geometry.state_dict()
    #     obj_dict = {k: v for k, v in ckpt_obj.items() if 'obj' in k}
    #     model_dict.update(obj_dict)
    #     model.geometry.load_state_dict(model_dict, strict=False)
        
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []
   

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)
    scaler = torch.cuda.amp.GradScaler(enabled=True)  
    
    rot_ang = 0
    if FLAGS.local_rank == 0:
        if FLAGS.enable_canonical:
            video_cano = Video(str(FLAGS.out_dir), name='human_cano.mp4')
            video_obj_cano = Video(str(FLAGS.out_dir), name='object_cano.mp4')
            video = Video(str(FLAGS.out_dir), name='human.mp4')
            video_obj = Video(str(FLAGS.out_dir), name='object.mp4')
        else:
            video = Video(str(FLAGS.out_dir), name='human.mp4')
            video_obj = Video(str(FLAGS.out_dir), name='object.mp4')
    if FLAGS.local_rank == 0:
        dataloader_train = tqdm(dataloader_train)
    
    # setup kaolin timelapse
    # save_ckpt3d = (FLAGS.mode =='geometry_modeling') and FLAGS.ckpt3d_interval
    save_ckpt3d = FLAGS.save_ckpt3d
    if save_ckpt3d:
        # import kaolin
        # timelapse = kaolin.visualize.Timelapse(str(FLAGS.out_dir / 'timelapse'))

        ''' Open3D version '''
        from open3d.visualization.tensorboard_plugin import summary
        from open3d.visualization.tensorboard_plugin.util import to_dict_batch
        from torch.utils.tensorboard import SummaryWriter

        log_dir = FLAGS.out_dir / 'tensorboard'
        writer = SummaryWriter(str(log_dir))

    for it, target in enumerate(dataloader_train):

        if FLAGS.resume:
            it = it + FLAGS.init_iter

        # Mix randomized background into dataset image
        target = prepare_batch(target, FLAGS.train_background, it, FLAGS.coarse_iter)  

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            save_video = FLAGS.video_interval and (it % FLAGS.video_interval == 0)
            save_ckpt3d = (FLAGS.mode =='geometry_modeling') and FLAGS.save_ckpt3d and \
                          FLAGS.ckpt3d_interval and (it % FLAGS.ckpt3d_interval == 0)
            if save_image and (it > 1 or FLAGS.mode == 'refine'):
                with torch.no_grad():
                    if FLAGS.mode =='geometry_modeling':
                        result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.train_background), geometry, opt_material, lgt, FLAGS)  #prepare_batch(next(v_it), FLAGS.background)
                    elif FLAGS.mode =='appearance_modeling':
                        result_image, result_dict = validate_itr_appearance(glctx, prepare_batch(next(v_it), FLAGS.train_background), geometry, opt_material, lgt, FLAGS)  #prepare_batch(next(v_it), FLAGS.background)
                    elif FLAGS.mode =='refine':
                        result_image, result_dict = validate_itr_appearance(glctx, prepare_batch(next(v_it), FLAGS.train_background), geometry, opt_material, lgt, FLAGS) 
                    np_result_image = result_image.detach().cpu().numpy()
                    util.save_image(str(FLAGS.out_dir) + '/' + ('img_%s_%06d.png' % (FLAGS.mode, img_cnt)), np_result_image)
                    img_cnt = img_cnt+1

            if save_video and it > 1:
            # if save_video and it >= FLAGS.init_iter:
                with torch.no_grad():
                    # params = get_camera_params(
                    #             resolution=512,
                    #             fov=45,
                    #             elev_angle=-20,
                    #             azim_angle =rot_ang,       
                    #         )  
                    # rot_ang += 1
                    params = get_camera_params(
                                resolution=512,
                                fov=45,
                                elev_angle=0,
                                azim_angle=0,       
                            )  
                    if FLAGS.mode =='geometry_modeling': 
                        #  or FLAGS.mode == 'refine'
                        bsdf_video = 'diffuse'
                        buffers = geometry.render('human', glctx, params, lgt, opt_material, bsdf=bsdf_video, if_use_bump = FLAGS.if_use_bump)
                        video_image = (buffers['shaded'][0, ..., 0:3]+1)/2
                        buffers_obj = geometry.render('obj', glctx, params, lgt, opt_material, bsdf=bsdf_video, if_use_bump = FLAGS.if_use_bump)
                        video_image_obj = (buffers_obj['shaded'][0, ..., 0:3]+1)/2
                    else:
                        buffers  = geometry.render('human', glctx, params, lgt, opt_material, bsdf='kd', if_use_bump = FLAGS.if_use_bump, refine=FLAGS.mode=='refine')
                        video_image = buffers['v_color'][0, ..., 0:3]
                        mask = buffers['v_color'][0, ..., 3:4]
                        video_image = (1 - mask) + mask * video_image
                        buffers_obj  = geometry.render('obj', glctx, params, lgt, opt_material, bsdf='kd', if_use_bump = FLAGS.if_use_bump, refine=FLAGS.mode=='refine')
                        video_image_obj = buffers_obj['v_color'][0, ..., 0:3]
                        mask_obj = buffers_obj['v_color'][0, ..., 3:4]
                        video_image_obj = (1 - mask_obj) + mask_obj * video_image_obj
                    video_image = video.ready_image(video_image)
                    video_image_obj = video_obj.ready_image(video_image_obj)

                    if FLAGS.enable_canonical:
                        if FLAGS.mode =='geometry_modeling':
                            buffers_cano = geometry.render('human', glctx, params, lgt, opt_material, bsdf=bsdf_video, 
                                                    if_use_bump = FLAGS.if_use_bump, canonical=True)
                            video_image_cano = (buffers_cano['shaded'][0, ..., 0:3]+1)/2
                            buffers_obj_cano = geometry.render('obj', glctx, params, lgt, opt_material, bsdf=bsdf_video, 
                                                        if_use_bump = FLAGS.if_use_bump, canonical=True)
                            video_image_obj_cano = (buffers_obj_cano['shaded'][0, ..., 0:3]+1)/2
                        else:
                            buffers_cano  = geometry.render('human', glctx, params, lgt, opt_material, bsdf='kd', 
                                                       if_use_bump = FLAGS.if_use_bump, canonical=True)
                            video_image_cano = buffers_cano['v_color'][0, ..., 0:3]
                            mask_cano = buffers_cano['v_color'][0, ..., 3:4]
                            video_image_cano = (1 - mask_cano) + mask_cano * video_image_cano
                            buffers_obj_cano  = geometry.render('obj', glctx, params, lgt, opt_material, bsdf='kd', 
                                                           if_use_bump = FLAGS.if_use_bump, canonical=True)
                            video_image_obj_cano = buffers_obj_cano['v_color'][0, ..., 0:3]
                            mask_obj_cano = buffers_obj_cano['v_color'][0, ..., 3:4]
                            video_image_obj_cano = (1 - mask_obj_cano) + mask_obj_cano * video_image_obj_cano
                        video_image_cano = video_cano.ready_image(video_image_cano)
                        video_image_obj_cano = video_obj_cano.ready_image(video_image_obj_cano)

            if save_ckpt3d and it > 1:
                if FLAGS.mode == 'geometry_modeling':
                    mesh_dict = geometry.getMesh(opt_material, mesh_type='comp', canonical=False)
                else:
                    mesh_dict = {}
                    for mesh_type in ['human', 'obj', 'comp']:
                        mesh_dict_per_type = geometry.getMesh(opt_material, mesh_type=mesh_type, canonical=False)
                        mesh_dict.update(mesh_dict_per_type)

                for key in mesh_dict.keys():
                    if key.startswith('mesh_'):
                        if it == FLAGS.ckpt3d_interval:  # initial writing 
                            mesh_for_vis = mesh_dict[key]
                            if FLAGS.mode =='geometry_modeling':
                                mesh_for_vis_o3d = get_o3d_mesh(mesh_for_vis.v_pos, mesh_for_vis.t_pos_idx, mesh_for_vis.v_nrm)
                            else:
                                mesh_for_vis_o3d = get_o3d_mesh(mesh_for_vis.v_pos, mesh_for_vis.t_pos_idx, 
                                                                mesh_for_vis.v_nrm, mesh_for_vis.v_color)
                            mesh_summary = to_dict_batch([mesh_for_vis_o3d])
                        else:
                            mesh_for_vis = mesh_dict[key]
                            if FLAGS.mode =='geometry_modeling':
                                mesh_for_vis_o3d = get_o3d_mesh(mesh_for_vis.v_pos, mesh_for_vis.t_pos_idx, mesh_for_vis.v_nrm)
                                mesh_summary = to_dict_batch([mesh_for_vis_o3d])
                            else:
                                mesh_for_vis_o3d = get_o3d_mesh(mesh_for_vis.v_pos, mesh_for_vis.t_pos_idx, 
                                                                mesh_for_vis.v_nrm, mesh_for_vis.v_color)
                                mesh_summary = to_dict_batch([mesh_for_vis_o3d])
                                # # for appearance geometry doesn't change
                                # mesh_summary['vertex_positions'] = 0
                                # mesh_summary['vertex_normals'] = 0
                                # mesh_summary['triangle_indices'] = 0
                                
                        writer.add_3d(key, mesh_summary, step=it)

                # timelapse.add_mesh_batch(category='human',
                #                         iteration=it,
                #                         faces_list=[mesh_hum_cano.t_pos_idx],
                #                         vertices_list=[mesh_hum_cano.v_pos])
                # mesh_obj_cano = geometry.getMesh(opt_material, mesh_type='obj', canonical=True)[0]
                # timelapse.add_mesh_batch(category='obj',
                #                         iteration=it,
                #                         faces_list=[mesh_obj_cano.t_pos_idx],
                #                         vertices_list=[mesh_obj_cano.v_pos])
                    
        iter_start_time = time.time()
        if FLAGS.mode =='geometry_modeling':
            if it<0:
                if_pretrain = True
            else:
                if_pretrain = False
            if_normal =True
        else:
            if_pretrain = False
            if_normal = False
        
        with torch.cuda.amp.autocast(enabled= True):
            # if FLAGS.mode == 'refine':

            if if_pretrain== True:
                reg_loss = model(target, it, if_normal, if_pretrain= if_pretrain, scene_and_vertices = scene_and_vertices)
                img_loss = 0 
                sds_loss = 0 
            if if_pretrain == False:
                sds_loss,img_loss, reg_loss = model(target, it, if_normal, if_pretrain= if_pretrain, scene_and_vertices=scene_and_vertices)
    
        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        
        total_loss = img_loss + reg_loss + sds_loss
        
        # model.geometry.decoder.net.params.grad /= 100
        if if_pretrain == True:
            scaler.scale(total_loss).backward()
            
        if if_pretrain == False:
            scaler.scale(total_loss).backward()
            img_loss_vec.append(img_loss.item())


        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================

        if if_normal == False and  if_pretrain == False:
            scaler.step(optimizer)
            optimizer.zero_grad()
          
        if if_normal == True or if_pretrain == True:
            if optimize_geometry:
                scaler.step(optimizer_mesh)
                # scheduler_mesh.step()
                optimizer_mesh.zero_grad()                

        scaler.update()

        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        # if it % log_interval == 0 and FLAGS.local_rank == 0 and if_pretrain == False:
        #     img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
        #     reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
        #     iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
        #     remaining_time = (FLAGS.iter-it)*iter_dur_avg
        #     if optimize_geometry:
        #         print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, mesh_lr=%.5f, time=%.1f ms, rem=%s, mat_lr=%.5f" % 
        #             (it, img_loss_avg, reg_loss_avg, optimizer_mesh.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time),optimizer.param_groups[0]['lr']))
        #     else:
        #         print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, time=%.1f ms, rem=%s, mat_lr=%.5f" % 
        #             (it, img_loss_avg, reg_loss_avg, iter_dur_avg*1000, util.time_to_text(remaining_time),optimizer.param_groups[0]['lr']))
    return geometry, opt_material

def seed_everything(seed, local_rank):
    random.seed(seed + local_rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True  

if __name__ == "__main__":
    config_base = OmegaConf.load('config/base.yaml')

    # specify config file

    cfg_sub = [el for el in sys.argv if '.yaml' in el][0]
    config_subject = OmegaConf.load(cfg_sub)
    sys.argv.remove(cfg_sub)
    
  # prevent config file info being saved to config_cli
    config_parent = OmegaConf.load(f'config/{config_subject.parent}.yaml')

    # update from cli  e.g. exp_name="test something"
    config_cli = OmegaConf.from_cli()

    FLAGS = OmegaConf.merge(config_base, config_parent, config_subject, config_cli)
    config_for_save = OmegaConf.merge(config_parent, config_subject, config_cli)

    FLAGS.mtl_override        = None                     # Override material of model                   
    FLAGS.env_scale           = 2.0                      # Env map intensity multiplier
    FLAGS.relight             = None                     # HDR environment probe(relight)
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [1, 50]
    FLAGS.learn_light         = False
    FLAGS.gpu_number          = 1
    FLAGS.sdf_init_shape_scale=[1.0, 1.0, 1.0]
    # FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1 
     
    if FLAGS.multi_gpu:
        FLAGS.gpu_number = int(os.environ["WORLD_SIZE"])
        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl", world_size = FLAGS.gpu_number, rank = FLAGS.local_rank)  
        torch.cuda.set_device(FLAGS.local_rank)
  
    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        # FLAGS.out_dir = Path(FLAGS.out_dir) / FLAGS.subject / FLAGS.mode / \
        #                 (datetime.strftime(datetime.now(), "%Y%m%d-%H%M_") + '_'.join(FLAGS.exp_name.split()))
        FLAGS.out_dir = Path(FLAGS.out_dir) / FLAGS.subject / FLAGS.mode / \
                        ('_'.join(FLAGS.exp_name.split()))

    FLAGS.data_dir = Path(FLAGS.data_dir) / FLAGS.subject
    mesh_name = FLAGS.subject
    if FLAGS.mode == 'geometry_modeling':
        FLAGS.base_mesh           = FLAGS.data_dir / f'{mesh_name}_norm.obj'
        FLAGS.smplx               = FLAGS.data_dir / 'smplx_param.pkl'
        FLAGS.hum_mesh            = FLAGS.data_dir / 'segms_hum.obj'
        FLAGS.obj_mesh            = FLAGS.data_dir / 'segms_obj.obj'
        FLAGS.faces_segm          = FLAGS.data_dir / 'faces_segms.npy'

    elif FLAGS.mode == "appearance_modeling":
        FLAGS.gt_mesh             = FLAGS.data_dir / f'{mesh_name}_norm.obj'
        FLAGS.gt_mtl              = FLAGS.data_dir / f'material0.mtl'
        FLAGS.geometry_dir = FLAGS.out_dir / FLAGS.exp_name if FLAGS.geometry_dir is None \
                        else FLAGS.out_dir.parent.parent / 'geometry_modeling' / FLAGS.geometry_dir
        FLAGS.base_mesh           = FLAGS.geometry_dir / f'dmtet_mesh/human.obj'
        FLAGS.base_mesh_obj       = FLAGS.geometry_dir / f'dmtet_mesh/object.obj'
        FLAGS.faces_segm          = FLAGS.data_dir / 'faces_segms.npy'
        FLAGS.smplx               = FLAGS.data_dir / 'smplx_param.pkl'

    elif FLAGS.mode == "refine":
        mesh_name_obj = FLAGS.subject_obj
        FLAGS.gt_mesh             = FLAGS.data_dir / f'{mesh_name}_norm.obj'
        FLAGS.gt_mtl              = FLAGS.data_dir / f'material.mtl'
        FLAGS.smplx               = FLAGS.data_dir / 'smplx_param.pkl'
        FLAGS.base_mesh           = Path(FLAGS.folder_hum_geo) / f'dmtet_mesh/human_cano.obj'
        FLAGS.base_mesh_obj       = Path(FLAGS.folder_obj_geo) / f'dmtet_mesh/object_cano.obj'
        FLAGS.base_mesh_tex           = Path(FLAGS.folder_hum_tex) / f'dmtet_mesh/human.ply'
        FLAGS.base_mesh_obj_tex       = Path(FLAGS.folder_obj_tex) / f'dmtet_mesh/object.ply'

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    seed_everything(FLAGS.seed, FLAGS.local_rank)
    
    FLAGS.out_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config_for_save, FLAGS.out_dir / 'config.yaml')

    # glctx = dr.RasterizeGLContext()
    glctx = dr.RasterizeCudaContext()
    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    dataset_train    = DatasetMesh(glctx, FLAGS, validate=False)
    dataset_validate = DatasetMesh(glctx, FLAGS, validate=True)
    dataset_gif      = DatasetMesh(glctx, FLAGS, gif=True)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    if FLAGS.mode == 'appearance_modeling' and FLAGS.base_mesh is not None:
        if FLAGS.learn_light:
            lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=1)
        else:
            # lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
            lgt = None
    else:
        lgt = None
        # lgt1 = light.load_env(FLAGS.envmap1, scale=FLAGS.env_scale)
    
    if FLAGS.sdf_init_shape in ['ellipsoid', 'cylinder', 'custom_mesh'] and FLAGS.mode == 'geometry_modeling':
        if FLAGS.sdf_init_shape == 'ellipsoid':
            init_shape = o3d.geometry.TriangleMesh.create_sphere(1)
            base_shape = get_normalize_mesh(FLAGS.base_mesh)
        elif FLAGS.sdf_init_shape == 'cylinder':
            init_shape = o3d.geometry.TriangleMesh.create_cylinder(radius=0.75, height=0.8, resolution=20, split=4, create_uv_map=False)
            init_shape_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.75, height=0.8, resolution=20, split=4, create_uv_map=False)
        elif FLAGS.sdf_init_shape == 'custom_mesh':
            if FLAGS.base_mesh:
                init_shape = get_normalize_mesh(FLAGS.base_mesh)
                init_shape_hum = get_normalize_mesh(FLAGS.hum_mesh)
                init_shape_obj = get_normalize_mesh(FLAGS.obj_mesh)
                gt_faces_segm = np.load(FLAGS.faces_segm)
            else:
                assert False, "[Error] The path of custom mesh is invalid ! (geometry modeling)"
        else:
            assert False, "Invalid init type"
  
        vertices = np.asarray(init_shape.vertices)
        faces = np.asarray(init_shape.triangles)
        vertices[...,0]=vertices[...,0] * FLAGS.sdf_init_shape_scale[0]
        vertices[...,1]=vertices[...,1] * FLAGS.sdf_init_shape_scale[1]
        vertices[...,2]=vertices[...,2] * FLAGS.sdf_init_shape_scale[2]
        vertices = vertices @ util.rotate_x_2(np.deg2rad(FLAGS.sdf_init_shape_rotate_x))
        vertices[...,1]=vertices[...,1] + FLAGS.translation_y
        vertices[...,2]=vertices[...,2] + FLAGS.translation_z
        init_shape.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
        points_surface = np.asarray(init_shape.sample_points_poisson_disk(5000).points)
        init_shape = o3d.t.geometry.TriangleMesh.from_legacy(init_shape)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(init_shape)
        # scene, sampled points on surface, verts, faces, vert segmentations, object label

        vertices_hum = np.asarray(init_shape_hum.vertices)
        faces_hum = np.asarray(init_shape_hum.triangles)
        vertices_obj = np.asarray(init_shape_obj.vertices)
        faces_obj = np.asarray(init_shape_obj.triangles)

        scene_and_vertices = [scene, points_surface, 
                              torch.from_numpy(vertices).float().cuda(), torch.from_numpy(faces).long().cuda(), 
                              torch.from_numpy(gt_faces_segm)[..., None].float().cuda(),
                              mesh.Mesh(torch.from_numpy(vertices_hum).float().cuda(), torch.from_numpy(faces_hum).long().cuda()),
                              mesh.Mesh(torch.from_numpy(vertices_obj).float().cuda(), torch.from_numpy(faces_obj).long().cuda())]

        # points_surface = np.asarray(base_shape.sample_points_poisson_disk(5000).points)
        # base_shape = o3d.t.geometry.TriangleMesh.from_legacy(base_shape)
        # scene = o3d.t.geometry.RaycastingScene()
        # scene.add_triangles(base_shape)
        
        base_scene_and_vertices = scene_and_vertices

    guidance = StableDiffusion(device = 'cuda',
                               mode = FLAGS.mode, 
                               text = FLAGS.text,
                               text_comp = FLAGS.text_obj,
                               add_directional_text = FLAGS.add_directional_text,
                               batch = FLAGS.batch,
                               guidance_weight = FLAGS.guidance_weight,
                               sds_weight_strategy = FLAGS.sds_weight_strategy,
                               early_time_step_range = FLAGS.early_time_step_range,
                               late_time_step_range= FLAGS.late_time_step_range,
                               negative_text = FLAGS.negative_text,
                               negative_text_comp = FLAGS.negative_text_obj,
                               sd_model=FLAGS.sd_model,
                               enable_controlnet=FLAGS.enable_controlnet,
                               use_inpaint=FLAGS.use_inpaint,
                               repaint=FLAGS.repaint,
                               use_legacy=FLAGS.use_legacy,
                               use_taesd=FLAGS.use_taesd)
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad_(False)

        
    if FLAGS.mode == 'geometry_modeling' :   
        geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        mat = initial_guness_material(geometry, True, FLAGS)
        # Run optimization
        geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, 
                        FLAGS, optimize_light=FLAGS.learn_light,optimize_geometry= not  FLAGS.lock_pos, guidance= guidance,
                        scene_and_vertices= scene_and_vertices, base_scene_and_vertices=base_scene_and_vertices)

        if FLAGS.local_rank == 0 and FLAGS.validate:
            validate(glctx, geometry, mat, lgt, dataset_gif, str(FLAGS.out_dir / "validate"), FLAGS)

        # Create textured mesh from result
        if FLAGS.local_rank == 0:
            base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)
            obj_mesh = xatlas_uvmap_obj(glctx, geometry, mat, FLAGS)
            if FLAGS.enable_canonical:
                base_mesh_cano = xatlas_uvmap(glctx, geometry, mat, FLAGS, canonical=True)
                obj_mesh_cano = xatlas_uvmap_obj(glctx, geometry, mat, FLAGS, canonical=True)

        # # Free temporaries / cached memory 
        torch.cuda.empty_cache()
        mat['kd_ks_normal'].cleanup()
        del mat['kd_ks_normal']
    

        if FLAGS.local_rank == 0:
            # Dump mesh for debugging.
            os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
            obj.write_obj(str(FLAGS.out_dir / "dmtet_mesh"), base_mesh, name='human')
            obj.write_obj(str(FLAGS.out_dir / "dmtet_mesh"), obj_mesh, name='object')

            if FLAGS.enable_canonical:
                obj.write_obj(str(FLAGS.out_dir / "dmtet_mesh/"), base_mesh_cano, name='human_cano')
                obj.write_obj(str(FLAGS.out_dir / "dmtet_mesh/"), obj_mesh_cano, name='object_cano')

    elif FLAGS.mode == 'appearance_modeling':
        # ==============================================================================================
        #  Train with fixed topology (mesh)
        # ==============================================================================================
        if FLAGS.base_mesh is None or FLAGS.base_mesh_obj is None:
            assert False, "[Error] The path of custom mesh is invalid ! (appearance modeling)"
        
        if FLAGS.enable_canonical:
        # Load initial guess mesh from file
            if FLAGS.layer:
                gt_mesh = mesh.load_mesh(FLAGS.gt_mesh)
            else: 
                gt_mesh = mesh.load_mesh(FLAGS.gt_mesh, FLAGS.gt_mtl)
            # load gt segm
            gt_faces_segm = np.load(FLAGS.faces_segm)
            gt_mesh.f_seg = torch.from_numpy(gt_faces_segm)[..., None].float().cuda()
            base_mesh = mesh.load_mesh(str(FLAGS.base_mesh).replace('.obj', '_cano.obj'))
            base_mesh_obj = mesh.load_mesh(str(FLAGS.base_mesh_obj).replace('.obj', '_cano.obj'))
        else:
            gt_mesh = mesh.load_mesh(FLAGS.gt_mesh, FLAGS.gt_mtl)
            # load gt segm
            gt_faces_segm = np.load(FLAGS.faces_segm)
            gt_mesh.f_seg = torch.from_numpy(gt_faces_segm)[..., None].float().cuda()
            base_mesh = mesh.load_mesh(str(FLAGS.base_mesh))
            base_mesh_obj = mesh.load_mesh(str(FLAGS.base_mesh_obj))

        geometry = DLMesh(base_mesh, base_mesh_obj, gt_mesh, FLAGS)
 
        # mat = initial_guness_material(geometry, False, FLAGS, init_mat=base_mesh.material)
        mat = initial_guness_material(geometry, True, FLAGS)
        geometry, mat = optimize_mesh(glctx, 
                                      geometry, 
                                      mat, 
                                      lgt, 
                                      dataset_train, 
                                      dataset_validate, 
                                      FLAGS, 
                                      optimize_light=FLAGS.learn_light,
                                      optimize_geometry= False, 
                                      guidance= guidance
                                      )
        
        # ==============================================================================================
        #  Validate
        # ==============================================================================================
        if FLAGS.validate and FLAGS.local_rank == 0:
            if FLAGS.relight != None:       
                relight = light.load_env(FLAGS.relight, scale=FLAGS.env_scale)
            else:
                relight = None
            validate(glctx, geometry, mat, lgt, dataset_gif, str(FLAGS.out_dir / "validate"), FLAGS, relight)
        # if FLAGS.local_rank == 0:
            # base_mesh = xatlas_uvmap1(glctx, geometry, mat, FLAGS)
        torch.cuda.empty_cache()
        mat['kd_ks_normal'].cleanup()
        del mat['kd_ks_normal']
        # lgt = lgt.clone()
        if FLAGS.local_rank == 0:
            os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh.v_pos.detach().cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh.t_pos_idx.detach().cpu().numpy())
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh.v_color.detach().cpu().numpy())
            o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/human.ply"), mesh_o3d, write_ascii=True)

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_obj.v_pos.detach().cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_obj.t_pos_idx.detach().cpu().numpy())
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_obj.v_color.detach().cpu().numpy())
            o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/object.ply"), mesh_o3d, write_ascii=True)

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_comp.v_pos.detach().cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_comp.t_pos_idx.detach().cpu().numpy())
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_comp.v_color.detach().cpu().numpy())
            o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/comp.ply"), mesh_o3d, write_ascii=True)

            if FLAGS.enable_canonical:
                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_posed.v_pos.detach().cpu().numpy())
                mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_posed.t_pos_idx.detach().cpu().numpy())
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_posed.v_color.detach().cpu().numpy())
                o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/human_posed.ply"), mesh_o3d, write_ascii=True)

                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_obj_posed.v_pos.detach().cpu().numpy())
                mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_obj_posed.t_pos_idx.detach().cpu().numpy())
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_obj_posed.v_color.detach().cpu().numpy())
                o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/object_posed.ply"), mesh_o3d, write_ascii=True)

                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_comp_posed.v_pos.detach().cpu().numpy())
                mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_comp_posed.t_pos_idx.detach().cpu().numpy())
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_comp_posed.v_color.detach().cpu().numpy())
                o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/comp_posed.ply"), mesh_o3d, write_ascii=True)
    
    elif FLAGS.mode == 'refine':
        base_mesh = mesh.load_mesh(str(FLAGS.base_mesh), vcolor_filename=str(FLAGS.base_mesh_tex))
        base_mesh_obj = mesh.load_mesh(str(FLAGS.base_mesh_obj), vcolor_filename=str(FLAGS.base_mesh_obj_tex))
        gt_mesh = mesh.load_mesh(FLAGS.gt_mesh, FLAGS.gt_mtl)

        scene_and_vertices = [base_mesh, base_mesh_obj]
        
        # geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        geometry = DLMesh(base_mesh, base_mesh_obj, gt_mesh, FLAGS)
        mat = initial_guness_material(geometry, True, FLAGS)
        # Run optimization
        geometry, mat = optimize_mesh(glctx, 
                                      geometry, 
                                      mat, 
                                      lgt, 
                                      dataset_train, 
                                      dataset_validate, 
                                      FLAGS, 
                                      optimize_light=FLAGS.learn_light,
                                      optimize_geometry= False, 
                                      guidance= guidance,
                                      scene_and_vertices= scene_and_vertices
                                      )

        # ==============================================================================================
        #  Validate
        # ==============================================================================================
        if FLAGS.validate and FLAGS.local_rank == 0:
            if FLAGS.relight != None:       
                relight = light.load_env(FLAGS.relight, scale=FLAGS.env_scale)
            else:
                relight = None
            validate(glctx, geometry, mat, lgt, dataset_gif, str(FLAGS.out_dir / "validate"), FLAGS, relight)
        # if FLAGS.local_rank == 0:
            # base_mesh = xatlas_uvmap1(glctx, geometry, mat, FLAGS)
        torch.cuda.empty_cache()
        mat['kd_ks_normal'].cleanup()
        del mat['kd_ks_normal']
        # lgt = lgt.clone()
        if FLAGS.local_rank == 0:
            os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh.v_pos.detach().cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh.t_pos_idx.detach().cpu().numpy())
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh.v_color.detach().cpu().numpy())
            o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/human.ply"), mesh_o3d, write_ascii=True)

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_obj.v_pos.detach().cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_obj.t_pos_idx.detach().cpu().numpy())
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_obj.v_color.detach().cpu().numpy())
            o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/object.ply"), mesh_o3d, write_ascii=True)

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_comp.v_pos.detach().cpu().numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_comp.t_pos_idx.detach().cpu().numpy())
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_comp.v_color.detach().cpu().numpy())
            o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/comp.ply"), mesh_o3d, write_ascii=True)

            if FLAGS.enable_canonical:
                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_posed.v_pos.detach().cpu().numpy())
                mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_posed.t_pos_idx.detach().cpu().numpy())
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_posed.v_color.detach().cpu().numpy())
                o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/human_posed.ply"), mesh_o3d, write_ascii=True)

                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_obj_posed.v_pos.detach().cpu().numpy())
                mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_obj_posed.t_pos_idx.detach().cpu().numpy())
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_obj_posed.v_color.detach().cpu().numpy())
                o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/object_posed.ply"), mesh_o3d, write_ascii=True)

                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(geometry.mesh_comp_posed.v_pos.detach().cpu().numpy())
                mesh_o3d.triangles = o3d.utility.Vector3iVector(geometry.mesh_comp_posed.t_pos_idx.detach().cpu().numpy())
                mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(geometry.mesh_comp_posed.v_color.detach().cpu().numpy())
                o3d.io.write_triangle_mesh(str(FLAGS.out_dir / "dmtet_mesh/comp_posed.ply"), mesh_o3d, write_ascii=True)

    else:
        assert False, "Invalid mode type"
   
    

