# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import torch

from . import texture
from . import mesh
from . import material

######################################################################################
# Utility functions
######################################################################################

def _find_mat(materials, name):
    for mat in materials:
        if mat['name'] == name:
            return mat
    return materials[0] # Materials 0 is the default

######################################################################################
# Create mesh object from objfile
######################################################################################

def load_obj(filename, clear_ks=True, mtl_override=None, vcolor_filename=None):
    obj_path = os.path.dirname(filename)

    # Read entire file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load materials
    all_materials = [
        # {
        #     'name' : '_default_mat',
        #     'bsdf' : 'pbr',
        #     'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
        #     'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        # }
    ]
    if mtl_override is None: 
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'mtllib':
                all_materials += material.load_mtl(os.path.join(obj_path, line.split()[1]), clear_ks) # Read in entire material library
    else:
        all_materials += material.load_mtl(mtl_override)

    # load vertices
    vertices, texcoords, normals, v_color  = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue
        
        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
            if len(line.split()) == 7:
                v_color.append([float(v) for v in line.split()[4:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'usemtl': # Track used materials
            mat = _find_mat(all_materials, line.split()[1])
            if not mat in used_materials:
                used_materials.append(mat)
            activeMatIdx = used_materials.index(mat)
        elif prefix == 'f': # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            
            for i in range(nv - 2): # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])

                try:
                    n0 = int(vv[2]) - 1 if vv[2] != "" else -1
                    n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                    n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                    nfaces.append([n0, n1, n2])
                except IndexError:
                    n0, n1, n2 = v0, v1, v2
                    nfaces.append([n0, n1, n2])
                    
    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)

    
    if vcolor_filename is not None:
        v_color = []
        with open(vcolor_filename, 'r') as f:
            lines = f.readlines()
            for line in lines[13:]:
                if len(line.split()) == 0:
                    continue
                if len(line.split()) == 6:
                    v_color.append([float(v)/255 for v in line.split()[3:]])


    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    elif len(used_materials) == 1:
        uber_material = used_materials[0]
    else:
        uber_material = None

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda') if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device='cuda') if len(normals) > 0 else None
    v_color = torch.tensor(v_color, dtype=torch.float32, device='cuda') if len(v_color) > 0 else None
    
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None

    return mesh.Mesh(v_pos=vertices, t_pos_idx=faces, v_nrm=normals, t_nrm=nfaces, v_tex=texcoords, t_tex_idx=tfaces, material=uber_material, v_color=v_color)


######################################################################################
# Save mesh object to objfile
######################################################################################

def write_obj(folder, mesh, save_material=True, name='mesh'):
    obj_file = os.path.join(folder, name + '.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write(f'mtllib {name}.mtl\n')
        f.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None
        # v_tex = None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        # if v_tex is not None:
        #     print("    writing %d texcoords" % len(v_tex))
        #     assert(len(t_pos_idx) == len(t_tex_idx))
        #     for v in v_tex:
        #         f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    if save_material:
        mtl_file = os.path.join(folder, name + '.mtl')
        print("Writing material: ", mtl_file)
        material.save_mtl(mtl_file, mesh.material)

    print("Done exporting mesh")


def write_ply(folder, mesh, save_material=False, name='mesh'):
    import numpy as np
    obj_file = os.path.join(folder, name + '.ply')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        # v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_color = mesh.v_color.detach().cpu().numpy() * 255 if mesh.v_color is not None else None
        # v_tex = None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        # t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        # t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        f.write(f'number of vertices: {len(v_pos)}\n')
        f.write(f'number of faces: {len(t_pos_idx)}\n')

        print("    writing %d vertices" % len(v_pos))
        for v in np.concatenate([v_pos, v_color], axis=1):
            f.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], v[3], v[4], v[5]))
       
        # if v_color is not None:
        #     print("    writing %d vertex colors" % len(v_tex))
        #     assert(len(v_color) == len(t_tex_idx))
            # for v in v_tex:
            #     f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        # if v_nrm is not None:
        #     print("    writing %d normals" % len(v_nrm))
        #     assert(len(t_pos_idx) == len(t_nrm_idx))
        #     for v in v_nrm:
        #         f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        # f.write("s 1 \n")
        # f.write("g pMesh1\n")
        # f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s' % (str(t_pos_idx[i][j])))
            f.write("\n")

    if save_material:
        mtl_file = os.path.join(folder, name + '.mtl')
        print("Writing material: ", mtl_file)
        material.save_mtl(mtl_file, mesh.material)

    print("Done exporting mesh")
