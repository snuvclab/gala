import torch

# code from https://github.com/huangyangyi/TeCH/blob/main/core/lib/dmtet_network.py
###############################################################################
# Compact tet grid
###############################################################################

def compact_tets(pos_nx3, sdf_n, tet_fx4):
    with torch.no_grad():
        # Find surface tets
        occ_n = sdf_n > 0
        occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)  # one value per tet, these are the surface tets

        valid_vtx = tet_fx4[valid_tets].reshape(-1)
        unique_vtx, idx_map = torch.unique(valid_vtx, dim=0, return_inverse=True)
        new_pos = pos_nx3[unique_vtx]
        new_sdf = sdf_n[unique_vtx]
        new_tets = idx_map.reshape(-1, 4)
        return new_pos, new_sdf, new_tets


###############################################################################
# Subdivide volume
###############################################################################

def sort_edges(edges_ex2):
    with torch.no_grad():
        order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
        order = order.unsqueeze(dim=1)
        a = torch.gather(input=edges_ex2, index=order, dim=1)
        b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
    return torch.stack([a, b], -1)


def batch_subdivide_volume(tet_pos_bxnx3, tet_bxfx4):
    device = tet_pos_bxnx3.device
    # get new verts
    tet_fx4 = tet_bxfx4[0]
    edges = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]
    all_edges = tet_fx4[:, edges].reshape(-1, 2)
    all_edges = sort_edges(all_edges)
    unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
    idx_map = idx_map + tet_pos_bxnx3.shape[1]
    all_values = tet_pos_bxnx3
    mid_points_pos = all_values[:, unique_edges.reshape(-1)].reshape(
        all_values.shape[0], -1, 2,
        all_values.shape[-1]).mean(2)
    new_v = torch.cat([all_values, mid_points_pos], 1)

    # get new tets

    idx_a, idx_b, idx_c, idx_d = tet_fx4[:, 0], tet_fx4[:, 1], tet_fx4[:, 2], tet_fx4[:, 3]
    idx_ab = idx_map[0::6]
    idx_ac = idx_map[1::6]
    idx_ad = idx_map[2::6]
    idx_bc = idx_map[3::6]
    idx_bd = idx_map[4::6]
    idx_cd = idx_map[5::6]

    tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
    tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
    tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
    tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
    tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
    tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
    tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
    tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)

    tet_np = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0)
    tet_np = tet_np.reshape(1, -1, 4).expand(tet_pos_bxnx3.shape[0], -1, -1)
    tet = tet_np.long().to(device)

    return new_v, tet


