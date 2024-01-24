from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3, resize_image
from pathlib import Path
from torchvision import transforms
import warnings
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import trimesh
import argparse
import cv2
from deformer.smplx import SMPLX
from deformer.lib import rotation_converter, helpers


# code from multiview smplifyx
def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            # body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
            #                          8, 1, 4, 7, 56, 57, 58, 59],
            #                         dtype=np.int32)

            # TODO: find a exact mapping between 3d openpose and smplx joints, currently found with Nogada
            body_mapping = np.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5,
                                     8, 1, 4, 7, 75, 82, 106, 125],
                                    dtype=np.int32)
            mapping = [body_mapping]
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


class OpenPoseDetectorRaw(OpenposeDetector): 
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]   

    # TODO: batch implementation
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, 
                 include_body=True, include_hand=False, include_face=False, hand_and_face=None, 
                 output_type="pil", **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        poses = self.detect_poses(input_image, include_hand, include_face)

        return poses

    @classmethod
    def draw_bodypose(cls, keypoints: np.ndarray):
        H, W = (512, 512)
        keypoints = 0.5 * (keypoints + 1)  # [-1, 1] => [0, 1]
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        stickwidth = 4
        for (k1_index, k2_index), color in zip(cls.limbSeq, cls.colors):
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1.sum() == 0 or keypoint2.sum() == 0:
                continue

            Y = np.array([keypoint1[0], keypoint2[0]]) * W
            X = np.array([keypoint1[1], keypoint2[1]]) * H
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

        for keypoint, color in zip(keypoints, cls.colors):
            if keypoint.sum() == 0:
                continue

            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

        return transforms.ToTensor()(canvas).to(dtype=torch.float16)
    
    @classmethod
    def draw_bodypose_wo_face(cls, keypoints: np.ndarray):
        H, W = (512, 512)
        keypoints = 0.5 * (keypoints + 1)  # [-1, 1] => [0, 1]
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        stickwidth = 4
        for (k1_index, k2_index), color in zip(cls.limbSeq[:12], cls.colors[:12]):
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1.sum() == 0 or keypoint2.sum() == 0:
                continue

            Y = np.array([keypoint1[0], keypoint2[0]]) * W
            X = np.array([keypoint1[1], keypoint2[1]]) * H
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

        for keypoint, color in zip(keypoints[1:13], cls.colors[1:13]):
            if keypoint.sum() == 0:
                continue

            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

        return transforms.ToTensor()(canvas).to(dtype=torch.float16)

class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, required=True)
    parser.add_argument("--vis_2d", action="store_true")
    parser.add_argument("--vis_3d", action="store_true")
    args = parser.parse_args()

    root_dir = args.root_dir
    vis_2d = args.vis_2d
    vis_3d = args.vis_3d


    # 3d openpose for smplx
    mapping = smpl_to_openpose(openpose_format='coco19')
    smplx_config = {
            'topology_path': "deformer/data/SMPL_X_template_FLAME_uv.obj",
            'smplx_model_path': "deformer/data/SMPLX_NEUTRAL_2020.npz",
            'extra_joint_path': "deformer/data/smplx_extra_joints.yaml",
            'j14_regressor_path': "deformer/data/SMPLX_to_J14.pkl",
            'mano_ids_path': "deformer/data/MANO_SMPLX_vertex_ids.pkl",
            'flame_vertex_masks_path': "deformer/data/FLAME_masks.pkl",
            'flame_ids_path': "deformer/data/SMPL-X__FLAME_vertex_ids.npy",
            "head_verts_path": "deformer/data/head_verts_idx.npy",
            'n_shape': 10,
            'n_exp': 10
        }
    smplx_config = Struct(**smplx_config)

    # set canonical space
    smplx = SMPLX(smplx_config)
    pose = torch.zeros([55,3], dtype=torch.float32, ) # 55
    pose_axispca = torch.zeros([29,3], dtype=torch.float32, )
    angle = 0*np.pi/180.
    pose[1, 2] = angle
    pose[2, 2] = -angle
    # pose_axispca[1, 2] = angle
    # pose_axispca[2, 2] = -angle
    pose_euler = pose.clone()
    pose = rotation_converter.batch_euler2matrix(pose)
    pose = pose[None,...]
    xyz_c, _, joints, A, T, shape_offsets, pose_offsets = smplx(full_pose = pose, return_T=True, transl=torch.tensor([0, 0.4, 0],dtype=torch.float32, ))

    smplx_op = joints[0][mapping][:18]


    # controlnet preprocessor
    # processor_ids = ["openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand"]
    processor_id = 'openpose'
    openpose_detector = OpenPoseDetectorRaw.from_pretrained("lllyasviel/Annotators")
    device = 'cuda:0'

    
    openpose_infos = {}
    radius = 2
    image_dir = root_dir / 'render' / 'images'
    for image_path in image_dir.glob('180_*.png'):
        yaw = image_path.stem.split('_')[-1]
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        h, w = image.size
        try:
            body_keypoints = openpose_detector(image)[0].body.keypoints
            openpose_joints = []
            for keypoint, color in zip(body_keypoints, OpenPoseDetectorRaw.colors):
                if keypoint is None:
                    openpose_joints.append([0.0, 0.0])
                else:
                    x, y = keypoint.x, keypoint.y
                    openpose_joints.append([x, y])
                    # draw.ellipse((x * w - radius, y * h - radius, x * w + radius, y * h + radius), fill=tuple(color))

            # # draw limbs
            # stickwidth = 2
            # for (k1_index, k2_index), color in zip(limbSeq, colors):
            #     keypoint1 = body_keypoints[k1_index - 1]
            #     keypoint2 = body_keypoints[k2_index - 1]

            #     if keypoint1 is None or keypoint2 is None:
            #         continue

            #     w1, h1 = keypoint1.x * w, keypoint1.y * h
            #     w2, h2 = keypoint2.x * w, keypoint2.y * h

            #     draw.line([(w1, h1), (w2, h2)], fill=tuple(color), width=1)

            openpose_joints = np.array(openpose_joints)
            openpose_infos[yaw] = torch.tensor(openpose_joints, dtype=torch.float32, device=device)
        except IndexError:
            pass

    # optimization code
    joints_3d = torch.nn.Parameter(torch.zeros([18, 3], device=device))
    num_iter = 100
    mse_loss = nn.MSELoss()
    optim = torch.optim.RMSprop([joints_3d], lr=0.01, momentum=0, weight_decay=0)
    losses = []
    r_xpi = R.from_euler('x', 180, degrees=True).as_matrix()
    for _ in range(num_iter):
        loss = 0.0
        for yaw in openpose_infos.keys():
            r_ypi = R.from_euler('y', int(yaw), degrees=True).as_matrix()
            r = r_xpi @ r_ypi
            r = torch.tensor(r, dtype=torch.float32, device=device)  

            # Project 3D joint positions into 2D using the curren
            # t camera matrix
            projected_points = (joints_3d @ r.T)[:, :2]
            projected_points = 0.5 * (projected_points + 1)  # [-1, 1] => [0, 1]

            openpose_joints = openpose_infos[yaw]
            op_valid_mask = openpose_joints.sum(axis=-1) != 0
            loss += ((projected_points - openpose_joints)[op_valid_mask] ** 2).mean()
        loss /= len(openpose_infos.keys())
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    joints_3d_np = joints_3d.detach().cpu().numpy()
    np.save(root_dir / 'op_3d.npy', joints_3d_np)

    smplx_op = smplx_op.detach().cpu().numpy()
    np.save(root_dir / 'cano_op_3d.npy', smplx_op)

    ''' draw joints_3d projected in the given yaw, azimuth '''
    if vis_2d:
        yaw = 0
        azimuth = 0
        canvas = Image.open(image_dir / f'{(azimuth + 180) % 360:03d}_{yaw:03d}.png')
        # canvas = Image.new('RGB', (h, w))
        draw = ImageDraw.Draw(canvas)

        r_xpi = R.from_euler('x', azimuth + 180, degrees=True).as_matrix()
        r_ypi = R.from_euler('y', yaw, degrees=True).as_matrix()
        r = r_xpi @ r_ypi
        r = torch.tensor(r, dtype=torch.float32, device=device)  
        joints_3d_rot = joints_3d @ r.T

        joints_2d = 0.5 * (joints_3d_rot[:, :2] + 1)
        joints_2d[:, 0] *= w
        joints_2d[:, 1] = h * joints_2d[:, 1]
        for keypoint, color in zip(joints_2d, OpenPoseDetectorRaw.colors):
            # TODO: exception handling for the case that joint is invisible in all views, we need at least two views.
            px, py = keypoint[0], keypoint[1]
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=tuple(color))

        # draw limbs
        stickwidth = 2
        for (k1_index, k2_index), color in zip(OpenPoseDetectorRaw.limbSeq, OpenPoseDetectorRaw.colors):
            keypoint1 = joints_2d[k1_index - 1]
            keypoint2 = joints_2d[k2_index - 1]

            w1, h1 = keypoint1[0], keypoint1[1]
            w2, h2 = keypoint2[0], keypoint2[1]

            draw.line([(w1, h1), (w2, h2)], fill=tuple([int(float(c) * 0.6) for c in color]), width=1)
        canvas.show()

    ''' draw joints_3d projected in the given yaw, azimuth '''
    if vis_3d:
        vp = vedo.Plotter(title="", size=(750, 1500), axes=0, bg='white', interactive=True)
        vis_list = []

        # load mesh
        mesh = vedo.load(str(root_dir / f'{root_dir.stem}_100k_norm.obj'))
        # mesh.texture(str(root_dir / 'material_0.jpeg'))
        mesh.alpha(0.6)
        vis_list.append(mesh)

        # load 3D skeleton
        joint_points = [vedo.Points(joint, c=c) for joint, c in zip(joints_3d_np, OpenPoseDetectorRaw.colors)]
        limb_lines = [vedo.Line([joints_3d_np[k1_index - 1], joints_3d_np[k2_index - 1]], 
                                c=tuple((0.6 *np.asarray(color)).astype(np.uint8)),
                                lw=3) 
                    for (k1_index, k2_index), color in zip(OpenPoseDetectorRaw.limbSeq, OpenPoseDetectorRaw.colors)]
        skeleton_3d = joint_points + limb_lines
        vis_list += skeleton_3d

        vp += vis_list
        vp.show()