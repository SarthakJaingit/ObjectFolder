"""Data loader for OSF data."""

import os
import torch
import numpy as np
import imageio 
import json

import cam_utils


trans_t = lambda t: torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]
], dtype=torch.float)

rot_phi = lambda phi: torch.tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]
], dtype=torch.float)

rot_theta = lambda th: torch.tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]
], dtype=torch.float)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def convert_cameras_to_nerf_format(anno):
    """
    Args:
        anno: List of annotations for each example. Each annotation is represented by a
            dictionary that must contain the key `RT` which is the world-to-camera
            extrinsics matrix with shape [3, 4], in [right, down, forward] coordinates.

    Returns:
        c2w: [N, 4, 4] np.float32. Array of camera-to-world extrinsics matrices in
            [right, up, backwards] coordinates.
    """
    c2w_list = []
    for a in anno:
        # Convert from w2c to c2w.
        w2c = np.array(a['RT'] + [[0.0, 0.0, 0.0, 1.0]])
        c2w = cam_utils.w2c_to_c2w(w2c)

        # Convert from [right, down, forwards] to [right, up, backwards]
        c2w[:3, 1] *= -1  # down -> up
        c2w[:3, 2] *= -1  # forwards -> back
        c2w_list.append(c2w)
    c2w = np.array(c2w_list)
    return c2w


def load_osf_data(test_file_path):
    """
    Returns:
        imgs: [H, W, 4] np.float32. Array of images in RGBA format, and normalized
            between [0, 1].
    """

    splits = ['test']
    metas = {}
    for s in splits:
        with open(test_file_path, 'r') as fp:
            anno = json.load(fp)

        # Convert camera matrices into NeRF format.
        c2w = convert_cameras_to_nerf_format(anno)
        for i in range(len(anno)):
            anno[i]['c2w'] = c2w[i]

        metas[s] = anno

    all_poses = []
    all_metadata = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        poses = []
        metadata = []
        skip = 1

        for frame in meta[::skip]:
            fname = os.path.join(frame['filename'])
            poses.append(frame['c2w'])  # [4, 4]
            metadata.append(frame['light_pos'])  # [3,]

        poses = np.array(poses).astype(np.float32)  # [N, 4, 4]
        metadata = np.array(metadata).astype(np.float32)  # [N, 3]
        counts.append(counts[-1] + poses.shape[0])
        all_poses.append(poses)
        all_metadata.append(metadata)

    # Create a list where each element contains example indices for each split.
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]

    imgs = None
    poses = np.concatenate(all_poses, 0)  # [N, 4, 4]
    metadata = np.concatenate(all_metadata, 0)  # [N, 3]

    # Extract the height and width from the shape of the first image example.
    H, W = 256, 256

    # Compute the focal length.
    focal = meta[0]['K'][0][0]

    return imgs, poses, [H, W, focal], i_split, metadata
