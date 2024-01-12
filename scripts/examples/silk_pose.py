# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from common import get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image
import yaml
from pathlib import Path
import numpy as np
import torch
import random
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from pose_utils import compute_epipolar_error, compute_pose_error, estimate_pose, pose_estimation_metrics
from Scannet import Scannet
from torch.utils.data import DataLoader


DATA_PATH = "silk/datasets"

@torch.no_grad()
def estimate_pose_errors(config, model, data_loader, device):
    
    all_metrics = []
    len_pairs = len(data_loader.dataset)

    for batch in tqdm(data_loader):
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        sparse_positions_0, sparse_descriptors_0 = model(batch["inp0"])
        sparse_positions_1, sparse_descriptors_1 = model(batch["inp1"])

        # get matches
        matches = SILK_MATCHER(sparse_descriptors_0[0], sparse_descriptors_1[0])

        # preprocess keypoints
        kpts0, kpts1 = sparse_positions_0[0].squeeze(), sparse_positions_1[0].squeeze()
        kpts0, kpts1 = kpts0[:, :2], kpts1[:, :2]
        
        mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
        mkpts0, mkpts1 = mkpts0 + config["matcher"]["bias"] , mkpts1 + config["matcher"]["bias"]
        mkpts0, mkpts1 = mkpts0[:, [1, 0]], mkpts1[:, [1, 0]]

        kpts0, kpts1 = kpts0.squeeze().detach().cpu().numpy(), kpts1.squeeze().detach().cpu().numpy()
        mkpts0, mkpts1 = mkpts0.squeeze().detach().cpu().numpy(), mkpts1.squeeze().detach().cpu().numpy()

        # get epipolar errors
        epi_errs = compute_epipolar_error(mkpts0.copy(), mkpts1.copy(), batch['T_0to1'], batch['K0'], batch['K1'])
        correct = epi_errs < config["data"]["epi_thrsehold"]
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        # get pose errors
        ret = estimate_pose(mkpts0.copy(), mkpts1.copy(), batch['K0'], batch['K1'], config["matcher"]["ransac_thresh"])
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(batch['T_0to1'], R, t)

        out_eval = {'error_t': err_t,
                'error_R': err_R,
                'precision': precision,
                'matching_score': matching_score,
                'num_correct': num_correct,
                'epipolar_errors': epi_errs}
    
        all_metrics.append(out_eval)

    pose_estimation_metrics(all_metrics, len_pairs)

    



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = "silk/scannet_pose.yaml"
    
    config = yaml.load(stream=open(config_path, 'r'), Loader=yaml.FullLoader)

    model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))

    dataset = Scannet(config["data"], device=device)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.batch_collator, pin_memory=True)

    estimate_pose_errors(config, model, dataloader, device)