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
from tqdm import tqdm
from hpatches_utils import compute_repeatability, estimate_homography, hpatches_auc, mean_matching_acc, hpatches_metrics
from HPatches import HPatches
from torch.utils.data import DataLoader


DATA_PATH = "silk/datasets"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "silk/pvgg-4.ckpt"
)

@torch.no_grad()
def estimate_hpatches_metrics(config, model, data_loader, device):
    
    repeatability = []
    homogaphy_est_acc = []
    homography_est_err = []
    avg_pre_match_points = 0.0
    avg_post_match_points = 0.0
    MMA = 0.0
    num_matches = 0.0

    for batch in tqdm(data_loader):
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        sparse_positions_0, sparse_descriptors_0 = model(batch["image"])
        sparse_positions_1, sparse_descriptors_1 = model(batch["warped_image"])

        # get matches
        matches = SILK_MATCHER(sparse_descriptors_0[0], sparse_descriptors_1[0])

        # preprocess keypoints
        sparse_positions_0 = from_feature_coords_to_image_coords(model, sparse_positions_0)
        sparse_positions_1 = from_feature_coords_to_image_coords(model, sparse_positions_1)

        kpts0, kpts1 = sparse_positions_0[0].squeeze(), sparse_positions_1[0].squeeze()
        kpts0, kpts1 = kpts0[:, :2], kpts1[:, :2]
        
        mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
        mkpts0, mkpts1 = mkpts0[:, [1, 0]], mkpts1[:, [1, 0]]

        kpts0, kpts1 = kpts0.squeeze().detach().cpu().numpy(), kpts1.squeeze().detach().cpu().numpy()
        mkpts0, mkpts1 = mkpts0.squeeze().detach().cpu().numpy(), mkpts1.squeeze().detach().cpu().numpy()

        rep = compute_repeatability(mkpts0.copy(), mkpts1.copy(), batch['H'].copy(), batch["size"], config["data"]["dist_thresh"])
        
        h_est_acc, h_est_err, inliers = estimate_homography(mkpts0.copy(), mkpts1.copy(), batch['H'].copy(), batch["size"], config["data"]["dist_thresh"])

        mm_acc = mean_matching_acc(mkpts0.copy(), mkpts1.copy(), batch['H'].copy(), config["data"]["dist_thresh"])

        avg_pre_match_points += int(kpts0.shape[0])
        avg_post_match_points += int(mkpts0.shape[0])
        repeatability.append(rep)
        homogaphy_est_acc.append(h_est_acc)
        homography_est_err.append(h_est_err)
        MMA += mm_acc
        num_matches += 1.0

    hpatches_metrics(repeatability, 
                     homogaphy_est_acc, 
                     homography_est_err, 
                     MMA, 
                     num_matches, 
                     avg_pre_match_points,
                     avg_post_match_points,
                     len(data_loader.dataset), 
                     config["data"]["dist_thresh"])

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config_path = "silk/hpatches_eval.yaml"
    
    config = yaml.load(stream=open(config_path, 'r'), Loader=yaml.FullLoader)

    model = get_model(checkpoint=CHECKPOINT_PATH,
                      default_outputs=("sparse_positions", "sparse_descriptors"),
                      top_k=config["matcher"]["top_k"])

    dataset = HPatches(config["data"], device=device)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.batch_collator, pin_memory=True)

    estimate_hpatches_metrics(config, model, dataloader, device)
