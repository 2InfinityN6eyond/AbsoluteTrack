# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import av
import fnmatch
import pickle
import numpy as np
import torch
import lib.data_utils.fs as fs
from functools import partial
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND
from lib.tracker.tracking_result import SingleHandPose
from multiprocessing import Pool
from typing import Optional, Tuple

from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.tracker.video_pose_data import SyncedImagePoseStream

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def _find_input_output_files(input_dir: str, output_dir: str, test_only: bool):
    res_input_paths = []
    res_output_paths = []
    for cur_dir, _, filenames in fs.walk(input_dir):
        if test_only and not "testing" in cur_dir:
            continue
        mp4_files = fnmatch.filter(filenames, "*.mp4")
        input_full_paths = [fs.join(cur_dir, fname) for fname in mp4_files]
        rel_paths = [f[len(input_dir):] for f in input_full_paths]
        output_full_paths = [fs.join(output_dir, f[:-4] + ".npy") for f in rel_paths]
        res_input_paths += input_full_paths
        res_output_paths += output_full_paths
    assert len(res_input_paths) == len(res_output_paths)
    logger.info(f"Found {len(res_input_paths)} files from {input_dir}")
    return res_input_paths, res_output_paths


def _track_sequence(
    input_output: Tuple[str, str],
    model_path: str,
    override: bool = False,
) -> Optional[np.ndarray]:
    data_path, output_path = input_output
    if not override and fs.exists(output_path):
        logger.info(f"Skipping '{data_path}' since output path '{output_path}' already exists")
        return None

    logger.info(f"Processing {data_path}...")
    model = load_pretrained_model(model_path)
    model.eval()

    image_pose_stream = SyncedImagePoseStream(data_path)

    gt_keypoints = np.zeros([NUM_HANDS, len(image_pose_stream), NUM_LANDMARKS_PER_HAND, 3])
    tracked_keypoints = np.zeros([NUM_HANDS, len(image_pose_stream), NUM_LANDMARKS_PER_HAND, 3])
    valid_tracking = np.zeros([NUM_HANDS, len(image_pose_stream)], dtype=bool)
    tracker = HandTracker(model, HandTrackerOpts())
    for frame_idx, (input_frame, gt_tracking) in enumerate(image_pose_stream):
        hand_model = image_pose_stream._hand_pose_labels.hand_model
        crop_cameras = tracker.gen_crop_cameras(
            [view.camera for view in input_frame.views],
            image_pose_stream._hand_pose_labels.camera_angles,
            hand_model,
            gt_tracking,
            min_num_crops=1,
        )
        
        print(input_frame.views[0].image.shape)
        
        
        res = tracker.track_frame(input_frame, hand_model, crop_cameras)

        for hand_idx in res.hand_poses.keys():
            tracked_keypoints[hand_idx, frame_idx] = landmarks_from_hand_pose(
                hand_model, res.hand_poses[hand_idx], hand_idx
            )
            gt_keypoints[hand_idx, frame_idx] = landmarks_from_hand_pose(
                hand_model, gt_tracking[hand_idx], hand_idx
            )
            valid_tracking[hand_idx, frame_idx] = True

    diff_keypoints = (gt_keypoints - tracked_keypoints)[valid_tracking]
    per_frame_mean_error = np.linalg.norm(diff_keypoints, axis=-1).mean(axis=-1)
    if not fs.exists(fs.dirname(output_path)):
        os.makedirs(fs.dirname(output_path))
    with fs.open(output_path, "wb") as fp:
        pickle.dump(
            {
                "tracked_keypoints": tracked_keypoints,
                "gt_keypoints": gt_keypoints,
                "valid_tracking": valid_tracking,
            },
            fp,
        )
    logger.info(f"Results saved at {output_path}")
    return per_frame_mean_error

if __name__ == '__main__':
    root = os.path.dirname(__file__)
    model_name = "pretrained_weights.torch"
    model_path = os.path.join(root, "pretrained_models", model_name)

    error_tensors = []
    input_dir = os.path.join(root, "UmeTrack_data", "raw_data", "real")
    output_dir = os.path.join(root, "tmp", "eval_results_known_skeleton", "real")
    input_paths, output_paths = _find_input_output_files(input_dir, output_dir, test_only=True)
    pool_size = 8
    with Pool(pool_size) as p:
        error_tensors = p.map_async(partial(_track_sequence, model_path=model_path), zip(input_paths, output_paths)).get()

    error_tensors = [t for t in error_tensors if t is not None]
    if len(error_tensors) != 0:
        logger.info(f"Final mean error: {np.concatenate(error_tensors).mean()}")
