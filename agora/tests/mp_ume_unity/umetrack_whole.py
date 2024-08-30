# basic_settings

import os
import sys


# get absolute path of parent of parent of parent of current directory
UMETRACK_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", ".."
))

sys.path.append(UMETRACK_ROOT)


import av
import fnmatch
import pickle
import numpy as np
import torch

import lib.data_utils.fs as fs

from functools import partial


from lib.models.model_loader import load_pretrained_model

from lib.tracker.video_pose_data import SyncedImagePoseStream

from typing import Dict, NamedTuple

NUM_HANDS = 2
NUM_LANDMARKS_PER_HAND = 21
NUM_FINGERTIPS_PER_HAND = 5
NUM_JOINTS_PER_HAND = 22
LEFT_HAND_INDEX = 0
RIGHT_HAND_INDEX = 1

NUM_DIGITS: int = 5
NUM_JOINT_FRAMES: int = 1 + 1 + 3 * 5  # root + wrist + finger frames * 5
DOF_PER_FINGER: int = 4

# ======================================================
# affine.py
import numpy as np
from scipy.spatial.transform import Rotation


def transform3(m, v):
    return transform_vec3(m, v) + m[..., :3, 3]


def transform_vec3(m, v):
    if m.ndim == 2:
        return (v.reshape(-1, 3) @ m[:3, :3].T).reshape(v.shape)
    else:
        return (m[..., :3, :3] @ v[..., None]).squeeze(-1)


def normalized(v: np.ndarray, axis: int = -1, eps: float = 5.43e-20) -> np.ndarray:
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d


def skew_matrix(v: np.ndarray) -> np.ndarray:
    res = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype
    )
    return res


def from_two_vectors(a_orig: np.ndarray, b_orig: np.ndarray) -> np.ndarray:
    a = normalized(a_orig)
    b = normalized(b_orig)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = skew_matrix(v)

    rot = np.eye(3, 3) + v_mat + np.matmul(v_mat, v_mat) * (1 - c) / (max(s * s, 1e-15))

    return rot


def make_look_at_matrix(
    orig_world_to_eye: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
) -> np.ndarray:
    """
    args:
        orig_world_to_eye:  world to eye transform
            inverse of camera.camera_to_world_xf
        center:  np.array. (3,)
            3D world coordinate of center of the object of interest
        camera_angle: the angle of the camera
            camera_angle: how the camera is oriented physically so that we can rotate the object of
            interest to the 'upright' direction
    """
    center_local = transform3(orig_world_to_eye, center)
    
    z_dir_local = center_local / np.linalg.norm(center_local)
    
    delta_r_local = from_two_vectors(
        np.array([0, 0, 1], dtype=center.dtype), z_dir_local
    )
    orig_eye_to_world = np.linalg.inv(orig_world_to_eye)

    new_eye_to_world = orig_eye_to_world.copy()
    new_eye_to_world[0:3, 0:3] = orig_eye_to_world[0:3, 0:3] @ delta_r_local

    # Locally rotate the z axis to align with the camera angle
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    new_eye_to_world[0:3, 0:3] = new_eye_to_world[0:3, 0:3] @ z_local_rot

    return np.linalg.inv(new_eye_to_world)


# camera.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
import math
from typing import NamedTuple, Sequence, Tuple, Type

import numpy as np
from typing_extensions import Protocol, runtime_checkable


class CameraProjection(Protocol):
    """
    Defines a projection from a 3D `xyz` direction or point to 2D.
    """

    @classmethod
    @abc.abstractmethod
    def project(cls, v):
        """
        Project a 3d vector in eye space down to 2d.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def unproject(cls, p):
        """
        Unproject a 2d point to a unit-length vector in eye space.

        `project(unproject(p)) == p`
        `unproject(project(v)) == v / |v|`
        """
        ...


@runtime_checkable
class DistortionModel(Protocol):
    @abc.abstractmethod
    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        """
        Arguments
        ---------
        p: ndarray[..., 2]
            Array of 2D points, of arbitrary batch shape.

        Returns
        -------
        q: ndarray[..., 2]
            Distorted points with same shape as input
        """
        ...


class PerspectiveProjection(CameraProjection):
    @staticmethod
    def project(v):
        # map to [x/z, y/z]
        assert v.shape[-1] == 3
        return v[..., :2] / v[..., 2, None]

    @staticmethod
    def unproject(p):
        # map to [u,v,1] and renormalize
        assert p.shape[-1] == 2
        x, y = np.moveaxis(p, -1, 0)
        v = np.stack((x, y, np.ones(shape=x.shape, dtype=x.dtype)), axis=-1)
        v = normalized(v, axis=-1)
        return v


class ArctanProjection(CameraProjection):
    @staticmethod
    def project(p, eps: float = 2.0**-128):
        assert p.shape[-1] == 3
        x, y, z = np.moveaxis(p, -1, 0)
        r = np.sqrt(x * x + y * y)
        s = np.arctan2(r, z) / np.maximum(r, eps)
        return np.stack((x * s, y * s), axis=-1)

    @staticmethod
    def unproject(uv):
        assert uv.shape[-1] == 2
        u, v = np.moveaxis(uv, -1, 0)
        r = np.sqrt(u * u + v * v)
        c = np.cos(r)
        s = np.sinc(r / np.pi)
        return np.stack([u * s, v * s, c], axis=-1)


class NoDistortion(NamedTuple):
    """
    A trivial distortion model that does not distort the incoming rays.
    """

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        return p


class Fisheye62CameraModel(NamedTuple):
    """
    Fisheye62CameraModel model, with 6 radial and 2 tangential coeffs.
    """

    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float
    k5: float
    k6: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4, p1, p2, k5, k6 = self
        # radial component
        r2 = (p * p).sum(axis=-1, keepdims=True)
        r2 = np.clip(r2, -np.pi**2, np.pi**2)
        r4 = r2 * r2
        r6 = r2 * r4
        r8 = r4 * r4
        r10 = r4 * r6
        r12 = r6 * r6
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
        uv = p * radial

        # tangential component
        x, y = uv[..., 0], uv[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        x += 2 * p2 * xy + p1 * (r2 + 2 * x2)
        y += 2 * p1 * xy + p2 * (r2 + 2 * y2)
        return np.stack((x, y), axis=-1)


# ---------------------------------------------------------------------
# API Conventions and naming
#
# Points have the xyz or uv components in the last axis, and may have
# arbitrary batch shapes. ([...,2] for 2d and [...,3] for 3d).
#
# v
#    3D xyz position in eye space, usually unit-length.
# p
#    projected uv coordinates: `p = project(v)`
# q
#    distorted uv coordinates: `q = distort(p)`
# w
#    window coordinates: `q = q * f + [cx, cy]`
#
# A trailing underscore (e.g. `p_`, `q_`) should be read as "hat", and
# generally indicates an approximation to another value.
# ---------------------------------------------------------------------


class CameraModel(CameraProjection, abc.ABC):
    """
    Parameters
    ----------
    width, height : int
        Size of the sensor window

    f : float or tuple(float, float)
        Focal length

    c : tuple(float, float)
        Optical center in window coordinates

    distort_coeffs
        Forward distortion coefficients (eye -> window).

        If this is an instance of DistortionModel, it will be used as-is
        (even if it's a different polynomial than this camera model
        would normally use.) If it's a simple tuple or array, it will
        used as coefficients for `self.distortion_model`.

    camera_to_world_xf : np.ndarray
        Camera's position and orientation in world space, represented as
        a 3x4 or 4x4 matrix.

        The matrix be a rigid transform (only rotation and translation).

        You can change a camera's camera_to_world_xf after construction by
        assigning to or modifying this matrix.

    Attributes
    ----------
    Most attributes are the same as constructor parameters.

    distortion_model
        Class attribute giving the distortion model for new instances.

    """

    width: int
    height: int

    f: Tuple[float, float]
    c: Tuple[float, float]

    camera_to_world_xf: np.ndarray

    distortion_model: Type[DistortionModel]
    distort: DistortionModel

    def __init__(
        self,
        width,
        height,
        f,
        c,
        distort_coeffs,
        camera_to_world_xf=None,
    ):  # pylint: disable=super-init-not-called (see issue 4790 on pylint github)
        self.width = width
        self.height = height

        # f can be either a scalar or (fx,fy) pair. We only fit scalars,
        # but may load (fx, fy) from a stored file.
        self.f = tuple(np.broadcast_to(f, 2))
        self.c = tuple(c)

        if camera_to_world_xf is None:
            self.camera_to_world_xf = np.eye(4)
        else:
            self.camera_to_world_xf = camera_to_world_xf

        if isinstance(distort_coeffs, DistortionModel):
            self.distort = distort_coeffs
        else:
            self.distort = self.distortion_model(*distort_coeffs)

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.width}x{self.height}, f={self.f} c={self.c}"
        )

    def copy(self, camera_to_world_xf=None):
        """Return a copy of this camera

        Arguments
        ---------
        camera_to_world_xf : 4x4 np.ndarray
            Optional new camera_to_world_xf for the new camera model.
            Default is to copy this camera's camera_to_world_xf.
        """
        return self.crop(0, 0, self.width, self.height, camera_to_world_xf=camera_to_world_xf)

    def world_to_eye(self, v):
        """
        Apply camera camera_to_world_xf to points `v` to get eye coords
        """
        return transform_vec3(self.camera_to_world_xf.T, v - self.camera_to_world_xf[:3, 3])

    def eye_to_world(self, v):
        """
        Apply inverse camera camera_to_world_xf to eye points `v` to get world coords
        """
        return transform3(self.camera_to_world_xf, v)

    def eye_to_window(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        q = self.distort.evaluate(p)
        return q * self.f + self.c

    def window_to_eye(self, w):
        """Unproject 2d window coordinates to unit-length 3D eye coordinates"""
        q = (np.asarray(w) - self.c) / self.f
        # assert isinstance(
        #     self.distort, NoDistortion
        # ), "Only unprojection for NoDistortion camera is supported"
        return self.unproject(q)

    def crop(
        self,
        src_x,
        src_y,
        target_width,
        target_height,
        scale=1,
        camera_to_world_xf=None,
    ):
        """
        Return intrinsics for a crop of the sensor image.

        No scaling is applied; this just returns the model for a sub-
        array of image data. (Or for a larger array, if (x,y)<=0 and
        (width, height) > (self.width, self.height).

        To do both cropping and scaling, use :meth:`subrect`

        Parameters
        ----------
        x, y, width, height
            Location and size in this camera's window coordinates
        """
        return type(self)(
            target_width,
            target_height,
            np.asarray(self.f) * scale,
            (np.array(self.c) - (src_x, src_y) + 0.5) * scale - 0.5,
            self.distort,
            self.camera_to_world_xf if camera_to_world_xf is None else camera_to_world_xf,
        )


# Camera models
# =============


class PinholePlaneCameraModel(PerspectiveProjection, CameraModel):
    distortion_model = NoDistortion

    def uv_to_window_matrix(self):
        """Return the 3x3 intrinsics matrix"""
        return np.array(
            [[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]]
        )


class Fisheye62CameraModel(ArctanProjection, CameraModel):
    distortion_model = Fisheye62CameraModel
    
    @staticmethod
    def unproject(uv):
        """
        Unproject 2D window coordinates to unit-length 3D eye coordinates.
        
        This implementation assumes that `uv` are in normalized image coordinates
        (i.e., after subtracting the principal point and dividing by the focal length).
        """
        assert uv.shape[-1] == 2, "Input should be a 2D point or array of 2D points."

        u, v = np.moveaxis(uv, -1, 0)
        r = np.sqrt(u * u + v * v)

        # Compute the angle of the ray in radians
        theta = np.arctan(r)

        # Compute the normalized z component (cosine of theta)
        z = np.cos(theta)

        # Compute the radial distance on the image plane
        r_normalized = np.sin(theta)

        # Normalize u and v by their radial distance to get the x and y components
        x = u * r_normalized / np.maximum(r, 1e-12)
        y = v * r_normalized / np.maximum(r, 1e-12)

        return np.stack([x, y, z], axis=-1)

    def window_to_eye(self, w):
        """
        Unproject 2D window coordinates to unit-length 3D eye coordinates
        with fisheye distortion.
        """
        q = (np.asarray(w) - self.c) / self.f
        
        return self.unproject(q)

def read_camera_from_json(js):
    if isinstance(js, str):
        js = json.loads(js)
    js = js.get("Camera", js)

    width = js["ImageSizeX"]
    height = js["ImageSizeY"]
    model = js["DistortionModel"]
    fx = js["fx"]
    fy = js["fy"]
    cx = js["cx"]
    cy = js["cy"]

    if model == "PinholePlane":
        cls = PinholePlaneCameraModel
    elif model == "FishEye62":
        cls = Fisheye62CameraModel

    distort_params = cls.distortion_model._fields
    coeffs = [js[name] for name in distort_params]

    return cls(width, height, (fx, fy), (cx, cy), coeffs)



# ======================================================
# crop.py


from typing import Tuple


import numpy as np


def gen_intrinsics_from_bounding_pts(
    pts_eye: np.ndarray, image_w: int, image_h: int, min_focal: float = 5
) -> Tuple[np.ndarray, np.ndarray]:
    pts_ndc = pts_eye[..., 0:2] / pts_eye[..., 2:]
    img_size = np.array([image_w, image_h], dtype=pts_eye.dtype)
    # Given our convention, we need to shift one pixel before dividing by 2.
    cx_cy = (img_size - 1) / 2
    fx_fy = cx_cy / np.absolute(pts_ndc).max()

    # Some sanity checks
    if np.any(pts_eye[..., 2:] < 0.0001) or np.any(fx_fy < min_focal):
        raise ValueError("Unable to create crop camera", fx_fy)

    return fx_fy, cx_cy


def gen_crop_parameters_from_points(
    camera_orig: CameraModel,
    pts_world,
    new_image_size: Tuple[int, int],
    mirror_img_x: bool,
    camera_angle: float = 0,
    focal_multiplier: float = 0.95,
) -> PinholePlaneCameraModel:
    """
    Given the original camera transform and a list of 3D points in the world space,
    compute the new perspective camera that makes sure after projection all the points
    can be projected inside the image.

    Auguments:
    * camera_orig: the original camera used for generating an image. The returned camera
        will have the same position but different rotation and intrinsics parameters.
    * pts_world: points in the world space that must be projected inside the image by
        the generated world to eye transform and intrinsics.
    * new_image_size: target image size
    * mirror_img_x: whether to flip the image. A typical use case is we usually mirror the
        right hand images so that a model need to handle left hand data only
    * camera_angle: how the camera is oriented physically so that we can rotate the object of
        interest to the 'upright' direction
    * focal_multiplier: when less than 1, we are zooming out a little. The effect on the image
        is some margin will be left at the boundary.
    """
    orig_world_to_eye_xf = np.linalg.inv(camera_orig.camera_to_world_xf)

    crop_center = (pts_world.min(axis=0) + pts_world.max(axis=0)) / 2.0
    new_world_to_eye = make_look_at_matrix(
        orig_world_to_eye_xf, crop_center, camera_angle
    )
    if mirror_img_x:
        mirrorx = np.eye(4, dtype=np.float32)
        mirrorx[0, 0] = -1
        new_world_to_eye = mirrorx @ new_world_to_eye

    fx_fy, cx_cy = gen_intrinsics_from_bounding_pts(
        transform3(new_world_to_eye, pts_world),
        new_image_size[0],
        new_image_size[1],
    )
    fx_fy = focal_multiplier * fx_fy

    return PinholePlaneCameraModel(
        width=new_image_size[0],
        height=new_image_size[1],
        f=fx_fy,
        c=cx_cy,
        distort_coeffs=[],
        camera_to_world_xf=np.linalg.inv(new_world_to_eye),
    )
    
    

# ======================================================
# tracking_results.py

from typing import Dict, NamedTuple

class SingleHandPose(NamedTuple):
    """
    A hand pose is composed of two fields:
    1) joint angles where # joints == # DoFs
    2) root-to-world rigid wrist transformation
    """

    joint_angles: np.ndarray = np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32)
    wrist_xform: np.ndarray = np.eye(4, dtype=np.float32)
    hand_confidence: float = 1.0


# Tracking result maps from hand_index to hand_pose
class TrackingResult(NamedTuple):
    hand_poses: Dict[int, SingleHandPose] = {}
    num_views: Dict[int, int] = {}
    predicted_scales: Dict[int, float] = {}



# ======================================================
# perspective_crop.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

# import lib.common.camera as camera
import numpy as np
import torch
#from lib.common.crop import gen_crop_parameters_from_points
from lib.common.hand import HandModel, NUM_JOINTS_PER_HAND, RIGHT_HAND_INDEX
from lib.common.hand_skinning import skin_landmarks

#from .tracking_result import SingleHandPose


def neutral_joint_angles(up: HandModel, lower_factor: float = 0.5) -> torch.Tensor:
    joint_limits = up.joint_limits
    assert joint_limits is not None
    return joint_limits[..., 0] * lower_factor + joint_limits[..., 1] * (
        1 - lower_factor
    )


def skin_landmarks_np(
    hand_model: HandModel,
    joint_angles: np.ndarray,
    wrist_transforms: np.ndarray,
) -> np.ndarray:
    landmarks = skin_landmarks(
        hand_model,
        torch.from_numpy(joint_angles).float(),
        torch.from_numpy(wrist_transforms).float(),
    )
    return landmarks.numpy()


def landmarks_from_hand_pose(
    hand_model: HandModel, hand_pose: SingleHandPose, hand_idx: int
) -> np.ndarray:
    """
    Compute 3D landmarks in the world space given the hand model and hand pose.
    """
    xf = hand_pose.wrist_xform.copy()
    # This function expects the user hand model to be a left hand.
    if hand_idx == RIGHT_HAND_INDEX:
        xf[:, 0] *= -1
    landmarks = skin_landmarks_np(hand_model, hand_pose.joint_angles, xf)
    return landmarks


def rank_hand_visibility_in_cameras(
    cameras: List[CameraModel],
    hand_model: HandModel,
    hand_pose: SingleHandPose,
    hand_idx: int,
    min_required_vis_landmarks: int,
) -> List[int]:
    landmarks_world = landmarks_from_hand_pose(
        hand_model, hand_pose, hand_idx
    )
    
    # list of number of 3D keypoints that project into views for each camera
    n_landmarks_in_view = []
    
    # list of camera indices that can see enough hand points
    ranked_cam_indices = []
    for cam_idx, camera in enumerate(cameras):
        # (21, 3)
        landmarks_eye = camera.world_to_eye(landmarks_world)
        
        # (21, 2)
        landmarks_win2 = camera.eye_to_window(landmarks_eye)
        
        n_visible = (
            (landmarks_win2[..., 0] >= 0)
            & (landmarks_win2[..., 0] <= camera.width - 1)
            & (landmarks_win2[..., 1] >= 0)
            & (landmarks_win2[..., 1] <= camera.height - 1)
            & (landmarks_eye[..., 2] > 0)
        ).sum()

        n_landmarks_in_view.append(n_visible)
        # Only push the cameras that can see enough hand points
        if n_visible >= min_required_vis_landmarks:
            ranked_cam_indices.append(cam_idx)

    # print("ranked_cam_indices", ranked_cam_indices)
    # print("n_landmarks_in_view", n_landmarks_in_view)   

    #  Favor the view that sees more landmarks
    ranked_cam_indices.sort(
        reverse=True,
        key=lambda x: n_landmarks_in_view[x],
    )
    # print("ranked_cam_indices", ranked_cam_indices)
    return ranked_cam_indices


def _get_crop_points_from_hand_pose(
    hand_model: HandModel,
    gt_hand_pose: SingleHandPose,
    hand_idx: int,
    num_crop_points: int,
) -> np.ndarray:
    '''
    args :
        hand_model : HandModel
        
        gt_hand_pose : SingleHandPose
        
        hand_idx : int

        num_crop_points : int
            ㅅㅂ WTF does num_crop_points do?
    
    return :
        np.ndarray of shape (num_crop_points, 3)
        each row is a 3D keypoint in the 3D coordinate
        (not sure world space or eye space)
    '''
    assert num_crop_points in [21, 42, 63]
    neutral_hand_pose = SingleHandPose(
        joint_angles=neutral_joint_angles(hand_model).numpy(),
        wrist_xform=gt_hand_pose.wrist_xform,
    )
    open_hand_pose = SingleHandPose(
        joint_angles=np.zeros(NUM_JOINTS_PER_HAND, dtype=np.float32),
        wrist_xform=gt_hand_pose.wrist_xform,
    )

    crop_points = []
    crop_points.append(landmarks_from_hand_pose(
        hand_model, gt_hand_pose, hand_idx)
    )
    if num_crop_points > 21:
        crop_points.append(
            landmarks_from_hand_pose(
                hand_model, neutral_hand_pose, hand_idx
            )
        )
    if num_crop_points > 42:
        crop_points.append(
            landmarks_from_hand_pose(
                hand_model, open_hand_pose, hand_idx
            )
        )
    
    result = np.concatenate(crop_points, axis=0)
    # print("crop_points", result.shape)
    
    return result

def gen_crop_cameras_from_pose(
    cameras: List[CameraModel],
    camera_angles: List[float],
    hand_model: HandModel,
    hand_pose: SingleHandPose,
    hand_idx: int,
    num_crop_points: int,
    new_image_size: Tuple[int, int],
    max_view_num: Optional[int] = None,
    sort_camera_index: bool = False,
    focal_multiplier: float = 0.95,
    mirror_right_hand: bool = True,
    min_required_vis_landmarks: int = 19,
) -> Dict[int, PinholePlaneCameraModel]:
    
    crop_cameras: Dict[int, PinholePlaneCameraModel] = {}

    # keypoints coordinates in the world space
    crop_points = _get_crop_points_from_hand_pose(
        hand_model,
        hand_pose,
        hand_idx,
        num_crop_points,
    )
    
    # list of camera indices that can see enough hand points
    # sorted by the number of landmarks that can be seen, descending
    cam_indices = rank_hand_visibility_in_cameras(
        cameras=cameras,
        hand_model=hand_model,
        hand_pose=hand_pose,
        hand_idx=hand_idx,
        min_required_vis_landmarks=min_required_vis_landmarks,
    )

    # cam_indices is sorted by the number of landmarks that can be seen, descending
    # if sort_camera_index is True, then sort cam_indices
    if sort_camera_index:
        cam_indices = sorted(cam_indices)


    for cam_idx in cam_indices:
        crop_cameras[cam_idx] = gen_crop_parameters_from_points(
            cameras[cam_idx],
            crop_points,
            new_image_size,
            mirror_img_x=(mirror_right_hand and hand_idx == 1),
            camera_angle=camera_angles[cam_idx],
            focal_multiplier=focal_multiplier,
        )
        if len(crop_cameras) == max_view_num:
            break

    return crop_cameras


# ======================================================
# tracker.py

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2

# import lib.common.camera as camera
import numpy as np
import torch


from lib.common.hand import HandModel, NUM_HANDS, scaled_hand_model

from lib.data_utils import bundles

from lib.models.regressor import RegressorOutput
from lib.models.umetrack_model import InputFrameData, InputFrameDesc, InputSkeletonData


MM_TO_M = 0.001
M_TO_MM = 1000.0
MIN_OBSERVED_LANDMARKS = 21
CONFIDENCE_THRESHOLD = 0.5
MAX_VIEW_NUM = 2


@dataclass
class ViewData:
    image: np.ndarray
    camera: CameraModel
    camera_angle: float


@dataclass
class InputFrame:
    views: List[ViewData]

@dataclass
class HandTrackerOpts:
    
    num_crop_points: int = 63
    # can be [21, 42, 63]
    # when performing perspective crop,
    # 21: crop along detected pose ?
    # 42: crop along detected pose + open pose
    # 63: crop along detected pose + open pose + neutral pose
    
    enable_memory: bool = True
    
    use_stored_pose_for_crop: bool = True
    
    hand_ratio_in_crop: float = 0.95
    
    min_required_vis_landmarks: int = 19

def _warp_image(
    src_camera: CameraModel,
    dst_camera: CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
) -> np.ndarray:
    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))


    # print("dst_camera.dostort", type(dst_camera.distort))
    # print("dst_camera.distort", dst_camera.distort)
    # print(isinstance(dst_camera.distort, NoDistortion))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)

class HandTracker:
    def __init__(self, model, opts: HandTrackerOpts) -> None:
        self._device: str = "cuda" if torch.cuda.device_count() else "cpu"
        
        self._model = model
        self._model.to(self._device)

        self._input_size = np.array(self._model.getInputImageSizes())
        self._num_crop_points = opts.num_crop_points
        self._enable_memory = opts.enable_memory
        self._hand_ratio_in_crop: float = opts.hand_ratio_in_crop
        self._min_required_vis_landmarks: int = opts.min_required_vis_landmarks
        self._valid_tracking_history = np.zeros(2, dtype=bool)

    def reset_history(self) -> None:
        self._valid_tracking_history[:] = False
        
    def gen_crop_cameras_from_stereo_camera_with_window_hand_pose(
        self,
        camera_left: CameraModel,
        camera_right: CameraModel,
        window_hand_pose_left: Dict[int, np.ndarray],
        window_hand_pose_right: Dict[int, np.ndarray],
    ) :
        """
        window coordinate : pixel coordinate
        get crop_cameras from camera, view and 2D pose
        
        Assumstions :
            camera is stereo camera. 
            2D Hand pose is already valid.
        args :
            camera_left  : camera.CameraModel
            camera_right : camera.CameraModel
            window_hand_pose_left  : Dict[int, np.ndarray]
            window_hand_pose_right : Dict[int, np.ndarray]
        """
        crop_camera_dict: Dict[int, Dict[int, PinholePlaneCameraModel]] = {}
        
        # perform for left camera
        camera_left_world_to_eye = np.linalg.inv(camera_left.camera_to_world_xf)
        for hand_idx, window_hand_pose in window_hand_pose_left.items():
            # hand_idx      left : 0  right : 1
            # window_hand_pose : np.array, (21, 2). window coordinate
            crop_camera_dict_per_hand: Dict[int, PinholePlaneCameraModel] = {}
            
            world_hand_pose = camera_left.eye_to_world(
                camera_left.window_to_eye(window_hand_pose[:, :2])
            )   
            
            world_hand_pose_center = (
                world_hand_pose.min(axis=0) + world_hand_pose.max(axis=0)
            ) / 2
            
            new_world_to_eye = make_look_at_matrix(
                camera_left_world_to_eye,
                world_hand_pose_center,
                0
            )
            if hand_idx == 1:
                mirrorx = np.eye(4, dtype=np.float32)
                mirrorx[0, 0] = -1
                new_world_to_eye = mirrorx @ new_world_to_eye
            
            fx_fy, cx_cy = gen_intrinsics_from_bounding_pts(
                transform3(new_world_to_eye, world_hand_pose),
                self._input_size[0], self._input_size[1],
            )
            fx_fy = self._hand_ratio_in_crop * fx_fy
            
            new_cam = PinholePlaneCameraModel(
                width=self._input_size[0],
                height=self._input_size[1],
                f=fx_fy,
                c=cx_cy,
                distort_coeffs = [],
                camera_to_world_xf=np.linalg.inv(new_world_to_eye),
            )
            crop_camera_dict_per_hand[hand_idx] = new_cam
            crop_camera_dict[hand_idx] = crop_camera_dict_per_hand
            
        camera_right_world_to_eye = np.linalg.inv(camera_right.camera_to_world_xf)
        for hand_idx, window_hand_pose in window_hand_pose_right.items():
            crop_camera_dict_per_hand = {}
            
            world_hand_pose = camera_right.eye_to_world(
                camera_right.window_to_eye(window_hand_pose[:, :2])
            )   
            
            world_hand_pose_center = (
                world_hand_pose.min(axis=0) + world_hand_pose.max(axis=0)
            ) / 2
            
            new_world_to_eye = make_look_at_matrix(
                camera_right_world_to_eye,
                world_hand_pose_center,
                0
            )
            if hand_idx == 1:
                mirrorx = np.eye(4, dtype=np.float32)
                mirrorx[0, 0] = -1
                new_world_to_eye = mirrorx @ new_world_to_eye
            
            fx_fy, cx_cy = gen_intrinsics_from_bounding_pts(
                transform3(new_world_to_eye, world_hand_pose),
                self._input_size[0], self._input_size[1],
            )
            fx_fy = self._hand_ratio_in_crop * fx_fy
            
            new_cam = PinholePlaneCameraModel(
                width=self._input_size[0],
                height=self._input_size[1],
                f=fx_fy,
                c=cx_cy,
                distort_coeffs = [],
                camera_to_world_xf=np.linalg.inv(new_world_to_eye),
            )
            crop_camera_dict_per_hand[hand_idx] = new_cam
            if hand_idx in crop_camera_dict :
                crop_camera_dict[hand_idx].update(crop_camera_dict_per_hand)
            else :
                crop_camera_dict[hand_idx] = crop_camera_dict_per_hand
            
        return crop_camera_dict
            
            # fig = go.Figure()
            # fig.add_trace(go.Scatter3d(
            #     x=world_hand_pose[:, 0], y=world_hand_pose[:, 1], z=world_hand_pose[:, 2],
            #     mode='markers',
            #     marker=dict(size=5, color='blue'),
            #     name='World Hand Pose'
            # ))
            # camera_left_pos = camera_left.camera_to_world_xf[:3, 3]
            # fig.add_trace(go.Scatter3d(
            #     x=[camera_left_pos[0]], y=[camera_left_pos[1]], z=[camera_left_pos[2]],
            #     mode='markers',
            #     marker=dict(size=10, color='red', symbol='square'),
            #     name='Camera Left'
            # ))
            # camera_right_pos = camera_right.camera_to_world_xf[:3, 3]
            # fig.add_trace(go.Scatter3d(
            #     x=[camera_right_pos[0]], y=[camera_right_pos[1]], z=[camera_right_pos[2]],
            #     mode='markers',
            #     marker=dict(size=10, color='green', symbol='square'),
            #     name='Camera Right'
            # ))
            # fig.update_layout(
            #     scene=dict(
            #         xaxis_title='X',
            #         yaxis_title='Y',
            #         zaxis_title='Z',
            #         aspectmode='data'
            #     ),
            #     title='3D Visualization of World Hand Pose and Cameras'
            # )
            # fig.show()
            
            
        
    def gen_crop_cameras_analysis(
        self,
        #cameras: List[camera.CameraModel],
        input_frame,
        camera_angles: List[float],
        hand_model: HandModel,
        gt_tracking: Dict[int, SingleHandPose],
        min_num_crops: int,
    ) -> Dict[int, Dict[int, PinholePlaneCameraModel]]:
        """
        functions to analyze the crop cameras
        """
    
        cameras = [view.camera for view in input_frame.views]
        
        crop_cameras: Dict[int, Dict[int, PinholePlaneCameraModel]] = {}
        if not gt_tracking:
            return crop_cameras

        # evaluate for each detected hands.
        for hand_idx, gt_hand_pose in gt_tracking.items():
            if gt_hand_pose.hand_confidence < CONFIDENCE_THRESHOLD:
                continue

            # contains croped cameras in multi origin views for single hand
            crop_camera_dict_per_hand: Dict[int, PinholePlaneCameraModel] = {}
            
            # contains camera indexes that target hand is visible in       
            valid_cam_index_list = sorted(rank_hand_visibility_in_cameras(
                cameras,
                hand_model,
                gt_hand_pose,
                hand_idx,
                self._min_required_vis_landmarks,
            ))

            world_hand_pose = _get_crop_points_from_hand_pose(
                hand_model,
                gt_hand_pose,
                hand_idx,
                self._num_crop_points,
                #21
            )
            
            for valid_cam_idx in valid_cam_index_list:
                orig_cam = cameras[valid_cam_idx]
                orig_cam_angle = camera_angles[valid_cam_idx]
                
                orig_world_to_eye_xf = np.linalg.inv(orig_cam.camera_to_world_xf)
                
                world_hand_pose_center = (
                    world_hand_pose.min(axis=0) + world_hand_pose.max(axis=0)
                ) / 2
                
                window_hand_pose = orig_cam.eye_to_window(
                    orig_cam.world_to_eye(world_hand_pose)
                ).astype(int)
                
                
                # plt.imshow(input_frame.views[valid_cam_idx].image, cmap="gray")
                # plt.scatter(window_hand_pose[:, 0], window_hand_pose[:, 1])
                # plt.show()
                
                new_world_to_eye = make_look_at_matrix(
                    orig_world_to_eye_xf,
                    world_hand_pose_center,
                    orig_cam_angle
                )
                
                if hand_idx == 1:
                    mirrorx = np.eye(4, dtype=np.float32)
                    mirrorx[0, 0] = -1
                    new_world_to_eye = mirrorx @ new_world_to_eye
                
                fx_fy, cx_cy = gen_intrinsics_from_bounding_pts(
                    transform3(new_world_to_eye, world_hand_pose),
                    self._input_size[0], self._input_size[1],
                )
                
                fx_fy = self._hand_ratio_in_crop * fx_fy
                
                new_cam = PinholePlaneCameraModel(
                    width=self._input_size[0],
                    height=self._input_size[1],
                    f=fx_fy,
                    c=cx_cy,
                    distort_coeffs = [],
                    camera_to_world_xf=np.linalg.inv(new_world_to_eye),
                )
                crop_camera_dict_per_hand[valid_cam_idx] = new_cam

            crop_cameras[hand_idx] = crop_camera_dict_per_hand

        # Remove empty crop_cameras
        del_list = []
        for hand_idx, per_hand_crop_cameras in crop_cameras.items():
            if not per_hand_crop_cameras or len(per_hand_crop_cameras) < min_num_crops:
                del_list.append(hand_idx)
        for hand_idx in del_list:
            del crop_cameras[hand_idx]

        return crop_cameras


    def gen_crop_cameras(
        self,
        cameras: List[CameraModel],
        camera_angles: List[float],
        hand_model: HandModel,
        gt_tracking: Dict[int, SingleHandPose],
        min_num_crops: int,
    ) -> Dict[int, Dict[int, PinholePlaneCameraModel]]:
    
        '''
        args :
        return :
            crop_cameras: Dict[int, Dict[int, camera.PinholePlaneCameraModel]]
                key : hand_idx
                value : Dict[int, camera.PinholePlaneCameraModel]
                    key : cam_idx
                    value : camera.PinholePlaneCameraModel
        '''
        crop_cameras: Dict[int, Dict[int, PinholePlaneCameraModel]] = {}
        if not gt_tracking:
            return crop_cameras

        for hand_idx, gt_hand_pose in gt_tracking.items():
            if gt_hand_pose.hand_confidence < CONFIDENCE_THRESHOLD:
                continue
            crop_cameras[hand_idx] = gen_crop_cameras_from_pose(
                cameras,
                camera_angles,
                hand_model,
                gt_hand_pose,
                hand_idx,
                self._num_crop_points,
                self._input_size,
                max_view_num=MAX_VIEW_NUM,
                sort_camera_index=True,
                focal_multiplier=self._hand_ratio_in_crop,
                mirror_right_hand=True,
                min_required_vis_landmarks=self._min_required_vis_landmarks,
            )

        # Remove empty crop_cameras
        del_list = []
        for hand_idx, per_hand_crop_cameras in crop_cameras.items():
            if not per_hand_crop_cameras or len(per_hand_crop_cameras) < min_num_crops:
                del_list.append(hand_idx)
        for hand_idx in del_list:
            del crop_cameras[hand_idx]

        return crop_cameras

    def track_frame(
        self,
        sample: InputFrame,
        hand_model: HandModel,
        crop_cameras: Dict[int, Dict[int, PinholePlaneCameraModel]],
    ) -> TrackingResult:
        if not crop_cameras:
            # Frame without hands
            self.reset_history()
            return TrackingResult()

        frame_data, frame_desc, skeleton_data = self._make_inputs(
            sample, hand_model, crop_cameras
        )
        with torch.no_grad():
            regressor_output = bundles.to_device(
                self._model.regress_pose_use_skeleton(
                    frame_data, frame_desc, skeleton_data
                ),
                torch.device("cpu"),
            )

        tracking_result = self._gen_tracking_result(
            regressor_output,
            frame_desc.hand_idx.cpu().numpy(),
            crop_cameras,
        )
        return tracking_result

    def track_frame_and_calibrate_scale(
        self,
        sample: InputFrame,
        crop_cameras: Dict[int, Dict[int, PinholePlaneCameraModel]],
    ) -> TrackingResult:
        if not crop_cameras:
            # Frame without hands
            self.reset_history()
            return TrackingResult()
        frame_data, frame_desc, _ = self._make_inputs(sample, None, crop_cameras)

        with torch.no_grad():
            regressor_output = bundles.to_device(
                self._model.regress_pose_pred_skel_scale(frame_data, frame_desc),
                torch.device("cpu"),
            )

        tracking_result = self._gen_tracking_result(
            regressor_output,
            frame_desc.hand_idx.cpu().numpy(),
            crop_cameras,
        )
        return tracking_result

    def _make_inputs(
        self,
        sample: InputFrame,
        hand_model_mm: Optional[HandModel],
        crop_cameras: Dict[int, Dict[int, PinholePlaneCameraModel]],
    ):
        image_idx = 0
        left_images = []
        intrinsics = []
        extrinsics_xf = []
        sample_range_n_hands = []
        hand_indices = []
        for hand_idx, crop_camera_info in crop_cameras.items():
            sample_range_start = image_idx
            for cam_idx, crop_camera in crop_camera_info.items():
                view_data = sample.views[cam_idx]
                crop_image = _warp_image(view_data.camera, crop_camera, view_data.image)
                left_images.append(crop_image.astype(np.float32) / 255.0)
                intrinsics.append(crop_camera.uv_to_window_matrix())

                crop_world_to_eye_xf = np.linalg.inv(crop_camera.camera_to_world_xf)
                crop_world_to_eye_xf[:3, 3] *= MM_TO_M
                extrinsics_xf.append(crop_world_to_eye_xf)

                image_idx += 1

            if image_idx > sample_range_start:
                hand_indices.append(hand_idx)
                sample_range_n_hands.append(np.array([sample_range_start, image_idx]))

        hand_indices = np.array(hand_indices)
        frame_data = InputFrameData(
            left_images=torch.from_numpy(np.stack(left_images)).float(),
            intrinsics=torch.from_numpy(np.stack(intrinsics)).float(),
            extrinsics_xf=torch.from_numpy(np.stack(extrinsics_xf)).float(),
        )
        frame_desc = InputFrameDesc(
            sample_range=torch.from_numpy(np.stack(sample_range_n_hands)).long(),
            memory_idx=torch.from_numpy(hand_indices).long(),
            # use memory if the hand is previously valid
            use_memory=torch.from_numpy(
                self._valid_tracking_history[hand_indices]
            ).bool(),
            hand_idx=torch.from_numpy(hand_indices).long(),
        )
        skeleton_data = None
        if hand_model_mm is not None:
            # m -> mm
            hand_model_m = scaled_hand_model(hand_model_mm, MM_TO_M)
            skeleton_data = InputSkeletonData(
                joint_rotation_axes=hand_model_m.joint_rotation_axes.float(),
                joint_rest_positions=hand_model_m.joint_rest_positions.float(),
            )
        return bundles.to_device((frame_data, frame_desc, skeleton_data), self._device)

    def _gen_tracking_result(
        self,
        regressor_output: RegressorOutput,
        hand_indices: np.ndarray,
        crop_cameras: Dict[int, Dict[int, PinholePlaneCameraModel]],
    ) -> TrackingResult:

        output_joint_angles = regressor_output.joint_angles.to("cpu").numpy()
        output_wrist_xforms = regressor_output.wrist_xfs.to("cpu").numpy()
        output_wrist_xforms[..., :3, 3] *= M_TO_MM
        output_scales = None
        if regressor_output.skel_scales is not None:
            output_scales = regressor_output.skel_scales.to("cpu").numpy()

        hand_poses = {}
        num_views = {}
        predicted_scales = {}

        for output_idx, hand_idx in enumerate(hand_indices):
            raw_handpose = SingleHandPose(
                joint_angles=output_joint_angles[output_idx],
                wrist_xform=output_wrist_xforms[output_idx],
                hand_confidence=1.0,
            )
            hand_poses[hand_idx] = raw_handpose
            num_views[hand_idx] = len(crop_cameras[hand_idx])
            if output_scales is not None:
                predicted_scales[hand_idx] = output_scales[output_idx]

        for hand_idx in range(NUM_HANDS):
            hand_valid = False
            if hand_idx in hand_poses:
                self._valid_tracking_history[hand_idx] = True
                hand_valid = True
            if hand_valid:
                continue
            self._valid_tracking_history[hand_idx] = False

        return TrackingResult(
            hand_poses=hand_poses,
            num_views=num_views,
            predicted_scales=predicted_scales,
        )




# ======================================================
if __name__ == "__main__" :
        
    import logging

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(__name__)

    # model_name = "pretrained_weights.torch"
    # model_path = os.path.join(".", "pretrained_models", model_name)
    # model = load_pretrained_model(model_path)
    # model.eval()
    # tracker = HandTracker(model, HandTrackerOpts())

    DATA_PATH = os.path.join(UMETRACK_ROOT, "sample_data/recording_00.mp4")
    image_pose_stream = SyncedImagePoseStream(DATA_PATH)
    hand_model = image_pose_stream._hand_pose_labels.hand_model



    # ======================================================





    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    left_to_right_r = np.array([
        9.9997658245714527e-01, 5.5910744958795095e-04, 6.8206990981942916e-03,
        -5.4903304536865717e-04, 9.9999875583076248e-01, -1.4788169738349651e-03,
        -6.8215174296769373e-03, 1.4750375543776898e-03, 9.9997564528550886e-01
    ]).reshape(3, 3)

    left_to_right_t = np.array([
        -5.9457914254177978e-02, -6.8318101539255457e-05, -1.8101725187729225e-04
    ])

    # k1, k2, k3, k4, p1, p2, k5, k6
    distortion_coeffs_left = (
        -3.7539305827469560e-02, 
        -8.7553205432575471e-03,
        2.2015408171895236e-03, 
        -6.6218076061138698e-04,
        0, 0, 0, 0
    )
    camera_to_world_xf_left = np.eye(4)
    rotation_left = np.array([
        [9.9997658245714527e-01,  5.5910744958795095e-04,  6.8206990981942916e-03,],
        [-5.4903304536865717e-04, 9.9999875583076248e-01, -1.4788169738349651e-03,],
        [-6.8215174296769373e-03, 1.4750375543776898e-03,  9.9997564528550886e-01 ],
    ]).reshape(3, 3)
    camera_to_world_xf_left[:3, :3] = rotation_left
    #camera_to_world_xf_left[:3, 3] = [
    cam_left = Fisheye62CameraModel(
        width   = IMG_WIDTH,
        height  = IMG_HEIGHT,
        f       = (2.3877057700850656e+02, 2.3903223316525276e+02),
        c       = (3.1846939219741773e+02, 2.4685137381795201e+02),
        distort_coeffs = distortion_coeffs_left,
        camera_to_world_xf = np.eye(4)
    )


    distortion_coeffs_right = (
        -3.6790400486095221e-02, 
        -8.2041573433038941e-03,
        1.0552974220937024e-03, 
        -2.5841665172692902e-04,
        0, 0, 0, 0
    )
    camera_to_world_xf_right = np.eye(4)
    rotation_right = np.array([
        [9.9999470555416226e-01, 1.1490100298631428e-03, 3.0444440536135159e-03,],
        [-1.1535052313709361e-03, 9.9999824663038117e-01, 1.4751819698614872e-03,],
        [-3.0427437166985561e-03, -1.4786859417328980e-03, 9.9999427758290704e-01 ],
    ]).reshape(3, 3)
    camera_to_world_xf_right[:3, :3] = rotation_right
    camera_to_world_xf_right[:3, 3] = left_to_right_t
    #camera_to_world_xf_right[:3, 3] = [
    cam_right = Fisheye62CameraModel(
        width   = IMG_WIDTH,
        height  = IMG_HEIGHT,
        f       = (2.3952183485043457e+02, 2.3981379751051574e+02),
        c       = (3.1286224145189811e+02, 2.5158397962108106e+02),
        distort_coeffs = distortion_coeffs_right,
        camera_to_world_xf = camera_to_world_xf_right
    )

    def open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX = 10) :
        for CAM_ID in range(-1, CAM_ID_MAX) :
            cap = cv2.VideoCapture(CAM_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH * 2)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
            if cap.isOpened() :
                print(f"Camera ID {CAM_ID} Frame Width {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
                return cap

    CAM_ID_MAX = 10






    import mediapipe as mp



    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_left_cam = mp_hands.Hands(
        max_num_hands = 2,
        model_complexity = 1,
        min_detection_confidence = 0.3, 
        min_tracking_confidence = 0.3
    )

    hands_right_cam = mp_hands.Hands(
        max_num_hands = 2,   
        model_complexity = 1,
        min_detection_confidence = 0.3, 
        min_tracking_confidence = 0.3
    )





    # Joint Altogether

    import socket

    cap = open_stereo_camera(IMG_WIDTH, IMG_HEIGHT, CAM_ID_MAX)

    model_name = "pretrained_weights.torch"
    model_path = os.path.join(UMETRACK_ROOT, "pretrained_models", model_name)
    model = load_pretrained_model(model_path)
    model.eval()
    tracker = HandTracker(model, HandTrackerOpts())

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5052)

    idx = 0

    while True :
        idx += 1
        
        ret, frame_stereo = cap.read()
        frame_stereo = cv2.cvtColor(frame_stereo, cv2.COLOR_BGR2RGB)
        frame_left = frame_stereo[:, :IMG_WIDTH]
        frame_right = frame_stereo[:, IMG_WIDTH:]
        frame_right_gray = cv2.cvtColor(frame_right, cv2.COLOR_RGB2GRAY)
        frame_left_gray = cv2.cvtColor(frame_left, cv2.COLOR_RGB2GRAY)

        hand_pose_result_left_cam = hands_left_cam.process(frame_left)
        hand_pose_window_left_cam = {}
        if hand_pose_result_left_cam.multi_handedness :
            hand_pose_window_left_cam : Dict[int, np.ndarray] = dict(zip(
                list(map(
                    lambda x : x.classification[0].index,
                    hand_pose_result_left_cam.multi_handedness
                )),
                list(map(
                    lambda landamrk_per_hand : np.array(list(map(
                        lambda l : [l.x, l.y, l.z],
                        landamrk_per_hand.landmark
                    ))) * np.array([frame_left.shape[1], frame_left.shape[0], 1]),
                    hand_pose_result_left_cam.multi_hand_landmarks
                ))
            ))

        hand_pose_result_right_cam = hands_right_cam.process(frame_right)
        hand_pose_window_right_cam = {}
        if hand_pose_result_right_cam.multi_handedness :
            hand_pose_window_right_cam : Dict[int, np.ndarray] = dict(zip(
                list(map(
                    lambda x : x.classification[0].index,
                    hand_pose_result_right_cam.multi_handedness
                )),
                list(map(
                    lambda landamrk_per_hand : np.array(list(map(
                        lambda l : [l.x, l.y, l.z],
                        landamrk_per_hand.landmark
                    ))) * np.array([frame_right.shape[1], frame_right.shape[0], 1]),
                    hand_pose_result_right_cam.multi_hand_landmarks
                ))
            ))
            hand_pose_window_right_cam

        fisheye_stereo_input_frame = InputFrame(
            views = [
                ViewData(
                    image = frame_left_gray,
                    camera = cam_left,
                    camera_angle = 0,
                ),
                ViewData(
                    image = frame_right_gray,
                    camera = cam_right,
                    camera_angle = 0,
                )
            ]
        )

        crop_camera_dict = tracker.gen_crop_cameras_from_stereo_camera_with_window_hand_pose(
            camera_left = cam_left,
            camera_right = cam_right,
            window_hand_pose_left = hand_pose_window_left_cam,
            window_hand_pose_right = hand_pose_window_right_cam
        )

        res = tracker.track_frame(fisheye_stereo_input_frame, hand_model, crop_camera_dict)

        tracked_keypoints_dict = {}
        for hand_idx in res.hand_poses.keys() :
            tracked_keypoints = landmarks_from_hand_pose(
                hand_model, res.hand_poses[hand_idx], hand_idx
            )
            tracked_keypoints_dict[hand_idx] = tracked_keypoints

        # sock.sendto(str.encode(str(data)), serverAddressPort)
        
        # for hand_idx, keypoints in tracked_keypoints_dict.items() :
        #     print(hand_idx, keypoints.shape)
        #     print(keypoints.tolist())
        if 0 in tracked_keypoints_dict :
            refined_keypoints = tracked_keypoints_dict[0] / 10 + 100
            refined_keypoints[:, 1] = 200 - refined_keypoints[:, 1]
            
            content = str(refined_keypoints.reshape(-1).tolist())
            print(content)
            sock.sendto(str.encode(content), serverAddressPort)
        
    cap.release()
   