# ─────────────────────────────────────────────────────────────────
# video_mocap/__init__.py
# Native Blender Video Motion Capture Add-on — v2.5
# ─────────────────────────────────────────────────────────────────

bl_info = {
    "name": "Video Motion Capture",
    "author": "Your Name",
    "version": (2, 5, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > VMoCap",
    "description": "Video-based motion capture with automatic and guided tracking",
    "category": "Animation",
}

import bpy
import numpy as np
import math
import os
import json
import sys
import site
import subprocess
import urllib.request
import time
import re
from pathlib import Path
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy.props import (
    StringProperty, EnumProperty, FloatProperty,
    IntProperty, BoolProperty, PointerProperty,
    CollectionProperty, FloatVectorProperty
)
from bpy.types import (
    Operator, Panel, PropertyGroup, AddonPreferences
)

# ═════════════════════════════════════════════════════════════════
# SECTION 1: DEPENDENCIES & BACKEND
# ═════════════════════════════════════════════════════════════════

_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.append(_user_site)

_win_user_packages = os.path.join(
    os.environ.get("APPDATA", ""),
    "Python",
    f"Python{sys.version_info.major}{sys.version_info.minor}",
    "site-packages"
)
if os.path.isdir(_win_user_packages) and _win_user_packages not in sys.path:
    sys.path.append(_win_user_packages)

_HAS_CV2 = False
_HAS_MEDIAPIPE = False
_MP_API_VERSION = None

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    pass

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        _MP_API_VERSION = 'tasks'
    except (ImportError, AttributeError):
        try:
            _ = mp.solutions
            _MP_API_VERSION = 'solutions'
        except AttributeError:
            _MP_API_VERSION = None
except ImportError:
    pass


def ensure_dependencies():
    """Returns list of missing package names (empty if all installed)."""
    missing = []
    if not _HAS_CV2:
        missing.append("opencv-contrib-python")
    if not _HAS_MEDIAPIPE:
        missing.append("mediapipe")
    elif _MP_API_VERSION is None:
        missing.append("mediapipe (reinstall — broken)")
    return missing


def get_addon_directory():
    addon_dir = Path(bpy.utils.user_resource('SCRIPTS')) / "addons" / "video_mocap"
    addon_dir.mkdir(parents=True, exist_ok=True)
    return addon_dir


def get_models_directory():
    models_dir = get_addon_directory() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


MODEL_URLS = {
    "pose_landmarker": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def download_model(model_name, progress_callback=None):
    models_dir = get_models_directory()
    model_path = models_dir / f"{model_name}.task"
    if model_path.exists():
        return str(model_path)
    url = MODEL_URLS.get(model_name)
    if not url:
        raise RuntimeError(f"Unknown model: {model_name}")
    if progress_callback:
        progress_callback(f"Downloading {model_name}...")
    try:
        urllib.request.urlretrieve(url, str(model_path))
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download {model_name}: {e}")
    return str(model_path)


# ═════════════════════════════════════════════════════════════════
# SECTION 2: LANDMARK DEFINITIONS
# ═════════════════════════════════════════════════════════════════

POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

BODY_LANDMARK_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "nose", "spine_direction",
]

LANDMARK_DIRECTION_PAIRS = {
    "left_shoulder": (11, 13),
    "right_shoulder": (12, 14),
    "left_elbow": (13, 15),
    "right_elbow": (14, 16),
    "left_wrist": (15, 19),
    "right_wrist": (16, 20),
    "left_hip": (23, 25),
    "right_hip": (24, 26),
    "left_knee": (25, 27),
    "right_knee": (26, 28),
    "left_ankle": (27, 31),
    "right_ankle": (28, 32),
    "nose": (0, 11),
    "spine_direction": (23, 11),
}

BONE_MAP_PRESETS = {
    'RIGIFY': {
        "spine_direction": "spine",
        "nose": "spine.006",
        "left_shoulder": "upper_arm.L",
        "right_shoulder": "upper_arm.R",
        "left_elbow": "forearm.L",
        "right_elbow": "forearm.R",
        "left_wrist": "hand.L",
        "right_wrist": "hand.R",
        "left_hip": "thigh.L",
        "right_hip": "thigh.R",
        "left_knee": "shin.L",
        "right_knee": "shin.R",
        "left_ankle": "foot.L",
        "right_ankle": "foot.R",
    },
    'MIXAMO': {
        "nose": "mixamorig:Head",
        "left_shoulder": "mixamorig:LeftArm",
        "right_shoulder": "mixamorig:RightArm",
        "left_elbow": "mixamorig:LeftForeArm",
        "right_elbow": "mixamorig:RightForeArm",
        "left_wrist": "mixamorig:LeftHand",
        "right_wrist": "mixamorig:RightHand",
        "left_hip": "mixamorig:LeftUpLeg",
        "right_hip": "mixamorig:RightUpLeg",
        "left_knee": "mixamorig:LeftLeg",
        "right_knee": "mixamorig:RightLeg",
        "left_ankle": "mixamorig:LeftFoot",
        "right_ankle": "mixamorig:RightFoot",
    },
    'CC3': {
        "spine_direction": "CC_Base_Spine01",
        "nose": "CC_Base_Head",
        "left_shoulder": "CC_Base_L_Upperarm",
        "right_shoulder": "CC_Base_R_Upperarm",
        "left_elbow": "CC_Base_L_Forearm",
        "right_elbow": "CC_Base_R_Forearm",
        "left_wrist": "CC_Base_L_Hand",
        "right_wrist": "CC_Base_R_Hand",
        "left_hip": "CC_Base_L_Thigh",
        "right_hip": "CC_Base_R_Thigh",
        "left_knee": "CC_Base_L_Calf",
        "right_knee": "CC_Base_R_Calf",
        "left_ankle": "CC_Base_L_Foot",
        "right_ankle": "CC_Base_R_Foot",
    },
    'CC4': {
        "spine_direction": "CC_Base_Spine01",
        "nose": "CC_Base_Head",
        "left_shoulder": "CC_Base_L_Upperarm",
        "right_shoulder": "CC_Base_R_Upperarm",
        "left_elbow": "CC_Base_L_Forearm",
        "right_elbow": "CC_Base_R_Forearm",
        "left_wrist": "CC_Base_L_Hand",
        "right_wrist": "CC_Base_R_Hand",
        "left_hip": "CC_Base_L_Thigh",
        "right_hip": "CC_Base_R_Thigh",
        "left_knee": "CC_Base_L_Calf",
        "right_knee": "CC_Base_R_Calf",
        "left_ankle": "CC_Base_L_Foot",
        "right_ankle": "CC_Base_R_Foot",
    },
    'UE_MANNEQUIN': {
        "nose": "head",
        "left_shoulder": "upperarm_l",
        "right_shoulder": "upperarm_r",
        "left_elbow": "lowerarm_l",
        "right_elbow": "lowerarm_r",
        "left_wrist": "hand_l",
        "right_wrist": "hand_r",
        "left_hip": "thigh_l",
        "right_hip": "thigh_r",
        "left_knee": "calf_l",
        "right_knee": "calf_r",
        "left_ankle": "foot_l",
        "right_ankle": "foot_r",
    },
}

# Face Action Units for legacy API (Solutions API fallback)
FACE_ACTION_UNITS = {
    "jaw_open": {"landmarks": [13, 14], "reference": [152, 10], "method": "distance_ratio"},
    "mouth_wide": {"landmarks": [61, 291], "reference": [152, 10], "method": "distance_ratio"},
    "mouth_pucker": {"landmarks": [61, 291], "reference": [152, 10], "method": "inverse_distance_ratio"},
    "brow_raise_L": {"landmarks": [65, 158], "reference": [33, 133], "method": "distance_ratio"},
    "brow_raise_R": {"landmarks": [295, 385], "reference": [362, 263], "method": "distance_ratio"},
    "eye_blink_L": {"landmarks": [159, 145], "reference": [33, 133], "method": "inverse_distance_ratio"},
    "eye_blink_R": {"landmarks": [386, 374], "reference": [362, 263], "method": "inverse_distance_ratio"},
    "smile_L": {"landmarks": [61, 48], "reference": [152, 10], "method": "distance_ratio"},
    "smile_R": {"landmarks": [291, 278], "reference": [152, 10], "method": "distance_ratio"},
}

# ═════════════════════════════════════════════════════════════════
# ARKit → Shape Key Mapping
#
# This map now includes CC4/iClone ActorCore naming conventions
# (Brow_Down_L, Eye_Squint_Inner_L, Mouth_Smile_L, etc.)
# alongside standard CC3 export names.
#
# Each ARKit blendshape maps to a list of candidate shape key names.
# The first match found on the mesh wins.
# For "split" blendshapes (one ARKit name drives both L+R),
# we list them as separate L/R entries — handled by the
# apply_blendshapes_direct method.
# ═════════════════════════════════════════════════════════════════

ARKIT_TO_CC3_MAP = {
    # ─── Brows ───
    "browDownLeft": [
        "Brow_Down_L", "Brow_Drop_L", "BrowDrop_L", "brow_drop_l",
    ],
    "browDownRight": [
        "Brow_Down_R", "Brow_Drop_R", "BrowDrop_R", "brow_drop_r",
    ],
    "browInnerUp": [
        # This is a single ARKit shape that raises both inner brows.
        # CC4 splits it into L and R — handled via ARKIT_SPLIT_MAP below.
        "Brow_Raise_Inner_L", "Brow_Raise_Inner_R",
        "Brow_Raise_In_L", "Brow_Raise_In_R",
        "BrowRaiseInner_L", "BrowRaiseInner_R",
        "Brow_Raise_Inner",
    ],
    "browOuterUpLeft": [
        "Brow_Raise_Outer_L", "BrowRaiseOuter_L", "brow_raise_outer_l",
    ],
    "browOuterUpRight": [
        "Brow_Raise_Outer_R", "BrowRaiseOuter_R", "brow_raise_outer_r",
    ],
    # ─── Eyes ───
    "eyeBlinkLeft": [
        "Eye_Blink_L", "EyeBlink_L", "eye_blink_l", "Blink_L",
    ],
    "eyeBlinkRight": [
        "Eye_Blink_R", "EyeBlink_R", "eye_blink_r", "Blink_R",
    ],
    "eyeLookDownLeft": [
        "Eye_Look_Down_L", "EyeLookDown_L",
    ],
    "eyeLookDownRight": [
        "Eye_Look_Down_R", "EyeLookDown_R",
    ],
    "eyeLookInLeft": [
        "Eye_Look_In_L", "EyeLookIn_L", "Eye_L_Look_R",
    ],
    "eyeLookInRight": [
        "Eye_Look_In_R", "EyeLookIn_R", "Eye_R_Look_L",
    ],
    "eyeLookOutLeft": [
        "Eye_Look_Out_L", "EyeLookOut_L", "Eye_L_Look_L",
    ],
    "eyeLookOutRight": [
        "Eye_Look_Out_R", "EyeLookOut_R", "Eye_R_Look_R",
    ],
    "eyeLookUpLeft": [
        "Eye_Look_Up_L", "EyeLookUp_L",
    ],
    "eyeLookUpRight": [
        "Eye_Look_Up_R", "EyeLookUp_R",
    ],
    "eyeSquintLeft": [
        "Eye_Squint_L", "Eye_Squint_Inner_L", "EyeSquint_L", "eye_squint_l",
    ],
    "eyeSquintRight": [
        "Eye_Squint_R", "Eye_Squint_Inner_R", "EyeSquint_R", "eye_squint_r",
    ],
    "eyeWideLeft": [
        "Eye_Wide_L", "Eye_Widen_L", "EyeWide_L", "eye_wide_l",
    ],
    "eyeWideRight": [
        "Eye_Wide_R", "Eye_Widen_R", "EyeWide_R", "eye_wide_r",
    ],
    # ─── Jaw ───
    "jawForward": [
        "Jaw_Forward", "Jaw_Thrust", "JawForward", "jaw_forward",
    ],
    "jawLeft": [
        "Jaw_Left", "Jaw_L", "JawLeft", "jaw_left",
    ],
    "jawRight": [
        "Jaw_Right", "Jaw_R", "JawRight", "jaw_right",
    ],
    "jawOpen": [
        "Jaw_Open", "JawOpen", "jaw_open", "V_Open", "Mouth_Open",
    ],
    # ─── Mouth ───
    "mouthClose": [
        "Mouth_Close", "Mouth_Up", "MouthClose", "mouth_close",
    ],
    "mouthFunnel": [
        "Mouth_Funnel", "V_Tight_O", "MouthFunnel", "mouth_funnel",
    ],
    "mouthPucker": [
        "Mouth_Pucker", "V_Tight", "MouthPucker", "mouth_pucker",
    ],
    "mouthLeft": [
        "Mouth_Left", "Mouth_L", "MouthLeft", "mouth_left",
    ],
    "mouthRight": [
        "Mouth_Right", "Mouth_R", "MouthRight", "mouth_right",
    ],
    "mouthSmileLeft": [
        "Mouth_Smile_L", "Mouth_Smile_Sharp_L", "MouthSmile_L",
        "mouth_smile_l", "Smile_L",
    ],
    "mouthSmileRight": [
        "Mouth_Smile_R", "Mouth_Smile_Sharp_R", "MouthSmile_R",
        "mouth_smile_r", "Smile_R",
    ],
    "mouthFrownLeft": [
        "Mouth_Frown_L", "Mouth_Sad_L", "MouthFrown_L", "mouth_frown_l", "Frown_L",
    ],
    "mouthFrownRight": [
        "Mouth_Frown_R", "Mouth_Sad_R", "MouthFrown_R", "mouth_frown_r", "Frown_R",
    ],
    "mouthDimpleLeft": [
        "Mouth_Dimple_L", "MouthDimple_L",
    ],
    "mouthDimpleRight": [
        "Mouth_Dimple_R", "MouthDimple_R",
    ],
    "mouthStretchLeft": [
        "Mouth_Stretch_L", "MouthStretch_L",
    ],
    "mouthStretchRight": [
        "Mouth_Stretch_R", "MouthStretch_R",
    ],
    "mouthRollLower": [
        "Mouth_LowerLip_RollIn_L", "Mouth_LowerLip_RollIn_R",
        "Mouth_Roll_In_Lower", "MouthRollLower", "Mouth_Roll_Lower",
    ],
    "mouthRollUpper": [
        "Mouth_UpperLip_RollIn_L", "Mouth_UpperLip_RollIn_R",
        "Mouth_Roll_In_Upper", "MouthRollUpper", "Mouth_Roll_Upper",
    ],
    "mouthShrugLower": [
        "Mouth_Shrug_Lower_L", "Mouth_Shrug_Lower_R",
        "Mouth_LowerLip_Push_L", "Mouth_LowerLip_Push_R",
        "Mouth_Shrug_Lower", "MouthShrugLower",
    ],
    "mouthShrugUpper": [
        "Mouth_Shrug_Upper_L", "Mouth_Shrug_Upper_R",
        "Mouth_UpperLip_Push_L", "Mouth_UpperLip_Push_R",
        "Mouth_Shrug_Upper", "MouthShrugUpper",
    ],
    "mouthPressLeft": [
        "Mouth_Press_L", "Mouth_LowerLip_Depress_L", "MouthPress_L",
    ],
    "mouthPressRight": [
        "Mouth_Press_R", "Mouth_LowerLip_Depress_R", "MouthPress_R",
    ],
    "mouthLowerDownLeft": [
        "Mouth_LowerLip_Down_L", "Mouth_Down_Lower_L",
        "Mouth_LowerLip_Drop_L", "MouthLowerDown_L",
    ],
    "mouthLowerDownRight": [
        "Mouth_LowerLip_Down_R", "Mouth_Down_Lower_R",
        "Mouth_LowerLip_Drop_R", "MouthLowerDown_R",
    ],
    "mouthUpperUpLeft": [
        "Mouth_UpperLip_Raise_L", "Mouth_Up_Upper_L", "MouthUpperUp_L",
    ],
    "mouthUpperUpRight": [
        "Mouth_UpperLip_Raise_R", "Mouth_Up_Upper_R", "MouthUpperUp_R",
    ],
    # ─── Cheeks ───
    "cheekPuff": [
        "Cheek_Puff_L", "Cheek_Puff_R", "Cheek_Blow_L", "Cheek_Blow_R",
        "CheekPuff", "cheek_puff",
    ],
    "cheekSquintLeft": [
        "Cheek_Raise_L", "CheekSquint_L", "cheek_squint_l",
    ],
    "cheekSquintRight": [
        "Cheek_Raise_R", "CheekSquint_R", "cheek_squint_r",
    ],
    # ─── Nose ───
    "noseSneerLeft": [
        "Nose_Sneer_L", "Nose_Scrunch_L", "Nose_Wrinkle_L",
        "NoseSneer_L", "nose_sneer_l",
    ],
    "noseSneerRight": [
        "Nose_Sneer_R", "Nose_Scrunch_R", "Nose_Wrinkle_R",
        "NoseSneer_R", "nose_sneer_r",
    ],
    # ─── Tongue ───
    "tongueOut": [
        "Tongue_Out", "TongueOut", "tongue_out",
    ],
}

# ARKit blendshapes that should drive BOTH L and R shape keys simultaneously.
# Format: arkit_name → [(left_shape_key_candidates), (right_shape_key_candidates)]
# If a blendshape is in this map AND in ARKIT_TO_CC3_MAP, this takes priority.
ARKIT_SPLIT_MAP = {
    "browInnerUp": (
        ["Brow_Raise_In_L", "Brow_Raise_Inner_L", "BrowRaiseInner_L"],
        ["Brow_Raise_In_R", "Brow_Raise_Inner_R", "BrowRaiseInner_R"],
    ),
    "cheekPuff": (
        ["Cheek_Puff_L", "Cheek_Blow_L"],
        ["Cheek_Puff_R", "Cheek_Blow_R"],
    ),
    "mouthRollLower": (
        ["Mouth_LowerLip_RollIn_L"],
        ["Mouth_LowerLip_RollIn_R"],
    ),
    "mouthRollUpper": (
        ["Mouth_UpperLip_RollIn_L"],
        ["Mouth_UpperLip_RollIn_R"],
    ),
    "mouthShrugLower": (
        ["Mouth_Shrug_Lower_L", "Mouth_LowerLip_Push_L"],
        ["Mouth_Shrug_Lower_R", "Mouth_LowerLip_Push_R"],
    ),
    "mouthShrugUpper": (
        ["Mouth_Shrug_Upper_L", "Mouth_UpperLip_Push_L"],
        ["Mouth_Shrug_Upper_R", "Mouth_UpperLip_Push_R"],
    ),
}


# ═════════════════════════════════════════════════════════════════
# SECTION 3: LANDMARK DETECTION ENGINE
# ═════════════════════════════════════════════════════════════════

class LandmarkDetector:
    """Handles all landmark detection using MediaPipe."""

    def __init__(self, detect_pose=True, detect_face=False, detect_hands=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 progress_callback=None):
        self.detect_pose = detect_pose
        self.detect_face = detect_face
        self.detect_hands = detect_hands
        self.min_det = min_detection_confidence
        self.min_track = min_tracking_confidence
        self.progress_callback = progress_callback
        self._pose_detector = None
        self._face_detector = None
        self._hand_detector = None
        self._holistic_detector = None
        self._api_version = _MP_API_VERSION
        self._frame_timestamp_ms = 0

    def initialize(self):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("MediaPipe not installed")
        if self._api_version == 'tasks':
            self._initialize_tasks_api()
        elif self._api_version == 'solutions':
            self._initialize_solutions_api()
        else:
            raise RuntimeError("MediaPipe installed but no usable API found.")

    def _initialize_tasks_api(self):
        if self.detect_pose:
            model_path = download_model("pose_landmarker", self.progress_callback)
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_det,
                min_tracking_confidence=self.min_track,
                output_segmentation_masks=False,
            )
            self._pose_detector = mp_vision.PoseLandmarker.create_from_options(options)

        if self.detect_face:
            model_path = download_model("face_landmarker", self.progress_callback)
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=self.min_det,
                min_tracking_confidence=self.min_track,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            self._face_detector = mp_vision.FaceLandmarker.create_from_options(options)

        if self.detect_hands:
            model_path = download_model("hand_landmarker", self.progress_callback)
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=self.min_det,
                min_tracking_confidence=self.min_track,
            )
            self._hand_detector = mp_vision.HandLandmarker.create_from_options(options)

    def _initialize_solutions_api(self):
        if self.detect_pose and self.detect_face:
            self._holistic_detector = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=self.min_det,
                min_tracking_confidence=self.min_track,
            )
        elif self.detect_face:
            self._face_detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_det,
                min_tracking_confidence=self.min_track,
            )
        elif self.detect_pose:
            self._pose_detector = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=self.min_det,
                min_tracking_confidence=self.min_track,
            )

    def process_frame(self, frame_bgr, timestamp_ms=None):
        """Process a single BGR frame. Returns dict with detection results."""
        if timestamp_ms is None:
            timestamp_ms = self._frame_timestamp_ms
            self._frame_timestamp_ms += 33

        output = {
            "pose": None,
            "face": None,
            "face_blendshapes": None,
            "left_hand": None,
            "right_hand": None,
        }

        if self._api_version == 'tasks':
            self._process_tasks_api(frame_bgr, timestamp_ms, output)
        elif self._api_version == 'solutions':
            self._process_solutions_api(frame_bgr, output)

        return output

    def _process_tasks_api(self, frame_bgr, timestamp_ms, output):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._pose_detector:
            try:
                result = self._pose_detector.detect_for_video(mp_image, timestamp_ms)
                if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                    lms = result.pose_world_landmarks[0]
                    output["pose"] = np.array([[lm.x, lm.y, lm.z] for lm in lms])
            except Exception as e:
                print(f"[VMoCap] Pose detection error: {e}")

        if self._face_detector:
            try:
                result = self._face_detector.detect_for_video(mp_image, timestamp_ms)
                if result.face_landmarks and len(result.face_landmarks) > 0:
                    lms = result.face_landmarks[0]
                    output["face"] = np.array([[lm.x, lm.y, lm.z] for lm in lms])
                if result.face_blendshapes and len(result.face_blendshapes) > 0:
                    bs = result.face_blendshapes[0]
                    output["face_blendshapes"] = {b.category_name: b.score for b in bs}
            except Exception as e:
                print(f"[VMoCap] Face detection error: {e}")

        if self._hand_detector:
            try:
                result = self._hand_detector.detect_for_video(mp_image, timestamp_ms)
                if result.hand_landmarks:
                    for i, hlm in enumerate(result.hand_landmarks):
                        side = "left" if result.handedness[i][0].category_name == "Left" else "right"
                        arr = np.array([[lm.x, lm.y, lm.z] for lm in hlm])
                        output[f"{side}_hand"] = arr
            except Exception as e:
                print(f"[VMoCap] Hand detection error: {e}")

    def _process_solutions_api(self, frame_bgr, output):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._holistic_detector:
            results = self._holistic_detector.process(frame_rgb)
            if results.pose_world_landmarks:
                output["pose"] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark
                ])
            if results.face_landmarks:
                output["face"] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
                ])
            if results.left_hand_landmarks:
                output["left_hand"] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
                ])
            if results.right_hand_landmarks:
                output["right_hand"] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
                ])
        elif self._face_detector:
            results = self._face_detector.process(frame_rgb)
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                output["face"] = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark])
        elif self._pose_detector:
            results = self._pose_detector.process(frame_rgb)
            if results.pose_world_landmarks:
                output["pose"] = np.array([
                    [lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark
                ])

    def close(self):
        for det in [self._pose_detector, self._face_detector,
                    self._hand_detector, self._holistic_detector]:
            if det:
                try:
                    det.close()
                except Exception:
                    pass


# ═════════════════════════════════════════════════════════════════
# SECTION 4: COORDINATE TRANSFORMATION & RETARGETING
# ═════════════════════════════════════════════════════════════════

class CoordinateTransformer:
    """
    MediaPipe world landmarks: X-right, Y-down, Z-toward-camera
    Blender:                    X-right, Y-forward, Z-up

    Transform: Bx = Mx, By = -Mz, Bz = -My
    """

    @staticmethod
    def batch_transform(landmarks_array, scale=1.0, camera_angle_deg=0.0):
        if landmarks_array is None:
            return None
        transformed = np.zeros_like(landmarks_array)
        transformed[:, 0] = landmarks_array[:, 0] * scale
        transformed[:, 1] = -landmarks_array[:, 2] * scale
        transformed[:, 2] = -landmarks_array[:, 1] * scale

        if abs(camera_angle_deg) > 0.1:
            angle_rad = math.radians(camera_angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            x = transformed[:, 0].copy()
            y = transformed[:, 1].copy()
            transformed[:, 0] = cos_a * x - sin_a * y
            transformed[:, 1] = sin_a * x + cos_a * y

        return transformed


class PoseRetargeter:
    """
    Retargets detected pose landmarks to an armature.
    Processes bones in hierarchy order and properly accounts for
    parent rotations using the bone's basis matrix.
    """

    def __init__(self, armature_obj, bone_mapping, scale=1.0):
        self.armature = armature_obj
        self.bone_map = bone_mapping
        self.scale = scale
        self._keyframed_count = 0
        self._warnings = set()
        self._sorted_mappings = None
        self._prepare_hierarchy()

    def _prepare_hierarchy(self):
        items_with_depth = []
        for landmark_name, bone_name in self.bone_map.items():
            depth = self._get_bone_depth(bone_name)
            items_with_depth.append((depth, landmark_name, bone_name))
        items_with_depth.sort(key=lambda x: x[0])
        self._sorted_mappings = [(item[1], item[2]) for item in items_with_depth]

    def _get_bone_depth(self, bone_name):
        pose_bone = self.armature.pose.bones.get(bone_name)
        if not pose_bone:
            return 999
        depth = 0
        parent = pose_bone.parent
        while parent:
            depth += 1
            parent = parent.parent
        return depth

    def _compute_basis_3x3(self, pose_bone, posed_matrices):
        bone = pose_bone.bone
        ancestor_pb = pose_bone.parent
        while ancestor_pb:
            if ancestor_pb.name in posed_matrices:
                ancestor_posed_3x3 = posed_matrices[ancestor_pb.name]
                relative = ancestor_pb.bone.matrix_local.to_3x3().inverted() @ bone.matrix_local.to_3x3()
                return ancestor_posed_3x3 @ relative
            ancestor_pb = ancestor_pb.parent
        return bone.matrix_local.to_3x3()

    def _get_target_direction(self, landmark_name, landmarks):
        direction_pair = LANDMARK_DIRECTION_PAIRS.get(landmark_name)
        if not direction_pair:
            return None

        from_idx, to_idx = direction_pair
        if from_idx >= len(landmarks) or to_idx >= len(landmarks):
            return None

        if landmark_name == "nose":
            mid_shoulders = (landmarks[11] + landmarks[12]) / 2.0
            nose_pos = landmarks[0]
            direction = Vector((nose_pos - mid_shoulders).tolist()).normalized()
        elif landmark_name == "spine_direction":
            mid_hips = (landmarks[23] + landmarks[24]) / 2.0
            mid_shoulders = (landmarks[11] + landmarks[12]) / 2.0
            direction = Vector((mid_shoulders - mid_hips).tolist()).normalized()
        else:
            pos_from = Vector(landmarks[from_idx].tolist())
            pos_to = Vector(landmarks[to_idx].tolist())
            direction = (pos_to - pos_from).normalized()

        if direction.length < 0.0001:
            return None
        return direction

    def apply_pose_frame(self, landmarks_blender, frame):
        if landmarks_blender is None:
            return

        posed_matrices = {}

        for landmark_name, bone_name in self._sorted_mappings:
            if not bone_name:
                continue

            pose_bone = self.armature.pose.bones.get(bone_name)
            if not pose_bone:
                if bone_name not in self._warnings:
                    self._warnings.add(bone_name)
                    print(f"[VMoCap] Bone '{bone_name}' not found on armature!")
                continue

            target_dir = self._get_target_direction(landmark_name, landmarks_blender)
            if target_dir is None:
                if landmark_name not in self._warnings:
                    self._warnings.add(landmark_name)
                    print(f"[VMoCap] Could not compute direction for '{landmark_name}'")
                continue

            basis_3x3 = self._compute_basis_3x3(pose_bone, posed_matrices)
            local_target = (basis_3x3.inverted() @ target_dir).normalized()

            if local_target.length < 0.0001:
                continue

            local_rot = Vector((0, 1, 0)).rotation_difference(local_target)

            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = local_rot
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            self._keyframed_count += 1

            posed_matrices[bone_name] = basis_3x3 @ local_rot.to_matrix()

    def get_keyframe_count(self):
        return self._keyframed_count

    def get_warnings(self):
        return self._warnings


class FaceRetargeter:
    """
    Retargets face data to shape keys on multiple mesh objects.
    Supports Tasks API blendshapes and legacy computed Action Units.

    Key features:
    - Strict side matching: Left ARKit shapes only match L shape keys.
    - Split blendshapes: Some ARKit shapes (browInnerUp, cheekPuff) drive
      both L and R shape keys simultaneously.
    - Multi-strategy matching with side-awareness.
    """

    def __init__(self, mesh_objects):
        self.meshes = [m for m in mesh_objects if m is not None]
        self.au_calibration = {}
        self.calibrated = False
        self._shapekey_mesh_map = {}
        self._shapekey_names_lower = {}
        self._shapekey_names_normalized = {}
        self._match_cache = {}
        self._split_cache = {}
        self._keyframed_count = 0
        self._unmatched = set()
        self._matched_report = {}
        self._build_shapekey_map()

    def _build_shapekey_map(self):
        """Scan all mesh targets and catalog available shape keys."""
        self._shapekey_mesh_map = {}
        self._shapekey_names_lower = {}
        self._shapekey_names_normalized = {}

        for mesh_obj in self.meshes:
            if not mesh_obj.data.shape_keys:
                continue
            for kb in mesh_obj.data.shape_keys.key_blocks:
                if kb.name == "Basis":
                    continue
                if kb.name not in self._shapekey_mesh_map:
                    self._shapekey_mesh_map[kb.name] = []
                self._shapekey_mesh_map[kb.name].append((mesh_obj, kb))

                lower = kb.name.lower()
                if lower not in self._shapekey_names_lower:
                    self._shapekey_names_lower[lower] = kb.name

                normalized = self._normalize_name(kb.name)
                if normalized not in self._shapekey_names_normalized:
                    self._shapekey_names_normalized[normalized] = kb.name

    def _normalize_name(self, name):
        """Normalize for fuzzy matching: strip prefixes, remove separators, lowercase."""
        s = name
        s = re.sub(r'^[A-Z]\d{2}_', '', s)
        s = re.sub(r'^CC_Base_', '', s, flags=re.IGNORECASE)
        s = re.sub(r'^C_', '', s)
        s = s.replace('_', '').replace('.', '').replace('-', '').replace(' ', '').lower()
        return s

    def _get_arkit_side(self, arkit_name):
        """
        Determine which side an ARKit blendshape targets.
        Returns 'L', 'R', or None (bilateral/center).
        """
        if arkit_name.endswith("Left"):
            return 'L'
        if arkit_name.endswith("Right"):
            return 'R'
        # Check for known single-side names
        if arkit_name.endswith("_L") or arkit_name.endswith(".L"):
            return 'L'
        if arkit_name.endswith("_R") or arkit_name.endswith(".R"):
            return 'R'
        return None

    def _get_shapekey_side(self, sk_name):
        """
        Determine side of a shape key from its name.
        Returns 'L', 'R', or None.
        """
        lower = sk_name.lower()
        # Check common suffixes
        if lower.endswith('_l') or lower.endswith('.l') or lower.endswith('l') and (
                len(lower) > 1 and lower[-2] in ('_', '.', ' ')):
            return 'L'
        if lower.endswith('_r') or lower.endswith('.r') or lower.endswith('r') and (
                len(lower) > 1 and lower[-2] in ('_', '.', ' ')):
            return 'R'
        # Check for L/R anywhere with separators
        if re.search(r'[_.]l[_.]|[_.]l$|_l_|^l_', lower):
            return 'L'
        if re.search(r'[_.]r[_.]|[_.]r$|_r_|^r_', lower):
            return 'R'
        # Check for "Left"/"Right" in name
        if 'left' in lower or '_l_' in lower:
            return 'L'
        if 'right' in lower or '_r_' in lower:
            return 'R'
        return None

    def _sides_compatible(self, arkit_side, sk_side):
        """
        Check if an ARKit blendshape side is compatible with a shape key side.
        - If ARKit is None (bilateral), it can match anything.
        - If ARKit is L, it can only match L or None (non-sided).
        - If ARKit is R, it can only match R or None (non-sided).
        """
        if arkit_side is None:
            return True
        if sk_side is None:
            return True
        return arkit_side == sk_side

    def _find_matching_shapekey(self, arkit_name):
        """
        Find the best matching shape key for an ARKit blendshape name.
        Enforces side-matching to prevent L shapes from matching R keys.
        """
        if arkit_name in self._match_cache:
            return self._match_cache[arkit_name]

        arkit_side = self._get_arkit_side(arkit_name)
        matched = None

        # Strategy 1: Direct match
        if arkit_name in self._shapekey_mesh_map:
            matched = arkit_name

        # Strategy 2: CC3/CC4 lookup table (explicit, already side-correct)
        if not matched and arkit_name in ARKIT_TO_CC3_MAP:
            for candidate in ARKIT_TO_CC3_MAP[arkit_name]:
                if candidate in self._shapekey_mesh_map:
                    # Verify side compatibility
                    sk_side = self._get_shapekey_side(candidate)
                    if self._sides_compatible(arkit_side, sk_side):
                        matched = candidate
                        break

        # Strategy 3: Generated variants (side-aware)
        if not matched:
            for variant in self._generate_variants(arkit_name):
                if variant in self._shapekey_mesh_map:
                    sk_side = self._get_shapekey_side(variant)
                    if self._sides_compatible(arkit_side, sk_side):
                        matched = variant
                        break

        # Strategy 4: Case-insensitive direct
        if not matched:
            lower = arkit_name.lower()
            if lower in self._shapekey_names_lower:
                candidate = self._shapekey_names_lower[lower]
                sk_side = self._get_shapekey_side(candidate)
                if self._sides_compatible(arkit_side, sk_side):
                    matched = candidate

        # Strategy 5: Case-insensitive CC3 table variants
        if not matched and arkit_name in ARKIT_TO_CC3_MAP:
            for candidate in ARKIT_TO_CC3_MAP[arkit_name]:
                cl = candidate.lower()
                if cl in self._shapekey_names_lower:
                    actual = self._shapekey_names_lower[cl]
                    sk_side = self._get_shapekey_side(actual)
                    if self._sides_compatible(arkit_side, sk_side):
                        matched = actual
                        break

        # Strategy 6: Normalized fuzzy match (side-aware)
        if not matched:
            normalized_arkit = self._normalize_name(arkit_name)
            if normalized_arkit in self._shapekey_names_normalized:
                candidate = self._shapekey_names_normalized[normalized_arkit]
                sk_side = self._get_shapekey_side(candidate)
                if self._sides_compatible(arkit_side, sk_side):
                    matched = candidate

        # Strategy 7: Normalized CC3 table variants
        if not matched and arkit_name in ARKIT_TO_CC3_MAP:
            for candidate in ARKIT_TO_CC3_MAP[arkit_name]:
                norm_candidate = self._normalize_name(candidate)
                if norm_candidate in self._shapekey_names_normalized:
                    actual = self._shapekey_names_normalized[norm_candidate]
                    sk_side = self._get_shapekey_side(actual)
                    if self._sides_compatible(arkit_side, sk_side):
                        matched = actual
                        break

        self._match_cache[arkit_name] = matched
        return matched

    def _find_split_targets(self, arkit_name):
        """
        For bilateral ARKit shapes that should drive both L and R,
        find both target shape keys.
        Returns list of matched shape key names, or empty list.
        """
        if arkit_name in self._split_cache:
            return self._split_cache[arkit_name]

        results = []
        if arkit_name not in ARKIT_SPLIT_MAP:
            self._split_cache[arkit_name] = results
            return results

        left_candidates, right_candidates = ARKIT_SPLIT_MAP[arkit_name]

        # Find left
        left_match = None
        for candidate in left_candidates:
            if candidate in self._shapekey_mesh_map:
                left_match = candidate
                break
            cl = candidate.lower()
            if cl in self._shapekey_names_lower:
                left_match = self._shapekey_names_lower[cl]
                break

        # Find right
        right_match = None
        for candidate in right_candidates:
            if candidate in self._shapekey_mesh_map:
                right_match = candidate
                break
            cl = candidate.lower()
            if cl in self._shapekey_names_lower:
                right_match = self._shapekey_names_lower[cl]
                break

        if left_match:
            results.append(left_match)
        if right_match:
            results.append(right_match)

        self._split_cache[arkit_name] = results
        return results

    def _generate_variants(self, name):
        """Generate naming variants for an ARKit blendshape name."""
        variants = []

        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        variants.append(snake)

        if name.endswith("Left"):
            base = name[:-4]
            base_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', base)
            variants.extend([
                base + "_L", base + ".L", base + "L",
                base_snake + "_L", base_snake + ".L",
                base_snake.lower() + "_l", base_snake.lower() + ".l",
            ])
            snake_base = snake[:-5] if snake.endswith("_left") else snake
            variants.extend([
                snake_base + "_l", snake_base + "_L",
            ])
        elif name.endswith("Right"):
            base = name[:-5]
            base_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', base)
            variants.extend([
                base + "_R", base + ".R", base + "R",
                base_snake + "_R", base_snake + ".R",
                base_snake.lower() + "_r", base_snake.lower() + ".r",
            ])
            snake_base = snake[:-6] if snake.endswith("_right") else snake
            variants.extend([
                snake_base + "_r", snake_base + "_R",
            ])

        title_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
        variants.append(title_snake)

        seen = set()
        unique = []
        for v in variants:
            if v not in seen and v != name:
                seen.add(v)
                unique.append(v)
        return unique

    def get_available_shapekeys(self):
        return set(self._shapekey_mesh_map.keys())

    def get_keyframe_count(self):
        return self._keyframed_count

    def print_diagnostics(self, sample_blendshapes=None):
        """Print diagnostic info about available shape keys and matching."""
        print(f"[VMoCap] --- Face Retargeter Diagnostics ---")
        print(f"[VMoCap] Total shape keys indexed: {len(self._shapekey_mesh_map)}")

        all_names = sorted(self._shapekey_mesh_map.keys())
        print(f"[VMoCap] First 20 shape key names on meshes:")
        for name in all_names[:20]:
            print(f"[VMoCap]   '{name}'")
        if len(all_names) > 20:
            print(f"[VMoCap]   ... and {len(all_names) - 20} more")

        if sample_blendshapes:
            print(f"[VMoCap] MediaPipe blendshape matching results:")
            matched_count = 0
            for bs_name in sorted(sample_blendshapes.keys()):
                if bs_name == "_neutral":
                    continue
                weight = sample_blendshapes[bs_name]

                # Check split first
                split_targets = self._find_split_targets(bs_name)
                if split_targets:
                    matched_count += 1
                    targets_str = " + ".join(split_targets)
                    print(f"[VMoCap]   {bs_name} (w={weight:.3f}) -> SPLIT: [{targets_str}]")
                    continue

                match = self._find_matching_shapekey(bs_name)
                if match:
                    matched_count += 1
                    status = f"-> '{match}'"
                else:
                    status = "X NO MATCH"
                print(f"[VMoCap]   {bs_name} (w={weight:.3f}) {status}")

            total_bs = len(sample_blendshapes) - (1 if "_neutral" in sample_blendshapes else 0)
            print(f"[VMoCap] Matched: {matched_count}/{total_bs} blendshapes")
        print(f"[VMoCap] -----------------------------------------")

    def calibrate(self, neutral_landmarks):
        """Calibrate neutral face from first detected frame (legacy API)."""
        if neutral_landmarks is None:
            return
        for au_name, au_config in FACE_ACTION_UNITS.items():
            lm_ids = au_config["landmarks"]
            ref_ids = au_config["reference"]
            p1 = neutral_landmarks[lm_ids[0]]
            p2 = neutral_landmarks[lm_ids[1]]
            dist = np.linalg.norm(p1 - p2)
            r1 = neutral_landmarks[ref_ids[0]]
            r2 = neutral_landmarks[ref_ids[1]]
            ref_dist = np.linalg.norm(r1 - r2)
            self.au_calibration[au_name] = {
                "neutral_ratio": dist / ref_dist if ref_dist > 0 else 0,
            }
        self.calibrated = True

    def compute_action_units(self, face_landmarks):
        """Compute AU weights from face landmarks (legacy fallback)."""
        if face_landmarks is None:
            return {}
        au_weights = {}
        for au_name, au_config in FACE_ACTION_UNITS.items():
            lm_ids = au_config["landmarks"]
            ref_ids = au_config["reference"]
            method = au_config["method"]
            p1 = face_landmarks[lm_ids[0]]
            p2 = face_landmarks[lm_ids[1]]
            dist = np.linalg.norm(p1 - p2)
            r1 = face_landmarks[ref_ids[0]]
            r2 = face_landmarks[ref_ids[1]]
            ref_dist = np.linalg.norm(r1 - r2)
            if ref_dist == 0:
                au_weights[au_name] = 0.0
                continue
            current_ratio = dist / ref_dist
            if self.calibrated and au_name in self.au_calibration:
                neutral_ratio = self.au_calibration[au_name]["neutral_ratio"]
            else:
                neutral_ratio = current_ratio * 0.7
            if method == "distance_ratio":
                weight = (current_ratio - neutral_ratio) / (neutral_ratio + 1e-6)
                weight = max(0.0, min(1.0, weight * 2.0))
            elif method == "inverse_distance_ratio":
                weight = (neutral_ratio - current_ratio) / (neutral_ratio + 1e-6)
                weight = max(0.0, min(1.0, weight * 2.0))
            else:
                weight = 0.0
            au_weights[au_name] = weight
        return au_weights

    def apply_blendshapes_direct(self, blendshape_weights, frame, is_first_frame=False):
        """
        Apply ARKit blendshape weights directly to shape keys (Tasks API).
        Handles split blendshapes (one ARKit name -> multiple shape keys).
        """
        if not blendshape_weights:
            return

        if is_first_frame:
            self.print_diagnostics(blendshape_weights)

        for bs_name, weight in blendshape_weights.items():
            if bs_name == "_neutral":
                continue
            if weight < 0.005:
                continue

            # Check if this is a split blendshape first
            split_targets = self._find_split_targets(bs_name)
            if split_targets:
                for sk_name in split_targets:
                    if sk_name in self._shapekey_mesh_map:
                        for mesh_obj, kb in self._shapekey_mesh_map[sk_name]:
                            kb.value = weight
                            kb.keyframe_insert(data_path="value", frame=frame)
                            self._keyframed_count += 1
                if bs_name not in self._matched_report:
                    self._matched_report[bs_name] = f"SPLIT: {split_targets}"
                continue

            # Standard single-target match
            matched_sk = self._find_matching_shapekey(bs_name)

            if matched_sk and matched_sk in self._shapekey_mesh_map:
                for mesh_obj, kb in self._shapekey_mesh_map[matched_sk]:
                    kb.value = weight
                    kb.keyframe_insert(data_path="value", frame=frame)
                    self._keyframed_count += 1
                if bs_name not in self._matched_report:
                    self._matched_report[bs_name] = matched_sk
            else:
                self._unmatched.add(bs_name)

    def apply_action_units(self, au_weights, frame):
        """Apply computed AU weights to shape keys (legacy fallback)."""
        if not au_weights:
            return
        au_to_shapekey = {
            "jaw_open": ["jawOpen", "jaw_open", "JawOpen", "Jaw_Open", "V_Open"],
            "mouth_wide": ["Mouth_Smile_L", "Mouth_Smile_R", "mouthSmile", "mouth_wide"],
            "mouth_pucker": ["Mouth_Pucker", "V_Tight", "mouthPucker", "mouth_pucker"],
            "brow_raise_L": ["Brow_Raise_In_L", "Brow_Raise_Inner_L", "browInnerUp"],
            "brow_raise_R": ["Brow_Raise_In_R", "Brow_Raise_Inner_R", "browInnerUp"],
            "eye_blink_L": ["Eye_Blink_L", "eyeBlinkLeft", "Blink_L"],
            "eye_blink_R": ["Eye_Blink_R", "eyeBlinkRight", "Blink_R"],
            "smile_L": ["Mouth_Smile_L", "mouthSmileLeft", "Smile_L"],
            "smile_R": ["Mouth_Smile_R", "mouthSmileRight", "Smile_R"],
        }
        for au_name, weight in au_weights.items():
            if weight < 0.005:
                continue
            possible_names = au_to_shapekey.get(au_name, [au_name])
            for sk_name in possible_names:
                if sk_name in self._shapekey_mesh_map:
                    for mesh_obj, kb in self._shapekey_mesh_map[sk_name]:
                        kb.value = weight
                        kb.keyframe_insert(data_path="value", frame=frame)
                        self._keyframed_count += 1
                    break

    def get_unmatched(self):
        return self._unmatched

    def get_matched_report(self):
        return self._matched_report


# ═════════════════════════════════════════════════════════════════
# SECTION 5: VIDEO PROCESSING
# ═════════════════════════════════════════════════════════════════

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0

    def open(self):
        if not _HAS_CV2:
            raise RuntimeError("OpenCV not installed")
        if not os.path.isfile(self.video_path):
            raise RuntimeError(f"Video file not found: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.total_frames <= 0:
            raise RuntimeError(f"Video reports 0 frames: {self.video_path}")

    def read_frame(self, frame_number=None):
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# ═════════════════════════════════════════════════════════════════
# SECTION 6: GUIDED MODE — POINT TRACKING
# ═════════════════════════════════════════════════════════════════

class PointTracker:
    def __init__(self):
        self.tracked_points = {}
        self.initial_points = {}

    def add_point(self, name, x, y):
        self.initial_points[name] = (x, y)
        self.tracked_points[name] = [(x, y)]

    def track_through_video(self, video_processor, start_frame=0):
        if not _HAS_CV2:
            raise RuntimeError("OpenCV not installed")
        video_processor.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = video_processor.cap.read()
        if not ret:
            return self.tracked_points
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        point_names = list(self.initial_points.keys())
        prev_points = np.array(
            [self.initial_points[n] for n in point_names], dtype=np.float32
        ).reshape(-1, 1, 2)
        lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        while True:
            ret, frame = video_processor.cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray, prev_points, None, **lk_params
            )
            for i, name in enumerate(point_names):
                if status[i][0] == 1:
                    x, y = next_points[i][0]
                    self.tracked_points[name].append((float(x), float(y)))
                else:
                    self.tracked_points[name].append(self.tracked_points[name][-1])
            prev_gray = frame_gray
            prev_points = next_points
        return self.tracked_points


# ═════════════════════════════════════════════════════════════════
# SECTION 7: SMOOTHING & POST-PROCESSING
# ═════════════════════════════════════════════════════════════════

class AnimationSmoother:
    @staticmethod
    def moving_average(data, window_size=5):
        if len(data) < window_size:
            return data.copy()
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')

    @staticmethod
    def one_euro_filter(data, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        dx = 0.0
        for i in range(1, len(data)):
            dx_raw = data[i] - filtered[i - 1]
            alpha_d = 1.0 / (1.0 + 1.0 / (2 * math.pi * d_cutoff))
            dx = alpha_d * dx_raw + (1 - alpha_d) * dx
            cutoff = min_cutoff + beta * abs(dx)
            alpha = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff))
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i - 1]
        return filtered

    @staticmethod
    def butterworth_lowpass(data, cutoff_freq=6.0, sample_rate=30.0, order=2):
        try:
            from scipy.signal import butter, filtfilt
            nyquist = 0.5 * sample_rate
            normal_cutoff = min(cutoff_freq / nyquist, 0.99)
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)
        except ImportError:
            return AnimationSmoother.moving_average(data, window_size=5)

    @staticmethod
    def smooth_fcurves(action, filter_type="one_euro", strength=0.5):
        if not action:
            return 0
        smoothed_count = 0
        for fcurve in action.fcurves:
            if len(fcurve.keyframe_points) < 3:
                continue
            values = np.array([kp.co[1] for kp in fcurve.keyframe_points])

            if filter_type == "one_euro":
                min_cutoff = 1.0 + (1.0 - strength) * 5.0
                beta_val = strength * 0.5
                smoothed = AnimationSmoother.one_euro_filter(
                    values, min_cutoff=min_cutoff, beta=beta_val
                )
            elif filter_type == "moving_average":
                window = int(3 + strength * 12)
                if window % 2 == 0:
                    window += 1
                smoothed = AnimationSmoother.moving_average(values, window_size=window)
            elif filter_type == "butterworth":
                cutoff = 3.0 + (1.0 - strength) * 12.0
                smoothed = AnimationSmoother.butterworth_lowpass(
                    values, cutoff_freq=cutoff
                )
            else:
                continue

            for i, kp in enumerate(fcurve.keyframe_points):
                kp.co[1] = smoothed[i]
            fcurve.update()
            smoothed_count += 1
        return smoothed_count

    @staticmethod
    def smooth_all_targets(armature, mesh_objects, filter_type, strength):
        """Smooth animation on armature and all mesh shape key actions."""
        total_smoothed = 0
        if armature and armature.animation_data and armature.animation_data.action:
            count = AnimationSmoother.smooth_fcurves(
                armature.animation_data.action, filter_type=filter_type, strength=strength
            )
            total_smoothed += count
            print(f"[VMoCap] Smoothed {count} F-Curves on armature action")

        for mesh_obj in mesh_objects:
            if not mesh_obj or not mesh_obj.data.shape_keys:
                continue
            anim_data = mesh_obj.data.shape_keys.animation_data
            if anim_data and anim_data.action:
                count = AnimationSmoother.smooth_fcurves(
                    anim_data.action, filter_type=filter_type, strength=strength
                )
                total_smoothed += count
                print(f"[VMoCap] Smoothed {count} F-Curves on '{mesh_obj.name}' shape keys")

        return total_smoothed


# ═════════════════════════════════════════════════════════════════
# SECTION 8: BLENDER PROPERTIES
# ═════════════════════════════════════════════════════════════════

class VMOCAP_BoneMapItem(PropertyGroup):
    landmark_name: StringProperty(name="Landmark", default="")
    bone_name: StringProperty(name="Bone", default="")
    enabled: BoolProperty(name="Enabled", default=True)


class VMOCAP_TrackedPointItem(PropertyGroup):
    name: StringProperty(name="Point Name")
    target_bone: StringProperty(name="Target Bone")
    position_x: FloatProperty(name="X")
    position_y: FloatProperty(name="Y")


class VMOCAP_MeshTargetItem(PropertyGroup):
    mesh_object: PointerProperty(
        name="Mesh", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )
    enabled: BoolProperty(name="Enabled", default=True)


class VMOCAP_Properties(PropertyGroup):
    video_path: StringProperty(
        name="Video File", subtype='FILE_PATH',
        description="Path to the input video file"
    )

    mode: EnumProperty(
        name="Capture Mode",
        items=[
            ('AUTO_BODY', "Auto - Body", "Automatic full body pose detection"),
            ('AUTO_FACE', "Auto - Face", "Automatic facial capture"),
            ('AUTO_HOLISTIC', "Auto - Full", "Body + Face + Hands"),
            ('GUIDED', "Guided", "Manual point placement and tracking"),
        ],
        default='AUTO_BODY'
    )

    target_armature: PointerProperty(
        name="Target Armature", type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )

    mesh_targets: CollectionProperty(type=VMOCAP_MeshTargetItem)
    mesh_targets_index: IntProperty()

    start_frame: IntProperty(name="Start Frame", default=1, min=1)

    scale_factor: FloatProperty(
        name="Scale", default=1.0, min=0.01, max=100.0,
        description="Scale factor for landmark positions"
    )

    camera_angle: FloatProperty(
        name="Camera Angle",
        default=0.0, min=-180.0, max=180.0,
        description=(
            "Horizontal angle of camera relative to subject's front face.\n"
            "0 = Camera facing front of subject\n"
            "90 = Camera to subject's right side\n"
            "-90 = Camera to subject's left side\n"
            "180 = Camera behind subject"
        ),
        subtype='ANGLE'
    )

    camera_preset: EnumProperty(
        name="Camera View",
        items=[
            ('FRONT', "Front", "Camera facing subject's front (0)"),
            ('BACK', "Back", "Camera behind subject (180)"),
            ('LEFT', "Left Side", "Camera to subject's left (-90)"),
            ('RIGHT', "Right Side", "Camera to subject's right (90)"),
            ('FRONT_LEFT', "Front-Left 45", "Camera at -45"),
            ('FRONT_RIGHT', "Front-Right 45", "Camera at 45"),
            ('CUSTOM', "Custom", "Use custom angle value"),
        ],
        default='FRONT',
        description="Quick preset for camera position relative to subject"
    )

    smoothing_type: EnumProperty(
        name="Smoothing",
        items=[
            ('NONE', "None", "No smoothing"),
            ('ONE_EURO', "One Euro", "Adaptive noise reduction (recommended)"),
            ('MOVING_AVG', "Moving Average", "Simple averaging filter"),
            ('BUTTERWORTH', "Butterworth", "Low-pass frequency filter"),
        ],
        default='ONE_EURO'
    )

    smoothing_strength: FloatProperty(
        name="Strength", default=0.5, min=0.0, max=1.0,
        description="How aggressively to smooth (higher = smoother but less responsive)"
    )

    detection_confidence: FloatProperty(
        name="Detection Confidence", default=0.5, min=0.1, max=1.0
    )
    tracking_confidence: FloatProperty(
        name="Tracking Confidence", default=0.5, min=0.1, max=1.0
    )
    frame_step: IntProperty(
        name="Frame Step", default=1, min=1, max=10,
        description="Process every Nth frame (1 = every frame)"
    )

    bone_mapping: CollectionProperty(type=VMOCAP_BoneMapItem)
    bone_mapping_index: IntProperty()

    tracked_points: CollectionProperty(type=VMOCAP_TrackedPointItem)
    tracked_points_index: IntProperty()

    is_processing: BoolProperty(default=False)
    progress: FloatProperty(default=0.0, min=0.0, max=1.0)
    status_message: StringProperty(default="")


# ═════════════════════════════════════════════════════════════════
# SECTION 9: OPERATORS
# ═════════════════════════════════════════════════════════════════

class VMOCAP_OT_install_dependencies(Operator):
    """Install required Python packages"""
    bl_idname = "vmocap.install_dependencies"
    bl_label = "Install Dependencies"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _HAS_CV2, _HAS_MEDIAPIPE, _MP_API_VERSION
        python_exe = sys.executable
        packages = []
        if not _HAS_CV2:
            packages.append("opencv-contrib-python")
        if not _HAS_MEDIAPIPE or _MP_API_VERSION is None:
            packages.append("mediapipe")
        if not packages:
            self.report({'INFO'}, "All dependencies already installed.")
            return {'FINISHED'}

        target_dir = None
        for p in sys.path:
            if "site-packages" in p and os.path.isdir(p) and os.access(p, os.W_OK):
                target_dir = p
                break

        installed, failed = [], []
        for pkg in packages:
            success = False
            attempts = []
            if target_dir:
                attempts.append([python_exe, "-m", "pip", "install", "--target", target_dir, pkg])
            attempts.append([python_exe, "-m", "pip", "install", "--user", pkg])
            attempts.append([python_exe, "-m", "pip", "install", pkg])

            for cmd in attempts:
                try:
                    subprocess.check_call(cmd, timeout=300)
                    success = True
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                    continue
            (installed if success else failed).append(pkg)

        try:
            import cv2
            _HAS_CV2 = True
        except ImportError:
            pass
        try:
            import mediapipe as mp_reload
            _HAS_MEDIAPIPE = True
            try:
                from mediapipe.tasks import python as _
                from mediapipe.tasks.python import vision as _
                _MP_API_VERSION = 'tasks'
            except (ImportError, AttributeError):
                try:
                    _ = mp_reload.solutions
                    _MP_API_VERSION = 'solutions'
                except AttributeError:
                    _MP_API_VERSION = None
        except ImportError:
            pass

        if failed:
            self.report({'WARNING'},
                f"Failed: {', '.join(failed)}. Try running Blender as Administrator.")
        if installed:
            self.report({'INFO'},
                f"Installed: {', '.join(installed)}. Restart Blender to complete setup.")
        return {'FINISHED'} if installed else {'CANCELLED'}


class VMOCAP_OT_download_models(Operator):
    """Download required MediaPipe model files"""
    bl_idname = "vmocap.download_models"
    bl_label = "Download Models"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.vmocap
        models_needed = []
        if props.mode in ('AUTO_BODY', 'AUTO_HOLISTIC'):
            models_needed.append("pose_landmarker")
        if props.mode in ('AUTO_FACE', 'AUTO_HOLISTIC'):
            models_needed.append("face_landmarker")
        try:
            for name in models_needed:
                self.report({'INFO'}, f"Downloading {name}...")
                download_model(name)
            self.report({'INFO'}, "All models downloaded successfully!")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


class VMOCAP_OT_add_mesh_target(Operator):
    """Add a mesh target for face/shape key capture"""
    bl_idname = "vmocap.add_mesh_target"
    bl_label = "Add Mesh Target"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.vmocap.mesh_targets.add()
        return {'FINISHED'}


class VMOCAP_OT_remove_mesh_target(Operator):
    """Remove a mesh target"""
    bl_idname = "vmocap.remove_mesh_target"
    bl_label = "Remove Mesh Target"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()

    def execute(self, context):
        props = context.scene.vmocap
        if 0 <= self.index < len(props.mesh_targets):
            props.mesh_targets.remove(self.index)
        return {'FINISHED'}


class VMOCAP_OT_auto_discover_meshes(Operator):
    """Find meshes with shape keys parented to or associated with the armature"""
    bl_idname = "vmocap.auto_discover_meshes"
    bl_label = "Auto-Find Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.vmocap.target_armature is not None

    def execute(self, context):
        props = context.scene.vmocap
        armature = props.target_armature
        props.mesh_targets.clear()
        found_set = set()

        for obj in bpy.data.objects:
            if obj.type != 'MESH' or not obj.data.shape_keys:
                continue
            if obj.name in found_set:
                continue
            if obj.parent == armature:
                item = props.mesh_targets.add()
                item.mesh_object = obj
                item.enabled = True
                found_set.add(obj.name)
                continue
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.object == armature:
                    item = props.mesh_targets.add()
                    item.mesh_object = obj
                    item.enabled = True
                    found_set.add(obj.name)
                    break

        if not found_set:
            for col in armature.users_collection:
                for obj in col.objects:
                    if obj.type == 'MESH' and obj.data.shape_keys and obj.name not in found_set:
                        item = props.mesh_targets.add()
                        item.mesh_object = obj
                        item.enabled = True
                        found_set.add(obj.name)

        self.report({'INFO'}, f"Found {len(found_set)} mesh(es) with shape keys")
        return {'FINISHED'}


class VMOCAP_OT_load_bone_mapping(Operator):
    """Load a preset bone mapping for common rig types"""
    bl_idname = "vmocap.load_bone_mapping"
    bl_label = "Load Mapping Preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(
        name="Preset",
        items=[
            ('RIGIFY', "Rigify", "Blender Rigify"),
            ('MIXAMO', "Mixamo", "Mixamo/Adobe"),
            ('CC3', "CC3/CC4", "Character Creator 3/4"),
            ('CC4', "CC4", "Character Creator 4 (same as CC3)"),
            ('UE_MANNEQUIN', "UE Mannequin", "Unreal Engine Mannequin"),
        ]
    )

    def execute(self, context):
        props = context.scene.vmocap
        props.bone_mapping.clear()
        mapping = BONE_MAP_PRESETS.get(self.preset, {})

        armature = props.target_armature
        valid_count = 0
        for landmark, bone in mapping.items():
            item = props.bone_mapping.add()
            item.landmark_name = landmark
            item.bone_name = bone
            item.enabled = True
            if armature and armature.pose.bones.get(bone):
                valid_count += 1

        if armature:
            self.report({'INFO'},
                f"Loaded {self.preset} preset: {valid_count}/{len(mapping)} bones found on rig")
        else:
            self.report({'INFO'}, f"Loaded {self.preset} preset ({len(mapping)} mappings)")
        return {'FINISHED'}


class VMOCAP_OT_add_bone_mapping(Operator):
    """Add a new blank bone mapping entry"""
    bl_idname = "vmocap.add_bone_mapping"
    bl_label = "Add Mapping Entry"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        item = context.scene.vmocap.bone_mapping.add()
        item.landmark_name = ""
        item.bone_name = ""
        item.enabled = True
        return {'FINISHED'}


class VMOCAP_OT_remove_bone_mapping(Operator):
    """Remove a bone mapping entry"""
    bl_idname = "vmocap.remove_bone_mapping"
    bl_label = "Remove Mapping Entry"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()

    def execute(self, context):
        props = context.scene.vmocap
        if 0 <= self.index < len(props.bone_mapping):
            props.bone_mapping.remove(self.index)
        return {'FINISHED'}


class VMOCAP_OT_clear_bone_mapping(Operator):
    """Clear all bone mapping entries"""
    bl_idname = "vmocap.clear_bone_mapping"
    bl_label = "Clear All Mappings"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.vmocap.bone_mapping.clear()
        return {'FINISHED'}


class VMOCAP_OT_auto_map_bones(Operator):
    """Attempt to automatically map landmarks to armature bones by name similarity"""
    bl_idname = "vmocap.auto_map_bones"
    bl_label = "Auto-Map Bones"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.vmocap.target_armature is not None

    def execute(self, context):
        props = context.scene.vmocap
        armature = props.target_armature
        props.bone_mapping.clear()

        bone_names = [b.name for b in armature.data.bones]

        preset_match = self._try_preset_match(bone_names)
        if preset_match:
            mapping = BONE_MAP_PRESETS[preset_match]
            valid = 0
            for landmark, bone in mapping.items():
                if bone in bone_names:
                    item = props.bone_mapping.add()
                    item.landmark_name = landmark
                    item.bone_name = bone
                    item.enabled = True
                    valid += 1
            self.report({'INFO'},
                f"Detected {preset_match} rig! Loaded preset ({valid} bones matched)")
            return {'FINISHED'}

        mapped_count = self._heuristic_match(props, bone_names)
        self.report({'INFO'},
            f"Auto-mapped {mapped_count} bones. Please verify the mapping is correct!")
        return {'FINISHED'}

    def _try_preset_match(self, bone_names):
        bone_set = set(bone_names)
        best_preset = None
        best_count = 0
        for preset_name, mapping in BONE_MAP_PRESETS.items():
            matches = sum(1 for bone in mapping.values() if bone in bone_set)
            if matches >= 8 and matches > best_count:
                best_count = matches
                best_preset = preset_name
        return best_preset

    def _heuristic_match(self, props, bone_names):
        bone_names_normalized = {}
        for bn in bone_names:
            normalized = bn.lower().replace("_", "").replace(".", "").replace("-", "").replace(":", "").replace(" ", "")
            bone_names_normalized[bn] = normalized

        search_config = [
            ("left_shoulder", "left", [("upperarm", 10), ("uparm", 9)]),
            ("right_shoulder", "right", [("upperarm", 10), ("uparm", 9)]),
            ("left_elbow", "left", [("forearm", 10), ("lowerarm", 9), ("lowarm", 8)]),
            ("right_elbow", "right", [("forearm", 10), ("lowerarm", 9), ("lowarm", 8)]),
            ("left_wrist", "left", [("hand", 8)]),
            ("right_wrist", "right", [("hand", 8)]),
            ("left_hip", "left", [("thigh", 10), ("upleg", 9), ("upperleg", 9)]),
            ("right_hip", "right", [("thigh", 10), ("upleg", 9), ("upperleg", 9)]),
            ("left_knee", "left", [("shin", 10), ("calf", 10), ("lowerleg", 9), ("lowleg", 8)]),
            ("right_knee", "right", [("shin", 10), ("calf", 10), ("lowerleg", 9), ("lowleg", 8)]),
            ("left_ankle", "left", [("foot", 8)]),
            ("right_ankle", "right", [("foot", 8)]),
            ("nose", None, [("head", 8)]),
            ("spine_direction", None, [("spine", 7)]),
        ]

        side_indicators = {
            "left": ["left", "_l_", ".l.", ".l", "_l"],
            "right": ["right", "_r_", ".r.", ".r", "_r"],
        }

        mapped_count = 0
        used_bones = set()

        for landmark, side, keywords in search_config:
            best_bone = None
            best_score = 0

            for bone_name in bone_names:
                if bone_name in used_bones:
                    continue

                bn_normalized = bone_names_normalized[bone_name]
                bn_lower = bone_name.lower()
                score = 0
                keyword_matched = False

                for kw, specificity in keywords:
                    if kw in bn_normalized:
                        score = specificity
                        keyword_matched = True
                        break

                if not keyword_matched:
                    continue

                if side:
                    side_found = False
                    for si in side_indicators[side]:
                        if si in bn_lower:
                            score += 5
                            side_found = True
                            break

                    if not side_found:
                        if side == "left":
                            if bn_lower.endswith("l") or bn_lower.endswith(".l") or "_l" in bn_lower:
                                score += 4
                                side_found = True
                        elif side == "right":
                            if bn_lower.endswith("r") or bn_lower.endswith(".r") or "_r" in bn_lower:
                                score += 4
                                side_found = True

                    wrong_side = "right" if side == "left" else "left"
                    for si in side_indicators[wrong_side]:
                        if si in bn_lower:
                            score = -100
                            break

                    if not side_found and score > 0:
                        score -= 3

                if landmark in ("left_ankle", "right_ankle"):
                    if "index" in bn_normalized or "tip" in bn_normalized or "toe" in bn_normalized:
                        score -= 10
                if landmark in ("left_wrist", "right_wrist"):
                    if "finger" in bn_normalized or "thumb" in bn_normalized or "index" in bn_normalized:
                        score -= 10
                if landmark == "nose":
                    if "headtop" in bn_normalized or "headend" in bn_normalized:
                        score -= 5

                if score > best_score:
                    best_score = score
                    best_bone = bone_name

            if best_bone and best_score >= 7:
                item = props.bone_mapping.add()
                item.landmark_name = landmark
                item.bone_name = best_bone
                item.enabled = True
                used_bones.add(best_bone)
                mapped_count += 1

        return mapped_count


class VMOCAP_OT_export_mapping(Operator):
    """Export bone mapping to JSON file"""
    bl_idname = "vmocap.export_mapping"
    bl_label = "Export Mapping"
    filepath: StringProperty(subtype='FILE_PATH', default="bone_mapping.json")
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        props = context.scene.vmocap
        data = {}
        for item in props.bone_mapping:
            data[item.landmark_name] = {"bone": item.bone_name, "enabled": item.enabled}
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.report({'INFO'}, f"Exported mapping to: {self.filepath}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class VMOCAP_OT_import_mapping(Operator):
    """Import bone mapping from JSON file"""
    bl_idname = "vmocap.import_mapping"
    bl_label = "Import Mapping"
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        props = context.scene.vmocap
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}

        props.bone_mapping.clear()
        for landmark, config in data.items():
            item = props.bone_mapping.add()
            item.landmark_name = landmark
            if isinstance(config, dict):
                item.bone_name = config.get("bone", "")
                item.enabled = config.get("enabled", True)
            else:
                item.bone_name = str(config)
                item.enabled = True

        self.report({'INFO'}, f"Imported {len(data)} mappings from: {self.filepath}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class VMOCAP_OT_set_camera_preset(Operator):
    """Set camera angle from preset"""
    bl_idname = "vmocap.set_camera_preset"
    bl_label = "Set Camera Angle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.vmocap
        preset_angles = {
            'FRONT': 0.0,
            'BACK': math.pi,
            'LEFT': -math.pi / 2,
            'RIGHT': math.pi / 2,
            'FRONT_LEFT': -math.pi / 4,
            'FRONT_RIGHT': math.pi / 4,
        }
        if props.camera_preset != 'CUSTOM':
            props.camera_angle = preset_angles.get(props.camera_preset, 0.0)
        return {'FINISHED'}


# ─── MAIN PROCESSING OPERATOR ────────────────────────────────

class VMOCAP_OT_process_video(Operator):
    """Process video and apply motion capture data to the target rig"""
    bl_idname = "vmocap.process_video"
    bl_label = "Process Video"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.vmocap
        if props.is_processing:
            return False
        if not props.video_path:
            return False
        if props.mode in ('AUTO_BODY', 'AUTO_HOLISTIC', 'GUIDED'):
            return props.target_armature is not None
        if props.mode == 'AUTO_FACE':
            return len(props.mesh_targets) > 0 or props.target_armature is not None
        return False

    def execute(self, context):
        props = context.scene.vmocap
        missing = ensure_dependencies()
        if missing:
            self.report({'ERROR'}, f"Missing: {', '.join(missing)}. Install from the panel above.")
            return {'CANCELLED'}

        if props.mode in ('AUTO_BODY', 'AUTO_HOLISTIC'):
            if len(props.bone_mapping) == 0:
                self.report({'ERROR'},
                    "No bone mapping configured! Use 'Bone Mapping' panel to load a preset.")
                return {'CANCELLED'}

            armature = props.target_armature
            enabled_mappings = [(item.landmark_name, item.bone_name)
                                for item in props.bone_mapping
                                if item.enabled and item.bone_name and item.landmark_name]
            if not enabled_mappings:
                self.report({'ERROR'}, "All bone mappings are disabled or empty!")
                return {'CANCELLED'}

            valid = [bn for _, bn in enabled_mappings if armature.pose.bones.get(bn)]
            invalid = [bn for _, bn in enabled_mappings if not armature.pose.bones.get(bn)]

            if not valid:
                bone_list = ", ".join(invalid[:5])
                self.report({'ERROR'},
                    f"None of the mapped bones exist on armature! "
                    f"Checked: {bone_list}... Use a correct preset.")
                return {'CANCELLED'}

            if invalid:
                self.report({'WARNING'},
                    f"{len(invalid)} bone(s) not found: {', '.join(invalid[:3])}...")

        props.is_processing = True
        props.progress = 0.0
        props.status_message = "Starting..."

        try:
            result_msg = self._run_capture(context)
            self.report({'INFO'}, result_msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            props.is_processing = False
            props.status_message = f"ERROR: {e}"
            return {'CANCELLED'}

        props.is_processing = False
        props.progress = 1.0
        props.status_message = "Complete!"
        return {'FINISHED'}

    def _run_capture(self, context):
        """Main capture pipeline."""
        props = context.scene.vmocap

        # ─── RESOLVE VIDEO PATH ───
        video_path = bpy.path.abspath(props.video_path)
        if not os.path.isfile(video_path):
            video_path = props.video_path
        if not os.path.isfile(video_path):
            raise RuntimeError(f"Video file not found: {props.video_path}")

        # ─── OPEN VIDEO ───
        video = VideoProcessor(video_path)
        video.open()
        print(f"[VMoCap] Video: {video_path}")
        print(f"[VMoCap] Frames: {video.total_frames}, FPS: {video.fps:.1f}, "
              f"Resolution: {video.width}x{video.height}")

        camera_angle_deg = math.degrees(props.camera_angle)
        print(f"[VMoCap] Camera angle: {camera_angle_deg:.1f}")

        context.scene.render.fps = int(round(video.fps))

        # ─── DETERMINE DETECTION MODES ───
        detect_pose = props.mode in ('AUTO_BODY', 'AUTO_HOLISTIC')
        detect_face = props.mode in ('AUTO_FACE', 'AUTO_HOLISTIC')
        detect_hands = props.mode == 'AUTO_HOLISTIC'

        # ─── INITIALIZE DETECTOR ───
        def progress_cb(msg):
            props.status_message = msg

        detector = LandmarkDetector(
            detect_pose=detect_pose,
            detect_face=detect_face,
            detect_hands=detect_hands,
            min_detection_confidence=props.detection_confidence,
            min_tracking_confidence=props.tracking_confidence,
            progress_callback=progress_cb
        )
        props.status_message = "Loading AI models..."
        detector.initialize()
        print(f"[VMoCap] Detector initialized (API: {_MP_API_VERSION})")

        # ─── PREPARE TARGETS ───
        armature = props.target_armature
        mesh_objects = [mt.mesh_object for mt in props.mesh_targets
                        if mt.enabled and mt.mesh_object]

        pose_retargeter = None
        face_retargeter = None

        if detect_pose and armature:
            bone_map_dict = {}
            for item in props.bone_mapping:
                if item.enabled and item.bone_name and item.landmark_name:
                    bone_map_dict[item.landmark_name] = item.bone_name

            print(f"[VMoCap] Bone mapping ({len(bone_map_dict)} entries):")
            for lm, bn in bone_map_dict.items():
                exists = "+" if armature.pose.bones.get(bn) else "X"
                print(f"  [{exists}] {lm} -> {bn}")

            if armature.animation_data is None:
                armature.animation_data_create()
            action_name = f"VMoCap_{os.path.basename(video_path).split('.')[0]}"
            action = bpy.data.actions.new(name=action_name)
            armature.animation_data.action = action
            print(f"[VMoCap] Created action: '{action.name}'")

            pose_retargeter = PoseRetargeter(armature, bone_map_dict, props.scale_factor)

        if detect_face and mesh_objects:
            face_retargeter = FaceRetargeter(mesh_objects)
            available = face_retargeter.get_available_shapekeys()
            print(f"[VMoCap] Face targets: {len(mesh_objects)} mesh(es), "
                  f"{len(available)} shape keys available")

        # ─── PREPARE ARMATURE ───
        if armature:
            context.view_layer.objects.active = armature
            bpy.ops.object.mode_set(mode='POSE')
            for pb in armature.pose.bones:
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
                pb.location = Vector((0, 0, 0))
                pb.scale = Vector((1, 1, 1))

        # ─── PROCESS FRAMES ───
        total = video.total_frames
        frame_step = max(1, props.frame_step)
        fps_ms = 1000.0 / video.fps

        props.status_message = "Processing video frames..."
        frames_processed = 0
        frames_with_pose = 0
        frames_with_face = 0

        for frame_idx in range(0, total, frame_step):
            frame_bgr = video.read_frame(frame_idx)
            if frame_bgr is None:
                break

            timestamp_ms = int(frame_idx * fps_ms)
            results = detector.process_frame(frame_bgr, timestamp_ms)
            blender_frame = props.start_frame + (frame_idx // frame_step)
            frames_processed += 1

            # Apply pose
            if results["pose"] is not None and pose_retargeter:
                frames_with_pose += 1
                transformed = CoordinateTransformer.batch_transform(
                    results["pose"],
                    scale=props.scale_factor,
                    camera_angle_deg=camera_angle_deg
                )
                pose_retargeter.apply_pose_frame(transformed, blender_frame)

            # Apply face
            if face_retargeter:
                if results.get("face_blendshapes"):
                    frames_with_face += 1
                    face_retargeter.apply_blendshapes_direct(
                        results["face_blendshapes"], blender_frame,
                        is_first_frame=(frames_with_face == 1)
                    )
                elif results["face"] is not None:
                    frames_with_face += 1
                    if not face_retargeter.calibrated:
                        face_retargeter.calibrate(results["face"])
                    au_weights = face_retargeter.compute_action_units(results["face"])
                    face_retargeter.apply_action_units(au_weights, blender_frame)

            # Progress
            props.progress = (frame_idx + 1) / max(total, 1)
            if frame_idx % max(1, total // 20) == 0:
                props.status_message = f"Frame {frame_idx + 1}/{total}"

        # ─── CLEANUP ───
        detector.close()
        video.close()

        # ─── REPORT ───
        pose_kf = pose_retargeter.get_keyframe_count() if pose_retargeter else 0
        face_kf = face_retargeter.get_keyframe_count() if face_retargeter else 0

        print(f"[VMoCap] ========================================")
        print(f"[VMoCap] RESULTS:")
        print(f"[VMoCap]   Frames processed: {frames_processed}")
        print(f"[VMoCap]   Frames with pose: {frames_with_pose}")
        print(f"[VMoCap]   Frames with face: {frames_with_face}")
        print(f"[VMoCap]   Pose keyframes inserted: {pose_kf}")
        print(f"[VMoCap]   Face keyframes inserted: {face_kf}")
        if pose_retargeter and pose_retargeter.get_warnings():
            print(f"[VMoCap]   Warnings: {pose_retargeter.get_warnings()}")
        print(f"[VMoCap] ========================================")

        # Face diagnostics
        if face_retargeter:
            matched = face_retargeter.get_matched_report()
            unmatched = face_retargeter.get_unmatched()
            if matched:
                print(f"[VMoCap] Successfully matched {len(matched)} blendshapes:")
                for arkit, sk in list(matched.items())[:10]:
                    print(f"[VMoCap]   {arkit} -> {sk}")
                if len(matched) > 10:
                    print(f"[VMoCap]   ... and {len(matched) - 10} more")
            if unmatched:
                print(f"[VMoCap] WARNING: {len(unmatched)} blendshapes could NOT be matched:")
                for name in sorted(unmatched)[:15]:
                    print(f"[VMoCap]   {name}")

        # Validate
        if detect_pose and pose_kf == 0:
            if frames_with_pose == 0:
                raise RuntimeError(
                    "No pose detected in any frame! Make sure a person is clearly visible.")
            else:
                raise RuntimeError(
                    f"Pose detected in {frames_with_pose} frames but 0 keyframes inserted. "
                    f"Check that bone names in your mapping actually exist on the armature.")

        # ─── SMOOTHING ───
        if props.smoothing_type != 'NONE' and (pose_kf > 0 or face_kf > 0):
            props.status_message = "Smoothing animation..."
            filter_map = {
                'ONE_EURO': 'one_euro',
                'MOVING_AVG': 'moving_average',
                'BUTTERWORTH': 'butterworth',
            }
            smoothed = AnimationSmoother.smooth_all_targets(
                armature, mesh_objects,
                filter_type=filter_map[props.smoothing_type],
                strength=props.smoothing_strength
            )
            print(f"[VMoCap] Smoothed {smoothed} F-Curves")

        # ─── SET FRAME RANGE ───
        end_frame = props.start_frame + frames_processed - 1
        context.scene.frame_start = props.start_frame
        context.scene.frame_end = end_frame
        context.scene.frame_current = props.start_frame

        # ─── RESTORE STATE ───
        if armature:
            bpy.ops.object.mode_set(mode='OBJECT')

        return (
            f"Done! {pose_kf} pose + {face_kf} face keyframes "
            f"over {frames_processed} frames (range: {props.start_frame}-{end_frame})"
        )


# ─── GUIDED MODE OPERATORS ───────────────────────────────────

class VMOCAP_OT_guided_add_point(Operator):
    """Add a tracking point for guided mode"""
    bl_idname = "vmocap.guided_add_point"
    bl_label = "Add Tracking Point"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.vmocap
        item = props.tracked_points.add()
        item.name = f"Point_{len(props.tracked_points)}"
        return {'FINISHED'}


class VMOCAP_OT_guided_remove_point(Operator):
    """Remove a tracking point"""
    bl_idname = "vmocap.guided_remove_point"
    bl_label = "Remove Point"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()

    def execute(self, context):
        props = context.scene.vmocap
        if 0 <= self.index < len(props.tracked_points):
            props.tracked_points.remove(self.index)
        return {'FINISHED'}


class VMOCAP_OT_guided_process(Operator):
    """Track points through video and apply to rig bones"""
    bl_idname = "vmocap.guided_process"
    bl_label = "Track & Apply"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.vmocap
        return (props.video_path and props.target_armature and
                len(props.tracked_points) > 0 and not props.is_processing)

    def execute(self, context):
        props = context.scene.vmocap
        missing = ensure_dependencies()
        if missing:
            self.report({'ERROR'}, f"Missing: {', '.join(missing)}")
            return {'CANCELLED'}

        video_path = bpy.path.abspath(props.video_path)
        if not os.path.isfile(video_path):
            video_path = props.video_path
        if not os.path.isfile(video_path):
            self.report({'ERROR'}, f"Video not found: {props.video_path}")
            return {'CANCELLED'}

        has_targets = any(pt.target_bone for pt in props.tracked_points)
        if not has_targets:
            self.report({'ERROR'}, "No tracking points have a target bone assigned!")
            return {'CANCELLED'}

        video = VideoProcessor(video_path)
        video.open()

        tracker = PointTracker()
        for pt in props.tracked_points:
            if pt.target_bone:
                tracker.add_point(pt.name, pt.position_x, pt.position_y)

        if not tracker.initial_points:
            video.close()
            self.report({'ERROR'}, "No valid tracking points with positions!")
            return {'CANCELLED'}

        tracked = tracker.track_through_video(video)
        video.close()

        armature = props.target_armature
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        if armature.animation_data is None:
            armature.animation_data_create()
        action = bpy.data.actions.new(name="VMoCap_Guided")
        armature.animation_data.action = action

        kf_count = 0
        for pt in props.tracked_points:
            if not pt.target_bone or pt.name not in tracked:
                continue
            pose_bone = armature.pose.bones.get(pt.target_bone)
            if not pose_bone:
                print(f"[VMoCap] Guided: bone '{pt.target_bone}' not found")
                continue
            positions = tracked[pt.name]
            for i, (px, py) in enumerate(positions):
                frame = props.start_frame + i
                loc = Vector((
                    (px / video.width - 0.5) * props.scale_factor,
                    0,
                    -(py / video.height - 0.5) * props.scale_factor
                ))
                pose_bone.location = loc
                pose_bone.keyframe_insert(data_path="location", frame=frame)
                kf_count += 1

        bpy.ops.object.mode_set(mode='OBJECT')

        if tracked:
            max_len = max(len(v) for v in tracked.values())
            context.scene.frame_start = props.start_frame
            context.scene.frame_end = props.start_frame + max_len - 1

        self.report({'INFO'}, f"Guided tracking done: {kf_count} keyframes")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════
# SECTION 10: UI PANELS
# ═════════════════════════════════════════════════════════════════

class VMOCAP_PT_main_panel(Panel):
    """Video Motion Capture - Main Panel"""
    bl_label = "Video MoCap"
    bl_idname = "VMOCAP_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VMoCap"

    def draw(self, context):
        layout = self.layout
        props = context.scene.vmocap

        # ─── DEPENDENCY CHECK ───
        missing = ensure_dependencies()
        if missing:
            box = layout.box()
            box.label(text="Dependencies Required:", icon='ERROR')
            for pkg in missing:
                box.label(text=f"  {pkg}")
            box.operator("vmocap.install_dependencies", icon='IMPORT')
            box.separator()
            box.label(text="Restart Blender after installing.", icon='INFO')
            return

        # ─── API STATUS ───
        if _MP_API_VERSION == 'tasks':
            layout.label(text="MediaPipe Tasks API OK", icon='CHECKMARK')
        else:
            layout.label(text="MediaPipe Legacy API OK", icon='CHECKMARK')

        # ─── MODE ───
        layout.prop(props, "mode")
        layout.separator()

        # ─── VIDEO INPUT ───
        box = layout.box()
        box.label(text="Video Input", icon='FILE_MOVIE')
        box.prop(props, "video_path", text="")
        if props.video_path:
            resolved = bpy.path.abspath(props.video_path)
            if os.path.isfile(resolved):
                box.label(text="File exists", icon='CHECKMARK')
            else:
                box.label(text="File not found!", icon='ERROR')

        # ─── TARGET ───
        box = layout.box()
        box.label(text="Target", icon='ARMATURE_DATA')
        if props.mode == 'AUTO_FACE':
            box.prop(props, "target_armature", text="Armature (optional)")
        else:
            box.prop(props, "target_armature")
            if props.target_armature:
                bone_count = len(props.target_armature.data.bones)
                box.label(text=f"   {bone_count} bones in rig")

        # ─── CAMERA PERSPECTIVE ───
        box = layout.box()
        box.label(text="Camera Perspective", icon='CAMERA_DATA')
        row = box.row(align=True)
        row.prop(props, "camera_preset", text="")
        row.operator("vmocap.set_camera_preset", text="", icon='CHECKMARK')
        if props.camera_preset == 'CUSTOM':
            box.prop(props, "camera_angle")
        else:
            angle_deg = math.degrees(props.camera_angle)
            box.label(text=f"   Angle: {angle_deg:.0f} deg")
        box.label(text="   (0 = facing camera, 90 = right side)", icon='INFO')

        # ─── SETTINGS ───
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        col = box.column(align=True)
        col.prop(props, "start_frame")
        col.prop(props, "scale_factor")
        col.prop(props, "frame_step")
        col.separator()
        col.prop(props, "detection_confidence")
        col.prop(props, "tracking_confidence")

        # ─── SMOOTHING ───
        box = layout.box()
        box.label(text="Smoothing", icon='SMOOTHCURVE')
        box.prop(props, "smoothing_type")
        if props.smoothing_type != 'NONE':
            box.prop(props, "smoothing_strength")

        layout.separator()

        # ─── MODEL DOWNLOAD ───
        if _MP_API_VERSION == 'tasks':
            models_dir = get_models_directory()
            needed = []
            if props.mode in ('AUTO_BODY', 'AUTO_HOLISTIC'):
                if not (models_dir / "pose_landmarker.task").exists():
                    needed.append("pose_landmarker")
            if props.mode in ('AUTO_FACE', 'AUTO_HOLISTIC'):
                if not (models_dir / "face_landmarker.task").exists():
                    needed.append("face_landmarker")
            if needed:
                box = layout.box()
                box.label(text="Models needed:", icon='IMPORT')
                for m in needed:
                    box.label(text=f"  {m}")
                box.operator("vmocap.download_models")
                layout.separator()

        # ─── VALIDATION ───
        if props.mode in ('AUTO_BODY', 'AUTO_HOLISTIC'):
            if len(props.bone_mapping) == 0:
                box = layout.box()
                box.alert = True
                box.label(text="No bone mapping!", icon='ERROR')
                box.label(text="Open 'Bone Mapping' panel below")
                layout.separator()
            elif props.target_armature:
                enabled = [item for item in props.bone_mapping
                           if item.enabled and item.bone_name]
                valid = sum(1 for item in enabled
                            if props.target_armature.pose.bones.get(item.bone_name))
                if valid < len(enabled):
                    box = layout.box()
                    box.alert = True
                    box.label(text=f"{len(enabled) - valid}/{len(enabled)} "
                              f"bones not found!", icon='ERROR')
                    layout.separator()

        # ─── PROCESS BUTTON ───
        if props.is_processing:
            box = layout.box()
            box.label(text=props.status_message, icon='TIME')
            box.progress(factor=props.progress, type='BAR',
                         text=f"{int(props.progress * 100)}%")
        else:
            row = layout.row()
            row.scale_y = 1.8
            if props.mode == 'GUIDED':
                row.operator("vmocap.guided_process", icon='PLAY')
            else:
                row.operator("vmocap.process_video", icon='PLAY')


class VMOCAP_PT_mesh_targets_panel(Panel):
    """Mesh targets for face capture"""
    bl_label = "Mesh Targets (Face)"
    bl_idname = "VMOCAP_PT_mesh_targets_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VMoCap"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.scene.vmocap.mode in ('AUTO_FACE', 'AUTO_HOLISTIC')

    def draw(self, context):
        layout = self.layout
        props = context.scene.vmocap

        row = layout.row(align=True)
        row.operator("vmocap.add_mesh_target", icon='ADD', text="Add")
        row.operator("vmocap.auto_discover_meshes", icon='VIEWZOOM', text="Auto-Find")

        if not props.mesh_targets:
            layout.label(text="No meshes added. Click Add or Auto-Find.", icon='INFO')
        else:
            for i, mt in enumerate(props.mesh_targets):
                row = layout.row(align=True)
                row.prop(mt, "enabled", text="")
                row.prop(mt, "mesh_object", text="")
                if mt.mesh_object and mt.mesh_object.data.shape_keys:
                    n = len(mt.mesh_object.data.shape_keys.key_blocks) - 1
                    row.label(text=f"({n})")
                elif mt.mesh_object:
                    row.label(text="(0!)", icon='ERROR')
                op = row.operator("vmocap.remove_mesh_target", text="", icon='X')
                op.index = i


class VMOCAP_PT_mapping_panel(Panel):
    """Bone mapping panel"""
    bl_label = "Bone Mapping"
    bl_idname = "VMOCAP_PT_mapping_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VMoCap"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.scene.vmocap.mode in ('AUTO_BODY', 'AUTO_HOLISTIC', 'GUIDED')

    def draw(self, context):
        layout = self.layout
        props = context.scene.vmocap

        # ─── PRESET BUTTONS ───
        box = layout.box()
        box.label(text="Load Preset:", icon='PRESET')
        row = box.row(align=True)
        op = row.operator("vmocap.load_bone_mapping", text="Rigify")
        op.preset = 'RIGIFY'
        op = row.operator("vmocap.load_bone_mapping", text="Mixamo")
        op.preset = 'MIXAMO'
        row = box.row(align=True)
        op = row.operator("vmocap.load_bone_mapping", text="CC3/CC4")
        op.preset = 'CC3'
        op = row.operator("vmocap.load_bone_mapping", text="UE Mann.")
        op.preset = 'UE_MANNEQUIN'

        row = box.row(align=True)
        row.operator("vmocap.auto_map_bones", text="Auto-Detect", icon='VIEWZOOM')

        if props.target_armature:
            box.label(text=f"   Rig: {props.target_armature.name}", icon='ARMATURE_DATA')

        layout.separator()

        # ─── IMPORT/EXPORT ───
        row = layout.row(align=True)
        row.operator("vmocap.import_mapping", text="Import", icon='IMPORT')
        row.operator("vmocap.export_mapping", text="Export", icon='EXPORT')

        layout.separator()

        # ─── ADD/CLEAR ───
        row = layout.row(align=True)
        row.operator("vmocap.add_bone_mapping", text="Add Entry", icon='ADD')
        row.operator("vmocap.clear_bone_mapping", text="Clear All", icon='TRASH')

        layout.separator()

        # ─── MAPPING TABLE ───
        if not props.bone_mapping:
            layout.label(text="No mappings. Load a preset above.", icon='INFO')
        else:
            header = layout.row(align=True)
            header.label(text="On")
            header.label(text="Landmark")
            header.label(text="")
            header.label(text="Bone Name")
            header.label(text="")

            for i, item in enumerate(props.bone_mapping):
                row = layout.row(align=True)
                row.prop(item, "enabled", text="")
                row.prop(item, "landmark_name", text="")
                row.label(text="->")
                if props.target_armature and props.target_armature.data:
                    row.prop_search(item, "bone_name",
                                    props.target_armature.data, "bones", text="")
                else:
                    row.prop(item, "bone_name", text="")

                if props.target_armature and item.bone_name:
                    if props.target_armature.pose.bones.get(item.bone_name):
                        row.label(text="", icon='CHECKMARK')
                    else:
                        row.label(text="", icon='ERROR')
                else:
                    row.label(text="")
                op = row.operator("vmocap.remove_bone_mapping", text="", icon='X')
                op.index = i

        # ─── REFERENCE ───
        layout.separator()
        box = layout.box()
        box.label(text="Valid Landmark Names:", icon='HELP')
        col = box.column(align=True)
        col.scale_y = 0.75
        col.label(text="left_shoulder / right_shoulder  (upper arm)")
        col.label(text="left_elbow / right_elbow  (forearm)")
        col.label(text="left_wrist / right_wrist  (hand)")
        col.label(text="left_hip / right_hip  (thigh)")
        col.label(text="left_knee / right_knee  (shin/calf)")
        col.label(text="left_ankle / right_ankle  (foot)")
        col.label(text="nose  (head direction)")
        col.label(text="spine_direction  (torso/spine)")


class VMOCAP_PT_guided_panel(Panel):
    """Guided tracking panel"""
    bl_label = "Guided Tracking"
    bl_idname = "VMOCAP_PT_guided_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VMoCap"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.scene.vmocap.mode == 'GUIDED'

    def draw(self, context):
        layout = self.layout
        props = context.scene.vmocap

        layout.label(text="Place tracking points on the video frame", icon='INFO')
        layout.label(text="(pixel X, Y coordinates)")
        layout.separator()

        layout.operator("vmocap.guided_add_point", icon='ADD')
        layout.separator()

        if not props.tracked_points:
            layout.label(text="No points added yet.")
        else:
            for i, pt in enumerate(props.tracked_points):
                box = layout.box()
                row = box.row(align=True)
                row.prop(pt, "name", text="Name")
                op = row.operator("vmocap.guided_remove_point", text="", icon='X')
                op.index = i

                row = box.row(align=True)
                row.prop(pt, "position_x", text="Pixel X")
                row.prop(pt, "position_y", text="Pixel Y")

                if props.target_armature and props.target_armature.data:
                    box.prop_search(pt, "target_bone",
                                    props.target_armature.data, "bones",
                                    text="Target Bone")
                else:
                    box.prop(pt, "target_bone", text="Target Bone")


# ═════════════════════════════════════════════════════════════════
# SECTION 11: REGISTRATION
# ═════════════════════════════════════════════════════════════════

classes = [
    VMOCAP_BoneMapItem,
    VMOCAP_TrackedPointItem,
    VMOCAP_MeshTargetItem,
    VMOCAP_Properties,
    VMOCAP_OT_install_dependencies,
    VMOCAP_OT_download_models,
    VMOCAP_OT_add_mesh_target,
    VMOCAP_OT_remove_mesh_target,
    VMOCAP_OT_auto_discover_meshes,
    VMOCAP_OT_load_bone_mapping,
    VMOCAP_OT_add_bone_mapping,
    VMOCAP_OT_remove_bone_mapping,
    VMOCAP_OT_clear_bone_mapping,
    VMOCAP_OT_auto_map_bones,
    VMOCAP_OT_export_mapping,
    VMOCAP_OT_import_mapping,
    VMOCAP_OT_set_camera_preset,
    VMOCAP_OT_process_video,
    VMOCAP_OT_guided_add_point,
    VMOCAP_OT_guided_remove_point,
    VMOCAP_OT_guided_process,
    VMOCAP_PT_main_panel,
    VMOCAP_PT_mesh_targets_panel,
    VMOCAP_PT_mapping_panel,
    VMOCAP_PT_guided_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vmocap = PointerProperty(type=VMOCAP_Properties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.vmocap


if __name__ == "__main__":
    register()