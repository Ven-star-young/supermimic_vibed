"""Unitree G1 (29 DoF) mjlab entity, following BeyondMimic's actuator model.

The URDF is the one the LAFAN1 motions were retargeted to (g1_29dof_rev_1_0). Joint drives are
implicit PD with a 10 Hz natural frequency and damping ratio 2 on the reflected motor armature, and
the action scale is 0.25 * effort_limit / stiffness, as in BeyondMimic.
"""

from __future__ import annotations


import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .. import DATA_DIR
URDF_PATH = DATA_DIR / "g1_description" / "g1_29dof_rev_1_0.urdf"
MJCF_PATH = DATA_DIR / "g1_description" / "g1_29dof.xml"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _kp(arm):
    return arm * NATURAL_FREQ**2


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

# Per-joint (effort limit, velocity limit, armature) by motor type.
_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}


def motor_of(joint_name: str) -> tuple[float, float, float]:
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return JOINT_MOTORS[key]


def action_scale(joint_name: str) -> float:
    effort, _, arm = motor_of(joint_name)
    return 0.25 * effort / _kp(arm)


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def default_joint_pos_list() -> list[float]:
    import re

    out = []
    for name in G1_JOINT_NAMES:
        val = 0.0
        for pat, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pat, name):
                val = v
        out.append(val)
    return out


def _actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    acts = []
    for name in G1_JOINT_NAMES:
        effort, _, arm = motor_of(name)
        acts.append(BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            effort_limit=effort,
            stiffness=_kp(arm),
            damping=_kd(arm),
            armature=arm,
        ))
    return tuple(acts)


# MuJoCo's built-in position actuators have torque limits but no hard joint-speed
# limit. Keep the manufacturer's speed values available as metadata; do not clip
# simulated velocities, which would modify the equations of motion.
G1_VELOCITY_LIMITS = {name: motor_of(name)[1] for name in G1_JOINT_NAMES}


def make_g1_spec() -> mujoco.MjSpec:
    """Load the original retargeting robot directly as MJCF, without USD.

    Use the same source for simulation and differentiable FK. Never substitute a
    bundled G1 asset: a different revision can change motion-reference geometry.
    """
    if not MJCF_PATH.is_file():
        raise FileNotFoundError(
            f"Missing G1 MJCF: {MJCF_PATH}. Run bash tools/download_data.sh "
            "(or python tools/urdf_to_mjcf.py if the original URDF and meshes are present)."
        )
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    # The EntityCfg below owns all drives, including their armature and force caps.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for geom in spec.geoms:
        if geom.contype != 0 or geom.conaffinity != 0:
            # Preserve the previous environment's disabled self-collisions while
            # retaining contact with the ground (ground contype=1).
            geom.contype = 0
            geom.conaffinity = 1
            geom.friction = [1.0, 0.005, 0.0001]
            geom.condim = 3
    return spec


def make_g1_robot_cfg() -> EntityCfg:
    """Return an independent config with all 29 joints position-controlled."""
    return EntityCfg(
        spec_fn=make_g1_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76), joint_pos=DEFAULT_JOINT_POS.copy(), joint_vel={".*": 0.0}
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=_actuators(), soft_joint_pos_limit_factor=0.9,
        ),
    )
