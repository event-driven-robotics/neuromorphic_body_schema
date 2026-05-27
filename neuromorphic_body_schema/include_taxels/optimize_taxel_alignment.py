"""Optimize taxel patch placement against MuJoCo collision meshes.

The optimizer refines per-part taxel placements that are initially defined by
the manual transform chain in include_skin_to_mujoco_model.py. For each body
part, the pipeline:

1. Loads and filters tactile taxel coordinates from the positions file.
2. Applies the fixed, hand-tuned baseline transform chain for that part.
3. Evaluates multiple seed hypotheses around zero and any previously saved
   optimized solution.
4. Runs a local L-BFGS-B optimization in quaternion-plus-translation space.
5. Optionally performs a tighter local refinement and final polish pass.
6. Stores the best result only if it improves the previously saved solution.

This module is deliberately configuration-driven: per-part behavior is defined
in PartConfig rather than in part-specific control-flow branches.

Run:
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py
        
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --parts r_upper_arm torso
        
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --polish
    
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --polish --parts r_upper_arm torso
        
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py --preview-footprint --parts l_foot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import mujoco
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from shapely.geometry import GeometryCollection, MultiPoint, Point, Polygon
from shapely.ops import unary_union

from include_skin_to_mujoco_model import (
    read_calibration_data,
    read_taxel2repr_data,
    rebase_coordinate_system,
    rotate_position,
    validate_taxel_data,
)

# MuJoCo C-extension symbols are resolved dynamically for editor type-checking stability.
MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_forward = getattr(mujoco, "mj_forward")
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mjtGeom = getattr(mujoco, "mjtGeom")


Vector3: TypeAlias = tuple[float, float, float]
QuaternionWXYZ: TypeAlias = tuple[float, float, float, float]
ManualStep: TypeAlias = tuple[Vector3, Vector3]
SeedCandidate: TypeAlias = tuple[str, np.ndarray, np.ndarray]
PreviousSeed: TypeAlias = dict[str, np.ndarray]
PreviousSeedMap: TypeAlias = dict[str, PreviousSeed]
JsonDict: TypeAlias = dict[str, Any]

# PartConfig quick guide:
# - Base seed set: delta_angle_seed_candidates (+ implicit zero translation seed).
# - Previous-result seeding: include_previous_seed plus optional previous_*_jitter_candidates.
# - Previous-result dense grid: previous_angle_grid_values_deg + previous_offset_grid_values_m.
# - Local refine pass (optional): local_refine_angle_jitter_candidates with refine windows.


@dataclass(frozen=True)
class PartConfig:
    """Configuration for optimizing one anatomical taxel patch.

    Each instance fully defines the data source, target mesh, baseline manual
    transform chain, and optimization strategy for a single body part.
    """

    part_name: str
    position_file: str
    mesh_files: tuple[str, ...]
    rebase: bool
    include_to_model: bool = True
    optimizer_method: str = "L-BFGS-B"
    manual_steps: tuple[ManualStep, ...] = ()
    delta_angle_seed_candidates: tuple[Vector3, ...] = ((0.0, 0.0, 0.0),)
    delta_angle_bounds_deg: Vector3 = (10.0, 10.0, 10.0)
    delta_translation_bounds_m: Vector3 = (0.02, 0.02, 0.02)
    include_previous_seed: bool = True
    previous_angle_jitter_candidates: tuple[Vector3, ...] = ()
    previous_offset_jitter_candidates: tuple[Vector3, ...] = ()
    previous_angle_grid_values_deg: tuple[tuple[float, ...],
                                          tuple[float, ...], tuple[float, ...]] | None = None
    previous_offset_grid_values_m: tuple[tuple[float, ...],
                                         tuple[float, ...], tuple[float, ...]] | None = None
    local_refine_angle_jitter_candidates: tuple[Vector3, ...] = ()
    local_refine_angle_window_deg: Vector3 = (6.0, 6.0, 6.0)
    local_refine_translation_window_m: Vector3 = (0.002, 0.002, 0.002)
    distance_focus_to_initial_patch: bool = False
    distance_focus_radius_m: float = 0.06
    distance_focus_min_samples: int = 3000
    inside_penalty_weight: float = 0.0


ROOT = Path(__file__).resolve().parents[2]
POSITIONS_DIR = ROOT / "neuromorphic_body_schema" / "include_taxels" / "positions"
MESH_DIR = ROOT / "neuromorphic_body_schema" / "meshes_improved"
MODEL_XML = ROOT / "neuromorphic_body_schema" / "models" / "icub_v2_full_body_improved.xml"
REPORT_JSON = ROOT / "neuromorphic_body_schema" / \
    "include_taxels" / "taxel_alignment_optimization_report.json"
REPORT_TXT = ROOT / "neuromorphic_body_schema" / \
    "include_taxels" / "taxel_alignment_optimization_report.txt"
LOCAL_OPTIMIZER_METHOD = "L-BFGS-B"
SOLE_2D_OPTIMIZER_METHOD = "sole-2d"
SOLE_2D_SOLVER = "Powell"
SEARCH_SPACE_SCALE = 0.5
INTRUSION_PENALTY_SCALE = 4.0
INTRUSION_FRACTION_EQUIV_DEPTH_M = 0.002
INTRUSION_QUADRATIC_SCALE_M = 0.002

GROUP_TORSO = 1
GROUP_RIGHT_ARM_HAND = 2
GROUP_LEFT_ARM_HAND = 3
GROUP_RIGHT_LEG = 4
GROUP_LEFT_LEG = 5

PART_MODEL_BODY: dict[str, str] = {
    "r_upper_leg": "r_upper_leg",
    "r_lower_leg": "r_lower_leg",
    "l_upper_leg": "l_upper_leg",
    "l_lower_leg": "l_lower_leg",
    "r_upper_arm": "r_shoulder_3",
    "r_forearm": "r_forearm",
    "l_upper_arm": "l_shoulder_3",
    "l_forearm": "l_forearm",
    "torso": "chest",
    "r_palm": "r_hand",
    "l_palm": "l_hand",
    "r_foot": "r_ankle_2",
    "l_foot": "l_ankle_2",
}

PART_SENSOR_GROUP: dict[str, int] = {
    "r_upper_leg": GROUP_RIGHT_LEG,
    "r_lower_leg": GROUP_RIGHT_LEG,
    "r_foot": GROUP_RIGHT_LEG,
    "l_upper_leg": GROUP_LEFT_LEG,
    "l_lower_leg": GROUP_LEFT_LEG,
    "l_foot": GROUP_LEFT_LEG,
    "torso": GROUP_TORSO,
    "r_upper_arm": GROUP_RIGHT_ARM_HAND,
    "r_forearm": GROUP_RIGHT_ARM_HAND,
    "r_palm": GROUP_RIGHT_ARM_HAND,
    "l_upper_arm": GROUP_LEFT_ARM_HAND,
    "l_forearm": GROUP_LEFT_ARM_HAND,
    "l_palm": GROUP_LEFT_ARM_HAND,
}


def _scaled_angle_bounds(config: PartConfig) -> np.ndarray:
    """Return per-axis rotation bounds scaled for stricter exploration."""

    return np.array(config.delta_angle_bounds_deg, dtype=float) * float(SEARCH_SPACE_SCALE)


def _scaled_translation_bounds(config: PartConfig) -> np.ndarray:
    """Return per-axis translation bounds scaled for stricter exploration."""

    return np.array(config.delta_translation_bounds_m, dtype=float) * float(SEARCH_SPACE_SCALE)

BODY_PARTS_PROPERTIES = {
    'cylindrical': [
        "r_upper_leg",
        "l_upper_leg",
        "r_upper_arm",
        "l_upper_arm",
        "r_forearm",
        "l_forearm",
        ],
    'curved_surface': [
        "r_lower_leg",
        "l_lower_leg",
        "torso"],   
    "flat_surface": [
        "r_foot",
        "l_foot",
        "r_palm",
        "l_palm",
    ]
}


@dataclass(frozen=True)
class StrategyProfile:
    """Optimization behavior selected from BODY_PARTS_PROPERTIES."""

    name: str
    use_flat_surface_optimizer: bool = False
    allow_distance_focus: bool = False
    trim_quantile: float = 0.9
    inside_penalty_weight: float = 0.0
    rotation_reg_scale_deg: float = 10.0
    translation_reg_scale_m: float = 0.01
    translation_axis_weights: Vector3 = (1.0, 1.0, 1.0)
    angle_axis_weights: Vector3 = (1.0, 1.0, 1.0)
    angle_reg_scale_deg: float = 10.0
    foot_edge_margin_m: float = 0.0
    foot_edge_margin_weight: float = 0.0
    density_balance_enabled: bool = False
    density_balance_neighbors: int = 8
    clearance_target_m: float = 0.0
    clearance_mean_weight: float = 0.0
    clearance_penalty_weight: float = 0.0
    clearance_balance_weight: float = 0.0
    min_clearance_m: float = 0.0
    min_clearance_penalty_weight: float = 0.0
    tilt_plane_penalty_weight: float = 0.0


BODY_PROPERTY_STRATEGIES: dict[str, StrategyProfile] = {
    "cylindrical": StrategyProfile(
        name="standard_cylindrical",
        use_flat_surface_optimizer=False,
        allow_distance_focus=False,
        trim_quantile=0.9,
        inside_penalty_weight=20.0,
        rotation_reg_scale_deg=10.0,
        translation_reg_scale_m=0.01,
        translation_axis_weights=(1.0, 1.0, 1.0),
        angle_axis_weights=(1.0, 1.0, 1.0),
        angle_reg_scale_deg=10.0,
    ),
    "curved_surface": StrategyProfile(
        name="curved_surface_balanced",
        use_flat_surface_optimizer=False,
        allow_distance_focus=False,
        trim_quantile=0.85,
        inside_penalty_weight=60.0,
        rotation_reg_scale_deg=10.0,
        translation_reg_scale_m=0.01,
        translation_axis_weights=(1.0, 1.0, 2.5),
        angle_axis_weights=(1.0, 1.0, 1.0),
        angle_reg_scale_deg=10.0,
    ),
    "flat_surface": StrategyProfile(
        name="flat_surface_planar",
        use_flat_surface_optimizer=True,
        allow_distance_focus=False,
        trim_quantile=0.9,
        inside_penalty_weight=30.0,
        rotation_reg_scale_deg=8.0,
        translation_reg_scale_m=0.01,
        translation_axis_weights=(1.0, 1.0, 1.0),
        angle_axis_weights=(1.0, 1.0, 1.0),
        angle_reg_scale_deg=8.0,
        foot_edge_margin_m=0.0015,
        foot_edge_margin_weight=25.0,
    ),
}

# Per-part strategy adjustments for known asymmetries.
PART_STRATEGY_OVERRIDES: dict[str, dict[str, Any]] = {
    "r_forearm": {
        "name": "cylindrical_asymmetric_patch",
        "angle_axis_weights": (3.0, 0.5, 3.0),
        "translation_axis_weights": (2.5, 1.0, 1.0),
        "angle_reg_scale_deg": 7.0,
    },
    "l_lower_leg": {
        "name": "curved_surface_z_stabilized",
        "translation_axis_weights": (1.0, 1.0, 3.5),
        "inside_penalty_weight": 80.0,
    },
    "l_palm": {
        "name": "flat_surface_yaw_tight",
        "angle_axis_weights": (1.0, 1.8, 1.0),
        "angle_reg_scale_deg": 6.0,
    },
    "l_foot": {
        "name": "flat_surface_edge_guarded",
        "foot_edge_margin_m": 0.0025,
        "foot_edge_margin_weight": 45.0,
    },
    "r_foot": {
        "name": "flat_surface_edge_guarded",
        "foot_edge_margin_m": 0.0025,
        "foot_edge_margin_weight": 45.0,
    },
}


def _build_part_to_body_property_map() -> dict[str, str]:
    """Build and validate a part->property lookup from BODY_PARTS_PROPERTIES."""

    mapping: dict[str, str] = {}
    for body_property, part_names in BODY_PARTS_PROPERTIES.items():
        if body_property not in BODY_PROPERTY_STRATEGIES:
            raise ValueError(
                f"No strategy profile configured for body property: {body_property}")
        for part_name in part_names:
            if part_name in mapping:
                raise ValueError(
                    f"Part {part_name} appears in multiple body properties")
            mapping[part_name] = body_property
    return mapping


PART_TO_BODY_PROPERTY: dict[str, str] = _build_part_to_body_property_map()

# Contact-side convention for palm patches in the hand local frame:
# y < 0 is palm surface, y > 0 is dorsal/back-of-hand.
PALM_CONTACT_SIDE_OVERRIDES: dict[str, str] = {
    "r_palm": "min_y",
    "l_palm": "min_y",
}


def validate_body_property_coverage(part_names: set[str]) -> None:
    """Ensure BODY_PARTS_PROPERTIES covers exactly the configured optimization parts."""

    missing = sorted(part_names.difference(PART_TO_BODY_PROPERTY.keys()))
    extra = sorted(set(PART_TO_BODY_PROPERTY.keys()).difference(part_names))
    if missing:
        raise ValueError(
            f"BODY_PARTS_PROPERTIES missing parts: {', '.join(missing)}")
    if extra:
        raise ValueError(
            f"BODY_PARTS_PROPERTIES contains unknown parts: {', '.join(extra)}")


def resolve_strategy_for_part(part_name: str) -> tuple[str, StrategyProfile]:
    """Resolve body property and effective strategy profile for one part."""

    if part_name not in PART_TO_BODY_PROPERTY:
        raise KeyError(
            f"Part {part_name} is not present in BODY_PARTS_PROPERTIES")

    body_property = PART_TO_BODY_PROPERTY[part_name]
    base_profile = BODY_PROPERTY_STRATEGIES[body_property]
    override = PART_STRATEGY_OVERRIDES.get(part_name)
    if override:
        return body_property, replace(base_profile, **override)
    return body_property, base_profile

def normalize_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion in canonical wxyz form."""

    quat = np.array(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat = quat / norm
    # Canonical sign for stable reporting/comparisons.
    if quat[0] < 0.0:
        quat = -quat
    return quat


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a quaternion from wxyz to scipy's xyzw convention."""

    q = normalize_quaternion_wxyz(quat_wxyz)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert a quaternion from scipy's xyzw convention to wxyz."""

    q = np.array(quat_xyzw, dtype=float)
    return normalize_quaternion_wxyz(np.array([q[3], q[0], q[1], q[2]], dtype=float))


def euler_xyz_deg_to_quat_wxyz(angles_deg: np.ndarray) -> np.ndarray:
    """Convert xyz Euler angles in degrees to a normalized wxyz quaternion."""

    rot = Rotation.from_euler("xyz", np.array(
        angles_deg, dtype=float), degrees=True)
    return quat_xyzw_to_wxyz(rot.as_quat())


def quat_wxyz_to_euler_xyz_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a normalized wxyz quaternion to xyz Euler angles in degrees."""

    quat_xyzw = quat_wxyz_to_xyzw(quat_wxyz)
    rot = Rotation.from_quat(quat_xyzw)
    return np.array(rot.as_euler("xyz", degrees=True), dtype=float)


def quaternion_geodesic_distance_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Return the shortest angular distance between two rotations in degrees."""

    q1 = normalize_quaternion_wxyz(q1_wxyz)
    q2 = normalize_quaternion_wxyz(q2_wxyz)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def apply_quaternion_transform(
    pos: np.ndarray,
    offsets: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Apply a translation followed by a quaternion rotation to one point."""

    rot = Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
    pos_shifted = np.array(pos, dtype=float) + np.array(offsets, dtype=float)
    return np.array(rot.apply(pos_shifted), dtype=float)

def build_bottom_contact_footprint_xz(mesh: trimesh.Trimesh, bottom_slice_height_m: float = 0.002) -> Any:
    """Return a strict XZ footprint for the sole contact patch of a mesh."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    min_y = float(np.min(vertices[:, 1]))
    slice_limit = min_y + bottom_slice_height_m

    bottom_face_mask = np.all(vertices[faces][:, :, 1] <= slice_limit, axis=1)
    bottom_faces = faces[bottom_face_mask]
    if len(bottom_faces) == 0:
        return MultiPoint(vertices[:, [0, 2]]).convex_hull

    projected_triangles: list[Polygon] = []
    for face in bottom_faces:
        polygon = Polygon(vertices[face][:, [0, 2]])
        if polygon.is_valid and polygon.area > 0.0:
            projected_triangles.append(polygon)

    if not projected_triangles:
        return MultiPoint(vertices[bottom_faces.reshape(-1), :][:, [0, 2]]).convex_hull

    footprint = unary_union(projected_triangles)
    if isinstance(footprint, GeometryCollection) and len(footprint.geoms) == 0:
        return MultiPoint(vertices[bottom_faces.reshape(-1), :][:, [0, 2]]).convex_hull
    return footprint


def build_point_hull_xz(points: np.ndarray) -> Any:
    """Return the XZ convex hull of a point cloud."""

    return MultiPoint(np.asarray(points, dtype=float)[:, [0, 2]]).convex_hull


def sole_surface_y(mesh: trimesh.Trimesh) -> float:
    """Return the Y level of the lowest sole surface."""

    return float(np.min(np.asarray(mesh.vertices, dtype=float)[:, 1]))


def _foot_boundary_distance_mean(hull: Any, footprint: Any) -> float:
    """Return the mean distance from taxel-hull vertices to the sole boundary."""

    if hull.is_empty:
        return 0.0
    if hull.geom_type == "Polygon":
        coords = np.asarray(hull.exterior.coords[:-1], dtype=float)
    elif hull.geom_type == "LineString":
        coords = np.asarray(hull.coords, dtype=float)
    else:
        coords = np.asarray([hull.coords[0]], dtype=float)
    if len(coords) == 0:
        return 0.0
    distances = [float(footprint.boundary.distance(Point(x, z)))
                 for x, z in coords]
    return float(np.mean(distances))


def _foot_edge_margin_penalty(hull: Any, footprint: Any, margin_m: float) -> float:
    """Penalize foot hull vertices that sit too close to the footprint edge."""

    if margin_m <= 0.0 or hull.is_empty:
        return 0.0
    if hull.geom_type == "Polygon":
        coords = np.asarray(hull.exterior.coords[:-1], dtype=float)
    elif hull.geom_type == "LineString":
        coords = np.asarray(hull.coords, dtype=float)
    else:
        coords = np.asarray([hull.coords[0]], dtype=float)
    if len(coords) == 0:
        return 0.0
    distances = np.array([float(footprint.boundary.distance(Point(x, z)))
                          for x, z in coords], dtype=float)
    shortfall = np.maximum(0.0, float(margin_m) - distances)
    return float(np.mean(shortfall ** 2))


def _foot_point_outside_penalty_xz(points: np.ndarray, footprint: Any) -> float:
    """Penalize individual taxels that fall outside the sole footprint."""

    if len(points) == 0:
        return 0.0
    penalties: list[float] = []
    for x, z in np.asarray(points, dtype=float)[:, [0, 2]]:
        point = Point(float(x), float(z))
        if footprint.covers(point):
            penalties.append(0.0)
        else:
            penalties.append(float(footprint.distance(point)) ** 2)
    return float(np.mean(penalties))


def _foot_point_edge_margin_penalty_xz(points: np.ndarray, footprint: Any, margin_m: float) -> float:
    """Penalize taxels that sit too close to the sole boundary."""

    if margin_m <= 0.0 or len(points) == 0:
        return 0.0
    penalties: list[float] = []
    for x, z in np.asarray(points, dtype=float)[:, [0, 2]]:
        distance_to_boundary = float(footprint.boundary.distance(Point(float(x), float(z))))
        shortfall = max(0.0, float(margin_m) - distance_to_boundary)
        penalties.append(shortfall ** 2)
    return float(np.mean(penalties))


def _planar_boundary_distances_xz(points: np.ndarray, footprint: Any) -> np.ndarray:
    """Return per-point distance to the XZ footprint boundary."""

    distances: list[float] = []
    for x, z in np.asarray(points, dtype=float)[:, [0, 2]]:
        distances.append(
            float(footprint.boundary.distance(Point(float(x), float(z))))
        )
    return np.array(distances, dtype=float)


def _inside_fraction_xz(points: np.ndarray, footprint: Any) -> float:
    """Return the fraction of points that lie inside the XZ footprint."""

    inside = 0
    for x, z in np.asarray(points, dtype=float)[:, [0, 2]]:
        if footprint.covers(Point(float(x), float(z))):
            inside += 1
    return float(inside / max(len(points), 1))


def _build_contact_footprint_xz(mesh: trimesh.Trimesh, use_max_y: bool, slice_height_m: float = 0.002) -> Any:
    """Return an XZ footprint for either the min-Y or max-Y contact slice."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    surface_y = float(np.max(vertices[:, 1]) if use_max_y else np.min(vertices[:, 1]))
    if use_max_y:
        face_mask = np.all(vertices[faces][:, :, 1] >= (surface_y - slice_height_m), axis=1)
    else:
        face_mask = np.all(vertices[faces][:, :, 1] <= (surface_y + slice_height_m), axis=1)
    contact_faces = faces[face_mask]
    if len(contact_faces) == 0:
        return MultiPoint(vertices[:, [0, 2]]).convex_hull

    projected_triangles: list[Polygon] = []
    for face in contact_faces:
        polygon = Polygon(vertices[face][:, [0, 2]])
        if polygon.is_valid and polygon.area > 0.0:
            projected_triangles.append(polygon)

    if not projected_triangles:
        return MultiPoint(vertices[contact_faces.reshape(-1), :][:, [0, 2]]).convex_hull

    footprint = unary_union(projected_triangles)
    if isinstance(footprint, GeometryCollection) and len(footprint.geoms) == 0:
        return MultiPoint(vertices[contact_faces.reshape(-1), :][:, [0, 2]]).convex_hull
    return footprint


def optimize_palm_surface_part(
    config: PartConfig,
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    previous_seed_map: PreviousSeedMap,
    strategy: StrategyProfile,
    body_property: str,
    polish: bool = False,
) -> JsonDict:
    """Optimize a palm patch in XZ with exact Y snap and in-plane rotation."""

    baseline_points = apply_part_transform(points, config.part_name)
    min_footprint = _build_contact_footprint_xz(mesh, use_max_y=False)
    max_footprint = _build_contact_footprint_xz(mesh, use_max_y=True)
    side_override = PALM_CONTACT_SIDE_OVERRIDES.get(config.part_name)
    if side_override is None:
        min_outside = _foot_point_outside_penalty_xz(baseline_points, min_footprint)
        max_outside = _foot_point_outside_penalty_xz(baseline_points, max_footprint)
        use_max_y = max_outside < min_outside
        contact_side_selection = "heuristic"
    elif side_override == "min_y":
        use_max_y = False
        contact_side_selection = "override:min_y"
    elif side_override == "max_y":
        use_max_y = True
        contact_side_selection = "override:max_y"
    else:
        raise ValueError(
            f"Invalid palm contact side override for {config.part_name}: {side_override}"
        )
    footprint = max_footprint if use_max_y else min_footprint

    vertices = np.asarray(mesh.vertices, dtype=float)
    y_target = float(np.max(vertices[:, 1]) if use_max_y else np.min(vertices[:, 1]))
    footprint_area = max(float(footprint.area), 1e-12)
    footprint_scale = max(float(np.sqrt(footprint_area)), 1e-6)
    y_snap_offset = float(y_target - np.mean(baseline_points[:, 1]))

    dist_fn = build_distance_fn(mesh)
    use_distance_focus = False
    inside_penalty_weight = 0.0

    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    angles0 = quat_wxyz_to_euler_xyz_deg(quat0)
    offsets0 = np.zeros(3, dtype=float)

    def transform_with_palm_params(y_angle_deg: float, tx: float, tz: float) -> np.ndarray:
        return apply_part_transform(
            points,
            config.part_name,
            delta_angles_deg=np.array([0.0, y_angle_deg, 0.0], dtype=float),
            delta_offsets_m=np.array([tx, y_snap_offset, tz], dtype=float),
        )

    def mean_distance(params: np.ndarray) -> float:
        transformed = transform_with_palm_params(
            float(params[0]), float(params[1]), float(params[2]))
        return float(np.mean(dist_fn(transformed)))

    def distance_stats(params: np.ndarray) -> dict[str, float]:
        transformed = transform_with_palm_params(
            float(params[0]), float(params[1]), float(params[2]))
        abs_distances = dist_fn(transformed)
        return {
            "mean_m": float(np.mean(abs_distances)),
            "median_m": float(np.median(abs_distances)),
            "p90_m": float(np.quantile(abs_distances, 0.9)),
            "inside_fraction": _inside_fraction_xz(transformed, footprint),
        }

    def objective(params: np.ndarray, angle_center: float, xz_center: np.ndarray) -> float:
        y_angle_deg = float(params[0])
        tx = float(params[1])
        tz = float(params[2])
        transformed = transform_with_palm_params(y_angle_deg, tx, tz)

        outside = _foot_point_outside_penalty_xz(transformed, footprint)
        if outside > 1e-12:
            return float(1e6 + 1e8 * outside)

        boundary_dist = _planar_boundary_distances_xz(transformed, footprint)
        edge_target = max(float(strategy.foot_edge_margin_m), 0.0015)
        edge_penalty = float(np.mean((boundary_dist - edge_target) ** 2)) / max(footprint_scale ** 2, 1e-12)
        uniformity_penalty = float(np.var(boundary_dist)) / max(footprint_scale ** 2, 1e-12)
        mean_distance_penalty = float(np.mean(dist_fn(transformed))) / 0.01

        angle_reg = ((y_angle_deg - angle_center) / 5.0) ** 2
        xz_reg = float(
            np.sum((((np.array([tx, tz], dtype=float) - xz_center) / 0.005) ** 2)))

        return float(
            120.0 * edge_penalty
            + 40.0 * uniformity_penalty
            + 10.0 * mean_distance_penalty
            + 0.1 * angle_reg
            + 0.1 * xz_reg
        )

    initial_stats = distance_stats(np.array([0.0, 0.0, 0.0], dtype=float))
    initial_mean = float(initial_stats["mean_m"])
    print(
        f"[{config.part_name}] initial mean distance={initial_mean:.6f} m | y_snap_offset={y_snap_offset:.6f} m | contact_side={'max_y' if use_max_y else 'min_y'} ({contact_side_selection})"
    )

    previous_seed = previous_seed_map.get(config.part_name)
    seed_candidates = build_seed_candidates(config, previous_seed, polish=polish)
    seed_results: list[tuple[float, np.ndarray, Any, float, np.ndarray, str]] = []

    translation_bounds = _scaled_translation_bounds(config)
    angle_bound_y = float(_scaled_angle_bounds(config)[1])
    translation_bounds_xz = np.array([
        float(translation_bounds[0]),
        float(translation_bounds[2]),
    ], dtype=float)

    for seed_idx, (seed_label, seed_angles, seed_offsets) in enumerate(seed_candidates):
        seed_angle_y = float(seed_angles[1])
        seed_xz = np.array([seed_offsets[0], seed_offsets[2]], dtype=float)
        x_seed = np.array([seed_angle_y, seed_xz[0], seed_xz[1]], dtype=float)

        print(
            f"\n[{config.part_name}] Trying seed {seed_idx+1}/{len(seed_candidates)}: {seed_label}")
        print(
            f"    angle_y_deg: {seed_angle_y:.6f} offsets_xyz: {[seed_xz[0], y_snap_offset, seed_xz[1]]}",
            flush=True,
        )

        bounds = [
            (seed_angle_y - angle_bound_y, seed_angle_y + angle_bound_y),
            (seed_xz[0] - translation_bounds_xz[0], seed_xz[0] + translation_bounds_xz[0]),
            (seed_xz[1] - translation_bounds_xz[1], seed_xz[1] + translation_bounds_xz[1]),
        ]

        best_so_far = {"score": mean_distance(x_seed), "params": x_seed.copy()}

        def callback(xk: np.ndarray) -> None:
            md = mean_distance(xk)
            print(
                f"[{config.part_name}] seed={seed_idx+1} ({seed_label}) mean_distance={md:.6f} m",
                flush=True,
            )
            if md < best_so_far["score"]:
                best_so_far["score"] = md
                best_so_far["params"] = xk.copy()

        primary_maxiter = 800 if polish else 300
        primary_ftol = 1e-16 if polish else 1e-12
        result = minimize(
            lambda p: objective(p, seed_angle_y, seed_xz),
            x_seed,
            method=LOCAL_OPTIMIZER_METHOD,
            bounds=bounds,
            callback=callback,
            options={"maxiter": primary_maxiter, "ftol": primary_ftol, "gtol": 1e-12},
        )

        x_candidate = best_so_far["params"]
        score = best_so_far["score"]
        seed_results.append((score, x_candidate, result, seed_angle_y, seed_xz, seed_label))

    seed_results.sort(key=lambda item: item[0])
    _, x_final, result, selected_seed_angle_y, selected_seed_xz, selected_seed_label = seed_results[0]

    polish_refinement: dict[str, Any] | None = None
    if polish:
        polish_angle_window = min(angle_bound_y, 2.0)
        polish_translation_window_x = min(float(translation_bounds[0]), 0.003)
        polish_translation_window_z = min(float(translation_bounds[2]), 0.003)

        bounds_polish = [
            (float(x_final[0]) - polish_angle_window, float(x_final[0]) + polish_angle_window),
            (float(x_final[1]) - polish_translation_window_x, float(x_final[1]) + polish_translation_window_x),
            (float(x_final[2]) - polish_translation_window_z, float(x_final[2]) + polish_translation_window_z),
        ]

        polish_before = mean_distance(x_final)
        result_polish = minimize(
            lambda p: objective(p, float(x_final[0]), np.array([float(x_final[1]), float(x_final[2])], dtype=float)),
            x_final,
            method=LOCAL_OPTIMIZER_METHOD,
            bounds=bounds_polish,
            options={"maxiter": 1200, "ftol": 1e-20, "gtol": 1e-16},
        )
        x_polished = result_polish.x if result_polish.success else x_final
        if mean_distance(x_polished) < polish_before:
            x_final = x_polished
            result = result_polish

        polish_refinement = {
            "pre_polish_mean_distance_m": float(polish_before),
            "post_polish_mean_distance_m": float(mean_distance(x_final)),
            "angle_window_deg": [0.0, float(polish_angle_window), 0.0],
            "translation_window_m": [float(polish_translation_window_x), 0.0, float(polish_translation_window_z)],
            "rotation_space": "y_axis_only",
            "success": bool(result_polish.success),
        }

    final_angles = np.array([0.0, float(x_final[0]), 0.0], dtype=float)
    final_quat = euler_xyz_deg_to_quat_wxyz(final_angles)
    final_offsets = np.array([float(x_final[1]), y_snap_offset, float(x_final[2])], dtype=float)
    final_stats = distance_stats(x_final)
    final_mean = float(final_stats["mean_m"])

    return {
        "part": config.part_name,
        "model_body_name": PART_MODEL_BODY[config.part_name],
        "sensor_group": PART_SENSOR_GROUP[config.part_name],
        "position_file": config.position_file,
        "rebase": bool(config.rebase),
        "include_to_model": bool(config.include_to_model),
        "mesh_files": list(config.mesh_files),
        "manual_steps": [
            {"offsets_m": list(step[0]), "angles_deg": list(step[1])}
            for step in config.manual_steps
        ],
        "initial": {
            "delta_angles_deg": angles0.tolist(),
            "delta_quaternion_wxyz": quat0.tolist(),
            "delta_offsets_m": offsets0.tolist(),
            "mean_distance_m": initial_stats["mean_m"],
            "median_distance_m": initial_stats["median_m"],
            "p90_distance_m": initial_stats["p90_m"],
            "inside_fraction": initial_stats["inside_fraction"],
        },
        "optimized": {
            "delta_angles_deg": final_angles.tolist(),
            "delta_quaternion_wxyz": final_quat.tolist(),
            "delta_offsets_m": final_offsets.tolist(),
            "mean_distance_m": final_stats["mean_m"],
            "median_distance_m": final_stats["median_m"],
            "p90_distance_m": final_stats["p90_m"],
            "inside_fraction": final_stats["inside_fraction"],
        },
        "delta": {
            "angles_deg": (final_angles - angles0).tolist(),
            "quaternion_wxyz": final_quat.tolist(),
            "rotation_distance_deg": quaternion_geodesic_distance_deg(final_quat, quat0),
            "offsets_m": (final_offsets - offsets0).tolist(),
            "mean_distance_m": float(final_mean - initial_mean),
            "mean_distance_improvement_pct": float((initial_mean - final_mean) / max(initial_mean, 1e-12) * 100.0),
            "inside_fraction": float(final_stats["inside_fraction"] - initial_stats["inside_fraction"]),
        },
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "objective": float(result.fun),
            "optimizer_method": config.optimizer_method,
            "solver": LOCAL_OPTIMIZER_METHOD,
            "seed_count": len(seed_candidates),
            "selected_seed_label": selected_seed_label,
            "selected_seed_angles_deg": [0.0, float(selected_seed_angle_y), 0.0],
            "selected_seed_quaternion_wxyz": euler_xyz_deg_to_quat_wxyz(np.array([0.0, float(selected_seed_angle_y), 0.0], dtype=float)).tolist(),
            "selected_seed_offsets_m": [float(selected_seed_xz[0]), y_snap_offset, float(selected_seed_xz[1])],
            "angle_bounds_deg": [0.0, angle_bound_y, 0.0],
            "translation_bounds_m": [float(translation_bounds_xz[0]), 0.0, float(translation_bounds_xz[1])],
            "fine_refinement": None,
            "polish_mode": bool(polish),
            "polish_refinement": polish_refinement,
            "surface_y_m": y_target,
            "y_snap_offset_m": y_snap_offset,
            "footprint_area_m2": footprint_area,
            "contact_side": "max_y" if use_max_y else "min_y",
            "contact_side_selection": contact_side_selection,
            "body_property": body_property,
            "strategy_name": strategy.name,
            "inside_penalty_weight": inside_penalty_weight,
            "distance_focus_enabled": use_distance_focus,
        },
    }


def optimize_foot_sole_part(
    config: PartConfig,
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    previous_seed_map: PreviousSeedMap,
    strategy: StrategyProfile,
    body_property: str,
    polish: bool = False,
) -> JsonDict:
    """Optimize a flat foot sole in XY with exact Z snap and Z-rotation fitting."""

    if (
        config.optimizer_method != SOLE_2D_OPTIMIZER_METHOD
        and not strategy.use_flat_surface_optimizer
    ):
        raise ValueError(
            f"Unsupported foot optimizer method: {config.optimizer_method}")

    baseline_points = apply_part_transform(points, config.part_name)

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    z_target = float(np.max(vertices[:, 2]))
    slice_limit = z_target - 0.002
    sole_face_mask = np.all(vertices[faces][:, :, 2] >= slice_limit, axis=1)
    sole_faces = faces[sole_face_mask]
    if len(sole_faces) == 0:
        footprint = MultiPoint(vertices[:, [0, 1]]).convex_hull
    else:
        projected_triangles: list[Polygon] = []
        for face in sole_faces:
            polygon = Polygon(vertices[face][:, [0, 1]])
            if polygon.is_valid and polygon.area > 0.0:
                projected_triangles.append(polygon)
        if projected_triangles:
            footprint = unary_union(projected_triangles)
            if isinstance(footprint, GeometryCollection) and len(footprint.geoms) == 0:
                footprint = MultiPoint(vertices[sole_faces.reshape(-1), :][:, [0, 1]]).convex_hull
        else:
            footprint = MultiPoint(vertices[sole_faces.reshape(-1), :][:, [0, 1]]).convex_hull

    footprint_area = max(float(footprint.area), 1e-12)
    footprint_scale = max(float(np.sqrt(footprint_area)), 1e-6)
    dist_fn = build_distance_fn(mesh)
    use_distance_focus = False
    inside_penalty_weight = 0.0

    z_snap_offset = float(z_target - np.mean(baseline_points[:, 2]))

    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    angles0 = quat_wxyz_to_euler_xyz_deg(quat0)
    offsets0 = np.zeros(3, dtype=float)

    def transform_with_foot_params(z_angle_deg: float, tx: float, ty: float) -> np.ndarray:
        return apply_part_transform(
            points,
            config.part_name,
            delta_angles_deg=np.array([0.0, 0.0, z_angle_deg], dtype=float),
            delta_offsets_m=np.array([tx, ty, z_snap_offset], dtype=float),
        )

    def mean_distance(params: np.ndarray) -> float:
        transformed = transform_with_foot_params(
            float(params[0]), float(params[1]), float(params[2]))
        distances = dist_fn(transformed)
        return float(np.mean(distances))

    def outside_penalty_xy(points_world: np.ndarray) -> float:
        penalties: list[float] = []
        for x, y in np.asarray(points_world, dtype=float)[:, [0, 1]]:
            point = Point(float(x), float(y))
            if footprint.covers(point):
                penalties.append(0.0)
            else:
                penalties.append(float(footprint.distance(point)) ** 2)
        return float(np.mean(penalties)) if penalties else 0.0

    def boundary_distances_xy(points_world: np.ndarray) -> np.ndarray:
        boundary = footprint.boundary
        if boundary is None:
            return np.zeros(points_world.shape[0], dtype=float)
        distances: list[float] = []
        for x, y in np.asarray(points_world, dtype=float)[:, [0, 1]]:
            distances.append(
                float(boundary.distance(Point(float(x), float(y))))
            )
        return np.array(distances, dtype=float)

    def footprint_inside_fraction(points_world: np.ndarray) -> float:
        inside = 0
        for x, y in np.asarray(points_world, dtype=float)[:, [0, 1]]:
            if footprint.covers(Point(float(x), float(y))):
                inside += 1
        return float(inside / max(len(points_world), 1))

    def distance_stats(params: np.ndarray) -> dict[str, float]:
        transformed = transform_with_foot_params(
            float(params[0]), float(params[1]), float(params[2]))
        abs_distances = dist_fn(transformed)
        return {
            "mean_m": float(np.mean(abs_distances)),
            "median_m": float(np.median(abs_distances)),
            "p90_m": float(np.quantile(abs_distances, 0.9)),
            "inside_fraction": footprint_inside_fraction(transformed),
        }

    def objective(params: np.ndarray, angle_center: float, xy_center: np.ndarray) -> float:
        z_angle_deg = float(params[0])
        tx = float(params[1])
        ty = float(params[2])
        transformed = transform_with_foot_params(z_angle_deg, tx, ty)

        outside = outside_penalty_xy(transformed)
        if outside > 1e-12:
            # Priority 1: keep taxels inside footprint.
            return float(1e6 + 1e8 * outside)

        boundary_dist = boundary_distances_xy(transformed)
        edge_target = max(float(strategy.foot_edge_margin_m), 0.0015)
        edge_penalty = float(np.mean((boundary_dist - edge_target) ** 2)) / max(footprint_scale ** 2, 1e-12)
        uniformity_penalty = float(np.var(boundary_dist)) / max(footprint_scale ** 2, 1e-12)
        mean_distance_penalty = float(np.mean(dist_fn(transformed))) / 0.01

        angle_reg = ((z_angle_deg - angle_center) / 5.0) ** 2
        xy_reg = float(
            np.sum((((np.array([tx, ty], dtype=float) - xy_center) / 0.005) ** 2)))

        return float(
            120.0 * edge_penalty
            + 40.0 * uniformity_penalty
            + 10.0 * mean_distance_penalty
            + 0.1 * angle_reg
            + 0.1 * xy_reg
        )

    initial_stats = distance_stats(np.array([0.0, 0.0, 0.0], dtype=float))
    initial_mean = float(initial_stats["mean_m"])
    print(
        f"[{config.part_name}] initial mean distance={initial_mean:.6f} m | z_snap_offset={z_snap_offset:.6f} m"
    )

    previous_seed = previous_seed_map.get(config.part_name)
    seed_candidates = build_seed_candidates(
        config, previous_seed, polish=polish)
    seed_results: list[tuple[float, np.ndarray,
                             Any, float, np.ndarray, str]] = []

    translation_bounds = _scaled_translation_bounds(config)
    angle_bound_z = float(_scaled_angle_bounds(config)[2])
    translation_bounds_xy = np.array([
        float(translation_bounds[0]),
        float(translation_bounds[1]),
    ], dtype=float)

    for seed_idx, (seed_label, seed_angles, seed_offsets) in enumerate(seed_candidates):
        seed_angle_z = float(seed_angles[2])
        seed_xy = np.array([seed_offsets[0], seed_offsets[1]], dtype=float)
        x_seed = np.array([seed_angle_z, seed_xy[0], seed_xy[1]], dtype=float)

        print(
            f"\n[{config.part_name}] Trying seed {seed_idx+1}/{len(seed_candidates)}: {seed_label}")
        print(
            f"    angle_z_deg: {seed_angle_z:.6f} offsets_xyz: {[seed_xy[0], seed_xy[1], z_snap_offset]}",
            flush=True,
        )

        bounds = [
            (seed_angle_z - angle_bound_z, seed_angle_z + angle_bound_z),
            (seed_xy[0] - translation_bounds_xy[0],
             seed_xy[0] + translation_bounds_xy[0]),
            (seed_xy[1] - translation_bounds_xy[1],
             seed_xy[1] + translation_bounds_xy[1]),
        ]

        iteration = {"count": 0}
        best_so_far = {
            "score": mean_distance(x_seed),
            "params": x_seed.copy(),
        }

        def callback(xk: np.ndarray) -> None:
            iteration["count"] += 1
            md = mean_distance(xk)
            print(
                f"[{config.part_name}] seed={seed_idx+1} ({seed_label}) iter={iteration['count']:02d} mean_distance={md:.6f} m",
                flush=True,
            )
            if md < best_so_far["score"]:
                best_so_far["score"] = md
                best_so_far["params"] = xk.copy()

        primary_maxiter = 800 if polish else 300
        primary_ftol = 1e-16 if polish else 1e-12
        result = minimize(
            lambda p: objective(p, seed_angle_z, seed_xy),
            x_seed,
            method=LOCAL_OPTIMIZER_METHOD,
            bounds=bounds,
            callback=callback,
            options={"maxiter": primary_maxiter,
                     "ftol": primary_ftol, "gtol": 1e-12},
        )

        x_candidate = best_so_far["params"]
        score = best_so_far["score"]
        seed_results.append((score, x_candidate, result,
                            seed_angle_z, seed_xy, seed_label))

    seed_results.sort(key=lambda item: item[0])
    _, x_final, result, selected_seed_angle_z, selected_seed_xy, selected_seed_label = seed_results[
        0]

    polish_refinement: dict[str, Any] | None = None
    if polish:
        polish_angle_window = min(angle_bound_z, 2.0)
        polish_translation_window_x = min(float(translation_bounds[0]), 0.003)
        polish_translation_window_y = min(float(translation_bounds[1]), 0.003)

        bounds_polish = [
            (float(x_final[0]) - polish_angle_window,
             float(x_final[0]) + polish_angle_window),
            (float(x_final[1]) - polish_translation_window_x,
             float(x_final[1]) + polish_translation_window_x),
            (float(x_final[2]) - polish_translation_window_y,
             float(x_final[2]) + polish_translation_window_y),
        ]

        polish_before = mean_distance(x_final)
        result_polish = minimize(
            lambda p: objective(p, float(x_final[0]), np.array(
                [float(x_final[1]), float(x_final[2])], dtype=float)),
            x_final,
            method=LOCAL_OPTIMIZER_METHOD,
            bounds=bounds_polish,
            options={"maxiter": 1200, "ftol": 1e-20, "gtol": 1e-16},
        )
        x_polished = result_polish.x if result_polish.success else x_final
        polish_after = mean_distance(x_polished)
        if polish_after < polish_before:
            x_final = x_polished
            result = result_polish

        polish_refinement = {
            "pre_polish_mean_distance_m": float(polish_before),
            "post_polish_mean_distance_m": float(mean_distance(x_final)),
            "angle_window_deg": [0.0, 0.0, float(polish_angle_window)],
            "translation_window_m": [float(polish_translation_window_x), float(polish_translation_window_y), 0.0],
            "rotation_space": "z_axis_only",
            "success": bool(result_polish.success),
        }

    final_angles = np.array([0.0, 0.0, float(x_final[0])], dtype=float)
    final_quat = euler_xyz_deg_to_quat_wxyz(final_angles)
    final_offsets = np.array(
        [float(x_final[1]), float(x_final[2]), z_snap_offset], dtype=float)
    final_stats = distance_stats(x_final)
    final_mean = float(final_stats["mean_m"])

    return {
        "part": config.part_name,
        "model_body_name": PART_MODEL_BODY[config.part_name],
        "sensor_group": PART_SENSOR_GROUP[config.part_name],
        "position_file": config.position_file,
        "rebase": bool(config.rebase),
        "include_to_model": bool(config.include_to_model),
        "mesh_files": list(config.mesh_files),
        "manual_steps": [
            {"offsets_m": list(step[0]), "angles_deg": list(step[1])}
            for step in config.manual_steps
        ],
        "initial": {
            "delta_angles_deg": angles0.tolist(),
            "delta_quaternion_wxyz": quat0.tolist(),
            "delta_offsets_m": offsets0.tolist(),
            "mean_distance_m": initial_stats["mean_m"],
            "median_distance_m": initial_stats["median_m"],
            "p90_distance_m": initial_stats["p90_m"],
            "inside_fraction": initial_stats["inside_fraction"],
        },
        "optimized": {
            "delta_angles_deg": final_angles.tolist(),
            "delta_quaternion_wxyz": final_quat.tolist(),
            "delta_offsets_m": final_offsets.tolist(),
            "mean_distance_m": final_stats["mean_m"],
            "median_distance_m": final_stats["median_m"],
            "p90_distance_m": final_stats["p90_m"],
            "inside_fraction": final_stats["inside_fraction"],
        },
        "delta": {
            "angles_deg": (final_angles - angles0).tolist(),
            "quaternion_wxyz": final_quat.tolist(),
            "rotation_distance_deg": quaternion_geodesic_distance_deg(final_quat, quat0),
            "offsets_m": (final_offsets - offsets0).tolist(),
            "mean_distance_m": float(final_mean - initial_mean),
            "mean_distance_improvement_pct": float((initial_mean - final_mean) / max(initial_mean, 1e-12) * 100.0),
            "inside_fraction": float(final_stats["inside_fraction"] - initial_stats["inside_fraction"]),
        },
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "objective": float(result.fun),
            "optimizer_method": config.optimizer_method,
            "solver": LOCAL_OPTIMIZER_METHOD,
            "seed_count": len(seed_candidates),
            "selected_seed_label": selected_seed_label,
            "selected_seed_angles_deg": [0.0, 0.0, float(selected_seed_angle_z)],
            "selected_seed_quaternion_wxyz": euler_xyz_deg_to_quat_wxyz(np.array([0.0, 0.0, float(selected_seed_angle_z)], dtype=float)).tolist(),
            "selected_seed_offsets_m": [float(selected_seed_xy[0]), float(selected_seed_xy[1]), z_snap_offset],
            "angle_bounds_deg": [0.0, 0.0, angle_bound_z],
            "translation_bounds_m": [float(translation_bounds_xy[0]), float(translation_bounds_xy[1]), 0.0],
            "fine_refinement": None,
            "polish_mode": bool(polish),
            "polish_refinement": polish_refinement,
            "sole_surface_z_m": z_target,
            "z_snap_offset_m": z_snap_offset,
            "footprint_area_m2": footprint_area,
            "body_property": body_property,
            "strategy_name": strategy.name,
            "inside_penalty_weight": inside_penalty_weight,
            "distance_focus_enabled": use_distance_focus,
        },
    }

# manual_steps = (t_x, t_y, t_z), (r_x, r_y, r_z)
PARTS: tuple[PartConfig, ...] = (
    PartConfig(
        part_name="r_upper_leg",
        position_file="right_leg_upper.txt",
        mesh_files=("sim_sea_2-5_r_thigh.stl",),
        rebase=False,
        include_previous_seed=False,
        manual_steps=(
            # rotation
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_lower_leg",
        position_file="right_leg_lower.txt",
        mesh_files=("sim_sea_2-5_r_shank.stl",),
        rebase=False,
        manual_steps=(
            # rotation
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_upper_leg",
        position_file="left_leg_upper.txt",
        mesh_files=("sim_sea_2-5_l_thigh.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_lower_leg",
        position_file="left_leg_lower.txt",
        mesh_files=("sim_sea_2-5_l_shank.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_upper_arm",
        position_file="right_arm.txt",
        mesh_files=("sim_sea_2-5_r_elbow.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_forearm",
        position_file="right_forearm_V2.txt",
        mesh_files=("sim_sea_2-5_r_forearm.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_upper_arm",
        position_file="left_arm.txt",
        mesh_files=("sim_sea_2-5_l_elbow.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_forearm",
        position_file="left_forearm_V2.txt",
        mesh_files=("sim_sea_2-5_l_forearm.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="torso",
        position_file="torso.txt",
        mesh_files=("sim_ibbbbeba_chest.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
        delta_angle_bounds_deg=(10.0, 10.0, 10.0),
        delta_translation_bounds_m=(0.02, 0.02, 0.02),
        distance_focus_to_initial_patch=True,
        distance_focus_radius_m=0.07,
        distance_focus_min_samples=6000,
        inside_penalty_weight=30.0,
    ),
    PartConfig(
        part_name="r_palm",
        position_file="right_hand_V2_1.txt",
        mesh_files=("col_RightHandPalm.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_palm",
        position_file="left_hand_V2_1.txt",
        mesh_files=("col_LeftHandPalm.stl",),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_foot",
        position_file="left_foot.txt",
        mesh_files=("sim_sea_2-5_l_sole.stl",),
        rebase=False,
        include_to_model=False,
        optimizer_method=SOLE_2D_OPTIMIZER_METHOD,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_foot",
        position_file="right_foot.txt",
        mesh_files=("sim_sea_2-5_r_sole.stl",),
        rebase=False,
        include_to_model=False,
        optimizer_method=SOLE_2D_OPTIMIZER_METHOD,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
)

PARTS_BY_NAME: dict[str, PartConfig] = {cfg.part_name: cfg for cfg in PARTS}

_MJ_MODEL_CACHE: Any = None
_MJ_DATA_CACHE: Any = None
_MESH_TRANSFORM_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _load_mj_model_data() -> tuple[Any, Any]:
    """Lazily load and forward the reference model used for frame resolution."""

    global _MJ_MODEL_CACHE, _MJ_DATA_CACHE
    if _MJ_MODEL_CACHE is None or _MJ_DATA_CACHE is None:
        _MJ_MODEL_CACHE = MjModel.from_xml_path(str(MODEL_XML))
        _MJ_DATA_CACHE = MjData(_MJ_MODEL_CACHE)
        mj_forward(_MJ_MODEL_CACHE, _MJ_DATA_CACHE)
    return _MJ_MODEL_CACHE, _MJ_DATA_CACHE


def _is_descendant_or_self(model: Any, body_id: int, ancestor_id: int) -> bool:
    """Return True when body_id is identical to, or inside, ancestor_id subtree."""

    current = int(body_id)
    while current >= 0:
        if current == ancestor_id:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


def _quat_wxyz_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to normalized quaternion in wxyz order."""

    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
    )
    norm = np.linalg.norm(quat_wxyz)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat_wxyz / norm


def resolve_mesh_transform_from_model(part_name: str, mesh_file: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve a mesh transform in the part anchor-body frame from the MuJoCo model.

    Returns (pos_xyz, quat_wxyz) in the local frame of PART_MODEL_BODY[part_name].
    """

    cache_key = (part_name, mesh_file)
    if cache_key in _MESH_TRANSFORM_CACHE:
        pos_cached, quat_cached = _MESH_TRANSFORM_CACHE[cache_key]
        return pos_cached.copy(), quat_cached.copy()

    try:
        model, data = _load_mj_model_data()
        anchor_body_name = PART_MODEL_BODY[part_name]
        anchor_id = int(mj_name2id(model, mjtObj.mjOBJ_BODY, anchor_body_name))

        mesh_name = Path(mesh_file).stem
        mesh_id = int(mj_name2id(model, mjtObj.mjOBJ_MESH, mesh_name))

        mesh_geom_type = int(mjtGeom.mjGEOM_MESH)
        geom_ids: list[int] = []
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_type[geom_id]) != mesh_geom_type:
                continue
            if int(model.geom_dataid[geom_id]) != mesh_id:
                continue
            geom_ids.append(geom_id)
        if not geom_ids:
            return None

        preferred = [
            gid
            for gid in geom_ids
            if _is_descendant_or_self(model, int(model.geom_bodyid[gid]), anchor_id)
        ]
        candidates = preferred if preferred else geom_ids
        same_body_candidates = [
            gid for gid in candidates if int(model.geom_bodyid[gid]) == anchor_id
        ]
        if same_body_candidates:
            candidates = same_body_candidates

        # Prefer direct XML-local geom pose when a matching geom is attached to
        # the anchor body itself. This keeps optimization in the same local
        # frame used for taxel insertion.
        if same_body_candidates:
            selected_id = int(same_body_candidates[0])
            local_pos = np.array(model.geom_pos[selected_id], dtype=float)
            local_quat = normalize_quaternion_wxyz(
                np.array(model.geom_quat[selected_id], dtype=float)
            )
            _MESH_TRANSFORM_CACHE[cache_key] = (local_pos.copy(), local_quat.copy())
            return local_pos, local_quat

        anchor_pos = np.array(data.xpos[anchor_id], dtype=float)
        anchor_rot = np.array(data.xmat[anchor_id], dtype=float).reshape(3, 3)

        best_score = float("inf")
        best_pos: np.ndarray | None = None
        best_quat: np.ndarray | None = None

        for geom_id in candidates:
            geom_pos_world = np.array(data.geom_xpos[geom_id], dtype=float)
            geom_rot_world = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)

            rel_pos = anchor_rot.T @ (geom_pos_world - anchor_pos)
            rel_rot = anchor_rot.T @ geom_rot_world
            rel_quat = _quat_wxyz_from_rotation_matrix(rel_rot)
            score = float(np.linalg.norm(rel_pos))

            if score < best_score:
                best_score = score
                best_pos = rel_pos
                best_quat = rel_quat

        if best_pos is None or best_quat is None:
            return None

        _MESH_TRANSFORM_CACHE[cache_key] = (best_pos.copy(), best_quat.copy())
        return best_pos, best_quat
    except Exception:
        return None


def apply_part_transform(
    points: np.ndarray,
    part_name: str,
    delta_angles_deg: np.ndarray | None = None,
    delta_offsets_m: np.ndarray | None = None,
    delta_quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the full per-part transform pipeline to a taxel point cloud.

    The returned points include the fixed manual transform chain stored in the
    part configuration plus an optional final optimization delta represented as
    either Euler angles or a quaternion.
    """

    if delta_quat_wxyz is None:
        if delta_angles_deg is None:
            delta_angles_deg = np.zeros(3, dtype=float)
        delta_quat_wxyz = euler_xyz_deg_to_quat_wxyz(delta_angles_deg)
    else:
        delta_quat_wxyz = normalize_quaternion_wxyz(delta_quat_wxyz)
    if delta_offsets_m is None:
        delta_offsets_m = np.zeros(3, dtype=float)

    out = np.empty_like(points)
    cfg = PARTS_BY_NAME[part_name]
    steps = cfg.manual_steps
    for i in range(points.shape[0]):
        pos = np.array(points[i], dtype=float)
        for offsets, angles in steps:
            pos = rotate_position(
                pos=pos, offsets=offsets, angle_degrees=angles)
        pos = apply_quaternion_transform(
            pos=pos, offsets=delta_offsets_m, quat_wxyz=delta_quat_wxyz)
        out[i] = pos
    return out


def apply_manual_steps_with_euler_delta(
    points: np.ndarray,
    manual_steps: list[dict[str, Any]] | tuple[ManualStep, ...],
    delta_angles_deg: np.ndarray | None = None,
    delta_offsets_m: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the same per-taxel transform chain used by XML insertion.

    This mirrors include_skin_to_mujoco_model.py exactly: each point is run
    through all manual rotate_position steps and then one final delta step
    expressed as Euler XYZ degrees plus offsets.
    """

    if delta_angles_deg is None:
        delta_angles_deg = np.zeros(3, dtype=float)
    if delta_offsets_m is None:
        delta_offsets_m = np.zeros(3, dtype=float)

    steps: list[tuple[np.ndarray, np.ndarray]] = []
    for step in manual_steps:
        if isinstance(step, dict):
            offsets = np.array(step.get("offsets_m", [0.0, 0.0, 0.0]), dtype=float)
            angles = np.array(step.get("angles_deg", [0.0, 0.0, 0.0]), dtype=float)
        else:
            offsets_raw, angles_raw = step
            offsets = np.array(offsets_raw, dtype=float)
            angles = np.array(angles_raw, dtype=float)
        if offsets.shape != (3,) or angles.shape != (3,):
            raise ValueError("manual_steps entries must provide 3D offsets and 3D angles")
        steps.append((offsets, angles))

    out = np.empty_like(points)
    for i in range(points.shape[0]):
        pos = np.array(points[i], dtype=float)
        for offsets_step, angles_step in steps:
            pos = rotate_position(
                pos=pos,
                offsets=offsets_step.tolist(),
                angle_degrees=angles_step.tolist(),
            )
        pos = rotate_position(
            pos=pos,
            offsets=np.array(delta_offsets_m, dtype=float).tolist(),
            angle_degrees=np.array(delta_angles_deg, dtype=float).tolist(),
        )
        out[i] = pos
    return out


def load_taxel_points(position_file: Path, rebase: bool) -> np.ndarray:
    """Load tactile taxel coordinates from a positions file.

    The parser keeps only tactile channels according to `taxel2Repr`, drops
    unused zero rows, and optionally rebases the coordinate system to match the
    convention used by the original MuJoCo inclusion script.
    """

    calibration = read_calibration_data(str(position_file))
    taxel2repr = read_taxel2repr_data(str(position_file))
    taxels = validate_taxel_data(
        calibration, taxel2repr if taxel2repr else None)
    if rebase:
        taxels = rebase_coordinate_system(taxels)
    points = np.array([np.array(t[0], dtype=float)
                      for t in taxels], dtype=float)
    return points


def load_mesh(
    part_name: str,
    mesh_files: tuple[str, ...],
) -> trimesh.Trimesh:
    """Load target mesh in part-local frame, preferring MuJoCo compiled geometry.

    This keeps optimizer distance evaluation aligned with what MuJoCo actually
    simulates and what the visualizer renders. If compiled mesh lookup fails,
    we fall back to raw STL loading for robustness.
    """

    model, _ = _load_mj_model_data()
    compiled_meshes: list[trimesh.Trimesh] = []

    for mesh_file in mesh_files:
        mesh_name = Path(mesh_file).stem
        mesh_id = int(mj_name2id(model, mjtObj.mjOBJ_MESH, mesh_name))
        if mesh_id < 0:
            raise RuntimeError(f"Compiled MuJoCo mesh not found for asset: {mesh_name}")

        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])

        vertices = np.array(model.mesh_vert[vert_start:vert_start + vert_count], dtype=float)
        faces = np.array(model.mesh_face[face_start:face_start + face_count], dtype=int)

        resolved = resolve_mesh_transform_from_model(part_name, mesh_file)
        if resolved is None:
            raise RuntimeError(
                f"Could not resolve compiled mesh transform from model for part={part_name}, mesh={mesh_file}"
            )
        pos_arr, quat_arr = resolved
        rot = Rotation.from_quat(quat_wxyz_to_xyzw(quat_arr)).as_matrix()

        mesh_local = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
        mesh_local.vertices = (rot @ mesh_local.vertices.T).T + pos_arr
        compiled_meshes.append(mesh_local)

    if compiled_meshes:
        if len(compiled_meshes) == 1:
            return compiled_meshes[0]
        return trimesh.util.concatenate(compiled_meshes)

    print(
        f"[warn] Falling back to raw STL target meshes for part={part_name}; compiled mesh target unavailable."
    )

    meshes: list[trimesh.Trimesh] = []
    for mesh_file in mesh_files:
        mesh_path = MESH_DIR / mesh_file
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {mesh_file} in {MESH_DIR}")
        loaded = trimesh.load(mesh_path, force="mesh")
        if isinstance(loaded, trimesh.Trimesh):
            meshes.append(loaded)
        else:
            scene = cast(trimesh.Scene, loaded)
            meshes.extend(
                [m for m in scene.dump() if isinstance(m, trimesh.Trimesh)])
    if not meshes:
        raise RuntimeError(f"No valid meshes loaded from: {mesh_files}")

    # Most sim_* meshes are exported in millimeters, while taxel coordinates are in meters.
    # The palm collision meshes in this repo are already in meters.
    scales = []
    scaled_meshes: list[trimesh.Trimesh] = []
    for mesh_file, mesh in zip(mesh_files, meshes):
        mesh_local = mesh.copy()
        # Auto-detect mm-exported meshes by raw coordinate magnitude.
        max_abs = float(np.max(np.abs(mesh_local.vertices)))
        scale = 0.001 if max_abs > 2.0 else 1.0
        mesh_local.apply_scale(scale)

        resolved = resolve_mesh_transform_from_model(part_name, mesh_file)
        if resolved is None:
            raise RuntimeError(
                f"Could not resolve mesh transform from model for part={part_name}, mesh={mesh_file}. "
                "Fix mesh/body naming in icub_v2_full_body_improved.xml instead of using manual fallback transforms."
            )
        pos_arr, quat_arr = resolved

        # Apply geom orientation/translation from the MuJoCo model body frame.
        rot4 = trimesh.transformations.quaternion_matrix(quat_arr)
        mesh_local.apply_transform(rot4)
        mesh_local.apply_translation(pos_arr)

        scales.append(scale)
        scaled_meshes.append(mesh_local)

    print(f"mesh unit scales for {mesh_files}: {scales}")

    if len(meshes) == 1:
        return scaled_meshes[0]
    return trimesh.util.concatenate(scaled_meshes)


def build_distance_fn(
    mesh: trimesh.Trimesh,
    focus_reference_points: np.ndarray | None = None,
    focus_radius_m: float = 0.06,
    focus_min_samples: int = 3000,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a robust point-to-mesh distance function.

    Trimesh proximity queries are preferred when available. A KDTree built on a
    dense surface sample is kept as a fallback because proximity queries can be
    fragile for some meshes and environments.
    """

    sampled = np.asarray(mesh.sample(120000), dtype=float)

    use_restricted_surface = focus_reference_points is not None and len(focus_reference_points) > 0
    if use_restricted_surface:
        reference_points = np.asarray(focus_reference_points, dtype=float)
        reference_tree = KDTree(reference_points)
        d_ref, _ = reference_tree.query(sampled, k=1)
        d_ref = np.asarray(d_ref, dtype=float)

        keep_mask = d_ref <= float(focus_radius_m)
        if int(np.sum(keep_mask)) >= int(focus_min_samples):
            sampled = sampled[keep_mask]
        else:
            keep_count = max(1, min(int(focus_min_samples), sampled.shape[0]))
            nearest_idx = np.argpartition(d_ref, keep_count - 1)[:keep_count]
            sampled = sampled[nearest_idx]

        print(
            f"distance focus enabled: radius={focus_radius_m:.4f}m, samples={sampled.shape[0]}"
        )

    kdtree = KDTree(sampled)

    def dist(points: np.ndarray) -> np.ndarray:
        if use_restricted_surface:
            d, _ = kdtree.query(points, k=1)
            return np.asarray(d, dtype=float)
        try:
            _, distances, _ = trimesh.proximity.closest_point(mesh, points)
            return np.asarray(distances, dtype=float)
        except Exception:
            d, _ = kdtree.query(points, k=1)
            return np.asarray(d, dtype=float)

    return dist


def build_density_balancing_weights(points: np.ndarray, neighbors: int = 8) -> np.ndarray:
    """Return per-point weights that reduce bias from locally dense taxel regions.

    Points in sparse neighborhoods get larger weights; points in dense clusters get
    smaller weights. Weights are normalized to have mean 1.0.
    """

    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    k = max(2, min(int(neighbors) + 1, n))
    tree = KDTree(pts)
    distances, _ = tree.query(pts, k=k)
    distances = np.asarray(distances, dtype=float)

    local_scale = np.mean(distances[:, 1:], axis=1)
    eps = 1e-12
    mean_scale = float(np.mean(local_scale))
    raw_weights = local_scale / max(mean_scale, eps)
    clipped = np.clip(raw_weights, 0.25, 4.0)
    normalized = clipped / max(float(np.mean(clipped)), eps)
    return normalized.astype(float)


def mean_absolute_nearest_surface_distance(distances: np.ndarray) -> float:
    """Return plain mean absolute nearest-surface distance across all taxels."""

    d = np.asarray(distances, dtype=float)
    if d.size == 0:
        return 0.0
    return float(np.mean(np.abs(d)))


def weighted_signed_clearance_plane_metrics(
    signed_clearance: np.ndarray,
    patch_points: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return weighted RMS of clearance-plane residual and slope magnitude.

    The fitted model is d ~= a*u + b*v + c where (u, v) are patch-centered
    in-plane coordinates derived from PCA of patch points.
    """

    d = np.asarray(signed_clearance, dtype=float)
    p = np.asarray(patch_points, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = int(d.shape[0])

    if n < 3 or p.shape[0] != n:
        return 0.0, 0.0

    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return 0.0, 0.0

    centered = p - np.average(p, axis=0, weights=w)
    cov = (centered * w[:, None]).T @ centered / max(w_sum, 1e-12)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]
    uv = centered @ basis

    A = np.column_stack((uv[:, 0], uv[:, 1], np.ones(n, dtype=float)))
    Aw = A * np.sqrt(w)[:, None]
    dw = d * np.sqrt(w)
    coeff, *_ = np.linalg.lstsq(Aw, dw, rcond=None)
    residual = d - (A @ coeff)
    rms_residual = float(np.sqrt(np.average(residual ** 2, weights=w)))
    slope_mag = float(np.sqrt(coeff[0] ** 2 + coeff[1] ** 2))
    return rms_residual, slope_mag


def load_previous_optimized_seeds() -> PreviousSeedMap:
    """Load previously saved per-part optimization results as new seed states."""

    if not REPORT_JSON.exists():
        return {}

    try:
        raw = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}

    seeds: PreviousSeedMap = {}
    for entry in raw:
        if not isinstance(entry, dict) or "part" not in entry or "optimized" not in entry:
            continue

        part = str(entry["part"])
        optimized = cast(dict[str, Any], entry["optimized"])
        angle_values = optimized.get("delta_angles_deg")
        offset_values = optimized.get("delta_offsets_m")
        quat_values = optimized.get("delta_quaternion_wxyz")
        if offset_values is None:
            continue

        if quat_values is not None:
            quat = normalize_quaternion_wxyz(
                np.array(quat_values, dtype=float))
            angles = quat_wxyz_to_euler_xyz_deg(quat)
        elif angle_values is not None:
            angles = np.array(angle_values, dtype=float)
            quat = euler_xyz_deg_to_quat_wxyz(angles)
        else:
            continue

        seeds[part] = {
            "angles": angles,
            "quat_wxyz": quat,
            "offsets": np.array(offset_values, dtype=float),
        }

    return seeds


def build_seed_candidates(
    config: PartConfig,
    previous_seed: PreviousSeed | None,
    polish: bool = False,
) -> list[SeedCandidate]:
    """Construct unique seed hypotheses for one optimization run.

    Seeds can come from the static part configuration, the previously saved best
    solution, jittered variations around that solution, and an optional dense
    grid around the previous optimum.
    """

    candidates: list[SeedCandidate] = []
    seen: set[tuple[float, ...]] = set()

    def add_candidate(label: str, angles: np.ndarray, offsets: np.ndarray) -> None:
        key = tuple(np.round(np.hstack([angles, offsets]), 6).tolist())
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, angles.copy(), offsets.copy()))

    zero_offsets = np.zeros(3, dtype=float)

    if polish:
        if previous_seed is None:
            raise RuntimeError(
                f"Polish mode requires an existing optimized seed in {REPORT_JSON} for part '{config.part_name}'. "
                "Run optimize_taxel_alignment.py without --polish first to write baseline best results."
            )
        prev_angles = previous_seed["angles"]
        prev_offsets = previous_seed["offsets"]
        add_candidate("previous:optimized", prev_angles, prev_offsets)
        return candidates

    for idx, angle_seed in enumerate(config.delta_angle_seed_candidates):
        add_candidate(f"configured:{idx}", np.array(
            angle_seed, dtype=float), zero_offsets)

    if previous_seed is None or not config.include_previous_seed:
        return candidates

    prev_angles = previous_seed["angles"]
    prev_offsets = previous_seed["offsets"]
    add_candidate("previous:optimized", prev_angles, prev_offsets)

    for delta_angles in config.previous_angle_jitter_candidates:
        delta_arr = np.array(delta_angles, dtype=float)
        add_candidate(
            f"previous:angle_{delta_arr[0]:+.1f}_{delta_arr[1]:+.1f}_{delta_arr[2]:+.1f}",
            prev_angles + delta_arr,
            prev_offsets,
        )

    for delta_offsets in config.previous_offset_jitter_candidates:
        delta_arr = np.array(delta_offsets, dtype=float)
        add_candidate(
            f"previous:offset_{delta_arr[0]:+.3f}_{delta_arr[1]:+.3f}_{delta_arr[2]:+.3f}",
            prev_angles,
            prev_offsets + delta_arr,
        )

    if (
        config.previous_angle_grid_values_deg is not None
        and config.previous_offset_grid_values_m is not None
    ):
        ax, ay, az = config.previous_angle_grid_values_deg
        tx, ty, tz = config.previous_offset_grid_values_m
        for dx in ax:
            for dy in ay:
                for dz in az:
                    for ox in tx:
                        for oy in ty:
                            for oz in tz:
                                da = np.array([dx, dy, dz], dtype=float)
                                do = np.array([ox, oy, oz], dtype=float)
                                add_candidate(
                                    f"previous:grid_a{dx:+.1f}_{dy:+.1f}_{dz:+.1f}_t{ox:+.3f}_{oy:+.3f}_{oz:+.3f}",
                                    prev_angles + da,
                                    prev_offsets + do,
                                )

    return candidates


def optimize_part(
    config: PartConfig,
    previous_seed_map: PreviousSeedMap,
    polish: bool = False,
) -> JsonDict:
    """Optimize one body-part taxel cloud against its target mesh.

    The optimization runs in quaternion-plus-translation space around the exact
    manual baseline transform stored in the configuration. The reported values
    are therefore deltas around that manual baseline rather than absolute body
    transforms.
    """

    points = load_taxel_points(
        POSITIONS_DIR / config.position_file, config.rebase)
    mesh = load_mesh(config.part_name, config.mesh_files)
    body_property, strategy = resolve_strategy_for_part(config.part_name)
    use_flat_surface_optimizer = bool(
        config.optimizer_method == SOLE_2D_OPTIMIZER_METHOD
        or strategy.use_flat_surface_optimizer
    )
    if use_flat_surface_optimizer:
        if config.part_name in {"l_palm", "r_palm"}:
            return optimize_palm_surface_part(
                config,
                points,
                mesh,
                previous_seed_map,
                strategy=strategy,
                body_property=body_property,
                polish=polish,
            )
        return optimize_foot_sole_part(
            config,
            points,
            mesh,
            previous_seed_map,
            strategy=strategy,
            body_property=body_property,
            polish=polish,
        )

    baseline_points = apply_part_transform(points, config.part_name)
    use_distance_focus = bool(
        config.distance_focus_to_initial_patch and strategy.allow_distance_focus)
    dist_fn = build_distance_fn(
        mesh,
        focus_reference_points=baseline_points
        if use_distance_focus
        else None,
        focus_radius_m=config.distance_focus_radius_m,
        focus_min_samples=config.distance_focus_min_samples,
    )

    inside_penalty_weight = max(
        float(config.inside_penalty_weight), float(strategy.inside_penalty_weight)
    )

    point_weights = np.ones(points.shape[0], dtype=float)
    if strategy.density_balance_enabled:
        point_weights = build_density_balancing_weights(
            points,
            neighbors=int(strategy.density_balance_neighbors),
        )
        print(
            f"[{config.part_name}] density balancing enabled: neighbors={int(strategy.density_balance_neighbors)}",
            flush=True,
        )

    baseline_patch_points = apply_part_transform(points, config.part_name)

    # We optimize a final quaternion perturbation applied on top of the exact manual chain.
    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    angles0 = quat_wxyz_to_euler_xyz_deg(quat0)
    offsets0 = np.zeros(3, dtype=float)
    x0 = np.hstack([quat0, offsets0])

    angle_bounds = _scaled_angle_bounds(config)
    translation_bounds = _scaled_translation_bounds(config)

    def unpack_params(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split the optimizer state into quaternion, translation, and Euler views."""

        quat = normalize_quaternion_wxyz(params[:4])
        offsets = np.array(params[4:], dtype=float)
        angles = quat_wxyz_to_euler_xyz_deg(quat)
        return quat, offsets, angles

    def mean_distance(params: np.ndarray) -> float:
        """Return the mean point-to-surface distance for the current parameter vector."""

        quat, offsets, _ = unpack_params(params)
        transformed = apply_part_transform(
            points,
            config.part_name,
            delta_offsets_m=offsets,
            delta_quat_wxyz=quat,
        )
        distances = dist_fn(transformed)
        return mean_absolute_nearest_surface_distance(distances)

    def inside_mask(points_world: np.ndarray) -> np.ndarray:
        """Return inside-mesh mask, or all-false when containment queries are unavailable."""

        if (
            inside_penalty_weight <= 0.0
            and float(strategy.clearance_penalty_weight) <= 0.0
            and float(strategy.clearance_balance_weight) <= 0.0
        ):
            return np.zeros(points_world.shape[0], dtype=bool)
        try:
            return np.asarray(mesh.contains(points_world), dtype=bool)
        except Exception:
            return np.zeros(points_world.shape[0], dtype=bool)

    def distance_stats(params: np.ndarray) -> dict[str, float]:
        """Return summary distance statistics for report generation."""

        quat, offsets, _ = unpack_params(params)
        transformed = apply_part_transform(
            points,
            config.part_name,
            delta_offsets_m=offsets,
            delta_quat_wxyz=quat,
        )
        distances = dist_fn(transformed)
        inside = inside_mask(transformed)
        return {
            "mean_m": mean_absolute_nearest_surface_distance(distances),
            "median_m": float(np.median(distances)),
            "p90_m": float(np.quantile(distances, 0.9)),
            "inside_fraction": float(np.average(inside.astype(float), weights=point_weights)),
        }

    def objective(
        params: np.ndarray,
        quat_center: np.ndarray,
        offset_center: np.ndarray,
        angle_center: np.ndarray,
    ) -> float:
        """Robust objective with soft regularization around the current seed hypothesis."""

        quat, offsets, angles = unpack_params(params)
        transformed = apply_part_transform(
            points,
            config.part_name,
            delta_offsets_m=offsets,
            delta_quat_wxyz=quat,
        )
        distances = dist_fn(transformed)
        # Simplified objective: minimize plain mean absolute nearest-surface distance.
        return mean_absolute_nearest_surface_distance(distances)

    initial_stats = distance_stats(x0)
    initial_mean = float(initial_stats["mean_m"])
    print(f"[{config.part_name}] initial mean distance={initial_mean:.6f} m")

    previous_seed = previous_seed_map.get(config.part_name)
    seed_candidates = build_seed_candidates(
        config, previous_seed, polish=polish)

    seed_results: list[tuple[float, np.ndarray, Any,
                             np.ndarray, np.ndarray, np.ndarray, str]] = []

    for seed_idx, (seed_label, seed_angles, seed_offsets) in enumerate(seed_candidates):
        print(
            f"\n[{config.part_name}] Trying seed {seed_idx+1}/{len(seed_candidates)}: {seed_label}")
        print(
            f"    angles: {seed_angles.tolist()} offsets: {seed_offsets.tolist()}", flush=True)
        seed_quat = euler_xyz_deg_to_quat_wxyz(seed_angles)
        x_seed = np.hstack([seed_quat, seed_offsets])
        bounds = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (seed_offsets[0] - translation_bounds[0],
             seed_offsets[0] + translation_bounds[0]),
            (seed_offsets[1] - translation_bounds[1],
             seed_offsets[1] + translation_bounds[1]),
            (seed_offsets[2] - translation_bounds[2],
             seed_offsets[2] + translation_bounds[2]),
        ]

        x_start = x_seed

        iteration = {"count": 0}
        best_so_far = {"score": mean_distance(
            x_start), "params": x_start.copy()}

        def callback(xk: np.ndarray) -> None:
            iteration["count"] += 1
            md = mean_distance(xk)
            print(
                f"[{config.part_name}] seed={seed_idx+1} ({seed_label}) iter={iteration['count']:02d} mean_distance={md:.6f} m",
                flush=True,
            )
            if md < best_so_far["score"]:
                best_so_far["score"] = md
                best_so_far["params"] = xk.copy()

        primary_maxiter = 500 if polish else 200
        primary_ftol = 1e-16 if polish else 1e-12
        result = minimize(
            lambda p: objective(p, seed_quat, seed_offsets, seed_angles),
            x_start,
            method=LOCAL_OPTIMIZER_METHOD,
            bounds=bounds,
            callback=callback,
            options={"maxiter": primary_maxiter,
                     "ftol": primary_ftol, "gtol": 1e-12},
        )

        # Use best-so-far parameters, not just final iterate
        x_candidate = best_so_far["params"]
        score = best_so_far["score"]
        seed_results.append((score, x_candidate, result,
                            seed_angles, seed_quat, seed_offsets, seed_label))

    seed_results.sort(key=lambda x: x[0])
    (
        _,
        x_final,
        result,
        selected_seed_angles,
        selected_seed_quat,
        selected_seed_offsets,
        selected_seed_label,
    ) = seed_results[0]

    fine_refinement: dict[str, Any] | None = None
    if config.local_refine_angle_jitter_candidates:
        angle_window_local = np.minimum(
            angle_bounds,
            np.array(config.local_refine_angle_window_deg, dtype=float),
        )
        translation_window_local = np.minimum(
            translation_bounds,
            np.array(config.local_refine_translation_window_m, dtype=float),
        )
        angle_jitters = tuple(np.array(j, dtype=float)
                              for j in config.local_refine_angle_jitter_candidates)

        local_results: list[tuple[float, np.ndarray, Any, np.ndarray]] = []
        for local_idx, angle_jitter in enumerate(angle_jitters):
            x_seed_local = x_final.copy()
            _, _, x_final_euler = unpack_params(x_final)
            jittered_euler = x_final_euler + angle_jitter
            x_seed_local[:4] = euler_xyz_deg_to_quat_wxyz(jittered_euler)

            bounds_local = [
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (x_final[4] - translation_window_local[0],
                 x_final[4] + translation_window_local[0]),
                (x_final[5] - translation_window_local[1],
                 x_final[5] + translation_window_local[1]),
                (x_final[6] - translation_window_local[2],
                 x_final[6] + translation_window_local[2]),
            ]

            fine_maxiter = 1000 if polish else 200
            fine_ftol = 1e-20 if polish else 1e-15
            result_local = minimize(
                lambda p: objective(p, normalize_quaternion_wxyz(
                    x_final[:4]), x_final[4:], x_final_euler),
                x_seed_local,
                method=LOCAL_OPTIMIZER_METHOD,
                bounds=bounds_local,
                options={"maxiter": fine_maxiter,
                         "ftol": fine_ftol, "gtol": 1e-16},
            )
            x_candidate_local = result_local.x if result_local.success else x_seed_local
            score_local = mean_distance(x_candidate_local)
            local_results.append(
                (score_local, x_candidate_local, result_local, angle_jitter))
            print(
                f"[{config.part_name}] fine seed={local_idx+1} jitter={angle_jitter.tolist()} mean_distance={score_local:.6f} m"
            )

        local_results.sort(key=lambda x: x[0])
        best_local_score, best_local_x, best_local_result, best_local_jitter = local_results[
            0]
        previous_score = mean_distance(x_final)
        if best_local_score < previous_score:
            x_final = best_local_x
            result = best_local_result

        fine_refinement = {
            "candidate_count": len(angle_jitters),
            "selected_jitter_deg": best_local_jitter.tolist(),
            "pre_refine_mean_distance_m": float(previous_score),
            "post_refine_mean_distance_m": float(mean_distance(x_final)),
            "angle_window_deg": angle_window_local.tolist(),
            "translation_window_m": translation_window_local.tolist(),
        }

    polish_refinement: dict[str, Any] | None = None
    if polish:
        translation_window_polish = np.minimum(
            translation_bounds, np.array([0.001, 0.001, 0.001], dtype=float))
        bounds_polish = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (x_final[4] - translation_window_polish[0],
             x_final[4] + translation_window_polish[0]),
            (x_final[5] - translation_window_polish[1],
             x_final[5] + translation_window_polish[1]),
            (x_final[6] - translation_window_polish[2],
             x_final[6] + translation_window_polish[2]),
        ]

        polish_before = mean_distance(x_final)
        result_polish = minimize(
            lambda p: objective(p, normalize_quaternion_wxyz(
                x_final[:4]), x_final[4:], quat_wxyz_to_euler_xyz_deg(normalize_quaternion_wxyz(x_final[:4]))),
            x_final,
            method=LOCAL_OPTIMIZER_METHOD,
            bounds=bounds_polish,
            options={"maxiter": 2000, "ftol": 1e-25, "gtol": 1e-20},
        )
        x_polished = result_polish.x if result_polish.success else x_final
        polish_after = mean_distance(x_polished)
        if polish_after < polish_before:
            x_final = x_polished
            result = result_polish

        polish_refinement = {
            "pre_polish_mean_distance_m": float(polish_before),
            "post_polish_mean_distance_m": float(mean_distance(x_final)),
            "rotation_space": "quaternion_wxyz",
            "translation_window_m": translation_window_polish.tolist(),
            "success": bool(result_polish.success),
        }

    final_quat = normalize_quaternion_wxyz(x_final[:4])
    final_offsets = x_final[4:]
    final_angles = quat_wxyz_to_euler_xyz_deg(final_quat)

    final_stats = distance_stats(x_final)
    final_mean = float(final_stats["mean_m"])
    accepted_vs_initial = final_mean < initial_mean
    if not accepted_vs_initial:
        print(
            f"[{config.part_name}] optimized mean_distance={final_mean:.6f} m is not better than initial={initial_mean:.6f} m; reverting to initial baseline."
        )
        x_final = x0.copy()
        final_quat = normalize_quaternion_wxyz(x_final[:4])
        final_offsets = x_final[4:]
        final_angles = quat_wxyz_to_euler_xyz_deg(final_quat)
        final_stats = distance_stats(x_final)
        final_mean = float(final_stats["mean_m"])

    return {
        "part": config.part_name,
        "model_body_name": PART_MODEL_BODY[config.part_name],
        "sensor_group": PART_SENSOR_GROUP[config.part_name],
        "position_file": config.position_file,
        "rebase": bool(config.rebase),
        "include_to_model": bool(config.include_to_model),
        "mesh_files": list(config.mesh_files),
        "manual_steps": [
            {"offsets_m": list(step[0]), "angles_deg": list(step[1])}
            for step in config.manual_steps
        ],
        "initial": {
            "delta_angles_deg": angles0.tolist(),
            "delta_quaternion_wxyz": quat0.tolist(),
            "delta_offsets_m": offsets0.tolist(),
            "mean_distance_m": initial_stats["mean_m"],
            "median_distance_m": initial_stats["median_m"],
            "p90_distance_m": initial_stats["p90_m"],
            "inside_fraction": initial_stats["inside_fraction"],
        },
        "optimized": {
            "delta_angles_deg": final_angles.tolist(),
            "delta_quaternion_wxyz": final_quat.tolist(),
            "delta_offsets_m": final_offsets.tolist(),
            "mean_distance_m": final_stats["mean_m"],
            "median_distance_m": final_stats["median_m"],
            "p90_distance_m": final_stats["p90_m"],
            "inside_fraction": final_stats["inside_fraction"],
        },
        "delta": {
            "angles_deg": (final_angles - angles0).tolist(),
            "quaternion_wxyz": final_quat.tolist(),
            "rotation_distance_deg": quaternion_geodesic_distance_deg(final_quat, quat0),
            "offsets_m": (final_offsets - offsets0).tolist(),
            "mean_distance_m": float(final_mean - initial_mean),
            "mean_distance_improvement_pct": float((initial_mean - final_mean) / max(initial_mean, 1e-12) * 100.0),
            "inside_fraction": float(final_stats["inside_fraction"] - initial_stats["inside_fraction"]),
        },
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "objective": float(result.fun),
            "optimizer_method": config.optimizer_method,
            "solver": LOCAL_OPTIMIZER_METHOD,
            "seed_count": len(seed_candidates),
            "selected_seed_label": selected_seed_label,
            "selected_seed_angles_deg": selected_seed_angles.tolist(),
            "selected_seed_quaternion_wxyz": normalize_quaternion_wxyz(selected_seed_quat).tolist(),
            "selected_seed_offsets_m": selected_seed_offsets.tolist(),
            "angle_bounds_deg": angle_bounds.tolist(),
            "translation_bounds_m": translation_bounds.tolist(),
            "fine_refinement": fine_refinement,
            "polish_mode": bool(polish),
            "polish_refinement": polish_refinement,
            "body_property": body_property,
            "strategy_name": strategy.name,
            "inside_penalty_weight": inside_penalty_weight,
            "distance_focus_enabled": use_distance_focus,
            "density_balance_enabled": bool(strategy.density_balance_enabled),
            "density_balance_neighbors": int(strategy.density_balance_neighbors),
            "clearance_target_m": float(strategy.clearance_target_m),
            "clearance_mean_weight": float(strategy.clearance_mean_weight),
            "clearance_penalty_weight": float(strategy.clearance_penalty_weight),
            "clearance_balance_weight": float(strategy.clearance_balance_weight),
            "min_clearance_m": float(strategy.min_clearance_m),
            "min_clearance_penalty_weight": float(strategy.min_clearance_penalty_weight),
            "tilt_plane_penalty_weight": float(strategy.tilt_plane_penalty_weight),
            "objective_distance_mode": "mean_absolute_nearest_surface_all_taxels",
            "accepted_vs_initial": bool(accepted_vs_initial),
        },
    }


def write_reports(results: list[dict[str, Any]]) -> None:
    """Write the machine-readable JSON report and the human-readable text report."""

    REPORT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("Taxel alignment optimization report")
    lines.append("")
    for r in results:
        part = cast(str, r["part"])
        init = cast(dict[str, Any], r["initial"])
        opt = cast(dict[str, Any], r["optimized"])
        delta = cast(dict[str, Any], r["delta"])
        lines.append(f"Part: {part}")
        lines.append(
            f"  mean distance: {init['mean_distance_m']:.6f} -> {opt['mean_distance_m']:.6f} m "
            f"({delta['mean_distance_improvement_pct']:.2f}% improvement)"
        )
        lines.append(
            f"  mean abs improvement: {(init['mean_distance_m'] - opt['mean_distance_m']) * 1e3:.3f} mm"
        )
        lines.append(
            f"  median distance: {init['median_distance_m']:.6f} -> {opt['median_distance_m']:.6f} m"
        )
        lines.append(
            f"  p90 distance: {init['p90_distance_m']:.6f} -> {opt['p90_distance_m']:.6f} m"
        )
        lines.append(
            f"  delta angles deg (around manual): {init['delta_angles_deg']} -> {opt['delta_angles_deg']}")
        if opt.get("delta_quaternion_wxyz") is not None:
            lines.append(
                f"  delta quaternion wxyz (around manual): {opt['delta_quaternion_wxyz']}")
        lines.append(
            f"  delta offsets m (around manual): {init['delta_offsets_m']} -> {opt['delta_offsets_m']}")
        lines.append(f"  delta angles deg: {delta['angles_deg']}")
        if delta.get("rotation_distance_deg") is not None:
            lines.append(
                f"  delta rotation distance deg: {delta['rotation_distance_deg']}")
        lines.append(f"  delta offsets m: {delta['offsets_m']}")
        optimizer = cast(dict[str, Any], r["optimizer"])
        lines.append(
            f"  optimizer method: {optimizer.get('optimizer_method', LOCAL_OPTIMIZER_METHOD)}")
        lines.append(
            f"  solver: {optimizer.get('solver', LOCAL_OPTIMIZER_METHOD)}")
        lines.append(f"  seed count: {optimizer['seed_count']}")
        lines.append(
            f"  selected seed label: {optimizer['selected_seed_label']}")
        lines.append(
            f"  selected seed angles deg: {optimizer['selected_seed_angles_deg']}")
        if optimizer.get("selected_seed_quaternion_wxyz") is not None:
            lines.append(
                f"  selected seed quaternion wxyz: {optimizer['selected_seed_quaternion_wxyz']}")
        lines.append(
            f"  selected seed offsets m: {optimizer['selected_seed_offsets_m']}")
        lines.append(f"  angle bounds deg: {optimizer['angle_bounds_deg']}")
        lines.append(
            f"  translation bounds m: {optimizer['translation_bounds_m']}")
        if optimizer.get("fine_refinement") is not None:
            fr = cast(dict[str, Any], optimizer["fine_refinement"])
            lines.append(f"  fine refine candidates: {fr['candidate_count']}")
            lines.append(
                f"  fine refine selected jitter deg: {fr['selected_jitter_deg']}")
            lines.append(
                f"  fine refine mean distance: {fr['pre_refine_mean_distance_m']:.6f} -> {fr['post_refine_mean_distance_m']:.6f} m"
            )
            lines.append(
                f"  fine refine angle window deg: {fr['angle_window_deg']}")
            lines.append(
                f"  fine refine translation window m: {fr['translation_window_m']}")
        lines.append(f"  polish mode: {optimizer.get('polish_mode', False)}")
        if optimizer.get("polish_refinement") is not None:
            pr = cast(dict[str, Any], optimizer["polish_refinement"])
            lines.append(
                f"  polish mean distance: {pr['pre_polish_mean_distance_m']:.6f} -> {pr['post_polish_mean_distance_m']:.6f} m"
            )
            if pr.get("angle_window_deg") is not None:
                lines.append(
                    f"  polish angle window deg: {pr['angle_window_deg']}")
            if pr.get("rotation_space") is not None:
                lines.append(
                    f"  polish rotation space: {pr['rotation_space']}")
            lines.append(
                f"  polish translation window m: {pr['translation_window_m']}")
            lines.append(f"  polish success: {pr['success']}")
        lines.append("")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def load_existing_results() -> list[JsonDict]:
    """Load previously written result entries from the JSON report, if present."""

    if not REPORT_JSON.exists():
        return []

    try:
        raw = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict) and "part" in entry]


def build_baseline_result_entry(config: PartConfig) -> JsonDict:
    """Build a deterministic zero-delta report entry for one part.

    This is used to keep the persisted optimization report complete even when a
    run only optimizes a subset of parts. The entry captures the manual baseline
    transform chain with no optimized refinement applied.
    """

    points = load_taxel_points(POSITIONS_DIR / config.position_file, config.rebase)
    mesh = load_mesh(config.part_name, config.mesh_files)
    baseline_points = apply_part_transform(points, config.part_name)
    body_property, strategy = resolve_strategy_for_part(config.part_name)
    use_distance_focus = bool(
        config.distance_focus_to_initial_patch and strategy.allow_distance_focus
    )
    dist_fn = build_distance_fn(
        mesh,
        focus_reference_points=baseline_points if use_distance_focus else None,
        focus_radius_m=config.distance_focus_radius_m,
        focus_min_samples=config.distance_focus_min_samples,
    )

    abs_distances = np.asarray(dist_fn(baseline_points), dtype=float)
    mean_distance = float(np.mean(abs_distances))
    median_distance = float(np.median(abs_distances))
    p90_distance = float(np.quantile(abs_distances, 0.9))
    inside_penalty_weight = max(
        float(config.inside_penalty_weight), float(strategy.inside_penalty_weight)
    )

    zero_angles = np.zeros(3, dtype=float)
    zero_offsets = np.zeros(3, dtype=float)
    zero_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    return {
        "part": config.part_name,
        "model_body_name": PART_MODEL_BODY[config.part_name],
        "sensor_group": PART_SENSOR_GROUP[config.part_name],
        "position_file": config.position_file,
        "rebase": bool(config.rebase),
        "include_to_model": bool(config.include_to_model),
        "mesh_files": list(config.mesh_files),
        "manual_steps": [
            {"offsets_m": list(step[0]), "angles_deg": list(step[1])}
            for step in config.manual_steps
        ],
        "initial": {
            "delta_angles_deg": zero_angles.tolist(),
            "delta_quaternion_wxyz": zero_quat.tolist(),
            "delta_offsets_m": zero_offsets.tolist(),
            "mean_distance_m": mean_distance,
            "median_distance_m": median_distance,
            "p90_distance_m": p90_distance,
            "inside_fraction": 0.0,
        },
        "optimized": {
            "delta_angles_deg": zero_angles.tolist(),
            "delta_quaternion_wxyz": zero_quat.tolist(),
            "delta_offsets_m": zero_offsets.tolist(),
            "mean_distance_m": mean_distance,
            "median_distance_m": median_distance,
            "p90_distance_m": p90_distance,
            "inside_fraction": 0.0,
        },
        "delta": {
            "angles_deg": zero_angles.tolist(),
            "quaternion_wxyz": zero_quat.tolist(),
            "rotation_distance_deg": 0.0,
            "offsets_m": zero_offsets.tolist(),
            "mean_distance_m": 0.0,
            "mean_distance_improvement_pct": 0.0,
            "inside_fraction": 0.0,
        },
        "optimizer": {
            "success": True,
            "status": 0,
            "message": "baseline_only",
            "nit": 0,
            "nfev": 0,
            "objective": mean_distance,
            "optimizer_method": config.optimizer_method,
            "solver": "baseline-only",
            "seed_count": 0,
            "selected_seed_label": "baseline-only",
            "selected_seed_angles_deg": zero_angles.tolist(),
            "selected_seed_quaternion_wxyz": zero_quat.tolist(),
            "selected_seed_offsets_m": zero_offsets.tolist(),
            "angle_bounds_deg": list(config.delta_angle_bounds_deg),
            "translation_bounds_m": list(config.delta_translation_bounds_m),
            "fine_refinement": None,
            "polish_mode": False,
            "polish_refinement": None,
            "body_property": body_property,
            "strategy_name": strategy.name,
            "inside_penalty_weight": inside_penalty_weight,
            "distance_focus_enabled": use_distance_focus,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selective optimization and polishing."""

    parser = argparse.ArgumentParser(
        description="Optimize taxel patch alignment against body meshes.")
    parser.add_argument(
        "--parts",
        nargs="+",
        help="Optional part names to optimize, for example: --parts r_forearm",
    )
    parser.add_argument(
        "--polish",
        action="store_true",
        help="Run strict convergence and a tight local refinement around current best solutions.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Always store the newly optimized result for selected parts, even when mean distance is worse than the previously saved entry.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore previously saved seeds and results for this run (does not delete report files).",
    )
    return parser.parse_args()


def _result_mean_distance(result_entry: JsonDict) -> float:
    """Extract the optimized mean distance from a stored result entry."""

    return float(cast(dict[str, Any], result_entry["optimized"])["mean_distance_m"])


def _upgrade_result_entry_schema(entry: JsonDict) -> JsonDict:
    """Backfill newly introduced report fields on legacy entries."""

    part = entry.get("part")
    if not isinstance(part, str) or part not in PARTS_BY_NAME:
        return entry

    cfg = PARTS_BY_NAME[part]

    if "model_body_name" not in entry:
        entry["model_body_name"] = PART_MODEL_BODY[part]
    if "sensor_group" not in entry:
        entry["sensor_group"] = PART_SENSOR_GROUP[part]
    if "position_file" not in entry:
        entry["position_file"] = cfg.position_file
    if "rebase" not in entry:
        entry["rebase"] = bool(cfg.rebase)
    if "include_to_model" not in entry:
        if "inlcude_to_model" in entry:
            entry["include_to_model"] = bool(entry["inlcude_to_model"])
        else:
            entry["include_to_model"] = bool(cfg.include_to_model)
    if "inlcude_to_model" in entry:
        del entry["inlcude_to_model"]
    if "mesh_files" not in entry:
        entry["mesh_files"] = list(cfg.mesh_files)
    if "manual_steps" not in entry:
        entry["manual_steps"] = [
            {"offsets_m": list(step[0]), "angles_deg": list(step[1])}
            for step in cfg.manual_steps
        ]

    return entry


def main() -> None:
    """Run the configured optimization workflow and persist improved results."""

    args = parse_args()
    validate_body_property_coverage({cfg.part_name for cfg in PARTS})
    selected_parts = set(args.parts) if args.parts else None
    previous_seed_map = {} if args.fresh_start else load_previous_optimized_seeds()
    existing_results = load_existing_results()
    existing_results = [_upgrade_result_entry_schema(
        entry) for entry in existing_results]
    result_by_part = {cast(str, entry["part"])                      : entry for entry in existing_results}

    for part_cfg in PARTS:
        if selected_parts is not None and part_cfg.part_name not in selected_parts:
            continue
        print(f"\n=== Optimizing {part_cfg.part_name} ===")
        new_result = optimize_part(
            part_cfg, previous_seed_map, polish=args.polish)

        # Only update if new result is better than existing best (including initialized values)
        new_mean = _result_mean_distance(new_result)
        if args.overwrite_results:
            result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(
                new_result)
            print(
                f"  overwrite enabled: storing mean_distance {new_mean:.6f} m.")
            continue

        if part_cfg.part_name not in result_by_part:
            result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(
                new_result)
        else:
            best_mean = _result_mean_distance(
                result_by_part[part_cfg.part_name])
            if new_mean < best_mean:
                result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(
                    new_result)
                print(
                    f"  improved mean_distance {best_mean:.6f} m -> {new_mean:.6f} m, updating stored result.")
            else:
                result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(
                    result_by_part[part_cfg.part_name])
                print(
                    f"  mean_distance {new_mean:.6f} m >= best {best_mean:.6f} m, keeping previous result.")

    for part_cfg in PARTS:
        if part_cfg.part_name in result_by_part:
            continue
        baseline_entry = build_baseline_result_entry(part_cfg)
        result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(
            baseline_entry)
        print(
            f"  seeded canonical baseline entry for missing part: {part_cfg.part_name}"
        )

    results = [result_by_part[part_cfg.part_name]
               for part_cfg in PARTS if part_cfg.part_name in result_by_part]

    write_reports(results)
    print("\nOptimization finished.")
    print(f"JSON report: {REPORT_JSON}")
    print(f"Text report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
