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
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --part r_upper_arm torso
        
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --polish
    
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --polish --part r_upper_arm torso
        
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py --preview-footprint --parts l_foot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

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
    mesh_pos: tuple[Vector3, ...]
    mesh_quat_wxyz: tuple[QuaternionWXYZ, ...]
    rebase: bool
    include_to_model: bool = True
    optimizer_method: str = "L-BFGS-B"
    manual_steps: tuple[ManualStep, ...] = ()
    delta_angle_seed_candidates: tuple[Vector3, ...] = ((0.0, 0.0, 0.0),)
    delta_angle_bounds_deg: Vector3 = (25.0, 25.0, 25.0)
    delta_translation_bounds_m: Vector3 = (0.03, 0.03, 0.03)
    include_previous_seed: bool = True
    previous_angle_jitter_candidates: tuple[Vector3, ...] = ()
    previous_offset_jitter_candidates: tuple[Vector3, ...] = ()
    previous_angle_grid_values_deg: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = None
    previous_offset_grid_values_m: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = None
    local_refine_angle_jitter_candidates: tuple[Vector3, ...] = ()
    local_refine_angle_window_deg: Vector3 = (6.0, 6.0, 6.0)
    local_refine_translation_window_m: Vector3 = (0.002, 0.002, 0.002)


ROOT = Path(__file__).resolve().parents[2]
POSITIONS_DIR = ROOT / "neuromorphic_body_schema" / "include_taxels" / "positions"
MESH_DIR = ROOT / "neuromorphic_body_schema" / "meshes" / "iCub"
REPORT_JSON = ROOT / "neuromorphic_body_schema" / "include_taxels" / "taxel_alignment_optimization_report.json"
REPORT_TXT = ROOT / "neuromorphic_body_schema" / "include_taxels" / "taxel_alignment_optimization_report.txt"
LOCAL_OPTIMIZER_METHOD = "L-BFGS-B"
SOLE_2D_OPTIMIZER_METHOD = "sole-2d"

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

    rot = Rotation.from_euler("xyz", np.array(angles_deg, dtype=float), degrees=True)
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
    distances = [float(footprint.boundary.distance(Point(x, z))) for x, z in coords]
    return float(np.mean(distances))


def optimize_foot_sole_part(
    config: PartConfig,
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    previous_seed_map: PreviousSeedMap,
    polish: bool = False,
) -> JsonDict:
    """Optimize a flat foot sole with exact Y snap and 2D XZ plus Y-rotation fitting."""

    if config.optimizer_method != SOLE_2D_OPTIMIZER_METHOD:
        raise ValueError(f"Unsupported foot optimizer method: {config.optimizer_method}")

    baseline_points = apply_part_transform(points, config.part_name)
    footprint = build_bottom_contact_footprint_xz(mesh)
    footprint_area = max(float(footprint.area), 1e-12)
    footprint_scale = max(float(np.sqrt(footprint_area)), 1e-6)
    footprint_centroid = np.array([footprint.centroid.x, footprint.centroid.y], dtype=float)
    dist_fn = build_distance_fn(mesh)

    y_target = sole_surface_y(mesh)
    y_snap_offset = float(y_target - np.mean(baseline_points[:, 1]))

    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    angles0 = quat_wxyz_to_euler_xyz_deg(quat0)
    offsets0 = np.zeros(3, dtype=float)
    x0 = np.hstack([quat0, offsets0])

    def transform_with_foot_params(y_angle_deg: float, tx: float, tz: float) -> np.ndarray:
        return apply_part_transform(
            points,
            config.part_name,
            delta_angles_deg=np.array([0.0, y_angle_deg, 0.0], dtype=float),
            delta_offsets_m=np.array([tx, y_snap_offset, tz], dtype=float),
        )

    def mean_distance_full(y_angle_deg: float, tx: float, tz: float) -> float:
        transformed = transform_with_foot_params(y_angle_deg, tx, tz)
        distances = dist_fn(transformed)
        return float(np.mean(distances))

    def mean_distance(params: np.ndarray) -> float:
        return mean_distance_full(float(params[0]), float(params[1]), float(params[2]))

    def distance_stats(params: np.ndarray) -> dict[str, float]:
        transformed = transform_with_foot_params(float(params[0]), float(params[1]), float(params[2]))
        abs_distances = dist_fn(transformed)
        return {
            "mean_m": float(np.mean(abs_distances)),
            "median_m": float(np.median(abs_distances)),
            "p90_m": float(np.quantile(abs_distances, 0.9)),
        }

    def objective(params: np.ndarray, angle_center: float, xz_center: np.ndarray) -> float:
        y_angle_deg = float(params[0])
        tx = float(params[1])
        tz = float(params[2])
        transformed = transform_with_foot_params(y_angle_deg, tx, tz)
        hull = build_point_hull_xz(transformed)

        symmetric_diff_ratio = float(footprint.symmetric_difference(hull).area / footprint_area)
        hull_centroid = np.array([hull.centroid.x, hull.centroid.y], dtype=float)
        centroid_penalty = float(np.sum(((hull_centroid - footprint_centroid) / footprint_scale) ** 2))
        boundary_penalty = _foot_boundary_distance_mean(hull, footprint) / footprint_scale

        angle_reg = ((y_angle_deg - angle_center) / 8.0) ** 2
        xz_reg = float(np.sum((((np.array([tx, tz], dtype=float) - xz_center) / 0.01) ** 2)))

        return float(symmetric_diff_ratio + 0.5 * centroid_penalty + 0.35 * boundary_penalty + 0.001 * angle_reg + 0.001 * xz_reg)

    initial_stats = distance_stats(np.array([0.0, 0.0, 0.0], dtype=float))
    initial_mean = float(initial_stats["mean_m"])
    print(
        f"[{config.part_name}] initial mean distance={initial_mean:.6f} m | y_snap_offset={y_snap_offset:.6f} m"
    )

    previous_seed = previous_seed_map.get(config.part_name)
    seed_candidates = build_seed_candidates(config, previous_seed, polish=polish)
    seed_results: list[tuple[float, np.ndarray, Any, float, np.ndarray, str]] = []

    translation_bounds = np.array(config.delta_translation_bounds_m, dtype=float)
    angle_bound_y = float(config.delta_angle_bounds_deg[1])

    for seed_idx, (seed_label, seed_angles, seed_offsets) in enumerate(seed_candidates):
        seed_angle_y = float(seed_angles[1])
        seed_xz = np.array([seed_offsets[0], seed_offsets[2]], dtype=float)
        x_seed = np.array([seed_angle_y, seed_xz[0], seed_xz[1]], dtype=float)

        print(f"\n[{config.part_name}] Trying seed {seed_idx}/{len(seed_candidates)}: {seed_label}")
        print(
            f"    angle_y_deg: {seed_angle_y:.6f} offsets_xyz: {[seed_xz[0], y_snap_offset, seed_xz[1]]}",
            flush=True,
        )

        bounds = [
            (seed_angle_y - angle_bound_y, seed_angle_y + angle_bound_y),
            (seed_xz[0] - translation_bounds[0], seed_xz[0] + translation_bounds[0]),
            (seed_xz[1] - translation_bounds[2], seed_xz[1] + translation_bounds[2]),
        ]

        iteration = {"count": 0}
        best_so_far = {"score": mean_distance(x_seed), "params": x_seed.copy()}

        def callback(xk: np.ndarray) -> None:
            iteration["count"] += 1
            md = mean_distance(xk)
            print(
                f"[{config.part_name}] seed={seed_idx} ({seed_label}) iter={iteration['count']:02d} mean_distance={md:.6f} m",
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
        # Second-stage foot polish: tighter local window around best 2D solution.
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
        polish_after = mean_distance(x_polished)
        if polish_after < polish_before:
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
        },
        "optimized": {
            "delta_angles_deg": final_angles.tolist(),
            "delta_quaternion_wxyz": final_quat.tolist(),
            "delta_offsets_m": final_offsets.tolist(),
            "mean_distance_m": final_stats["mean_m"],
            "median_distance_m": final_stats["median_m"],
            "p90_distance_m": final_stats["p90_m"],
        },
        "delta": {
            "angles_deg": (final_angles - angles0).tolist(),
            "quaternion_wxyz": final_quat.tolist(),
            "rotation_distance_deg": quaternion_geodesic_distance_deg(final_quat, quat0),
            "offsets_m": (final_offsets - offsets0).tolist(),
            "mean_distance_m": float(final_mean - initial_mean),
            "mean_distance_improvement_pct": float((initial_mean - final_mean) / max(initial_mean, 1e-12) * 100.0),
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
            "translation_bounds_m": [float(translation_bounds[0]), 0.0, float(translation_bounds[2])],
            "fine_refinement": None,
            "polish_mode": bool(polish),
            "polish_refinement": polish_refinement,
            "sole_surface_y_m": y_target,
            "y_snap_offset_m": y_snap_offset,
            "footprint_area_m2": footprint_area,
        },
    }


PARTS: tuple[PartConfig, ...] = (
    PartConfig(
        part_name="r_upper_leg",
        position_file="right_leg_upper.txt",
        mesh_files=("sim_sea_2-5_r_thigh_prt-binary.stl",),
        mesh_pos=((0.043709, 0.0701, 0.240813),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_lower_leg",
        position_file="right_leg_lower.txt",
        mesh_files=("sim_sea_2-5_r_shank_prt-binary.stl",),
        mesh_pos=((0.043709, 0.0701, 0.386638),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_upper_leg",
        position_file="left_leg_upper.txt",
        mesh_files=("sim_sea_2-5_l_thigh_prt-binary.stl",),
        mesh_pos=((0.0437091, -0.0701, 0.240813),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_lower_leg",
        position_file="left_leg_lower.txt",
        mesh_files=("sim_sea_2-5_l_shank_prt-binary.stl",),
        mesh_pos=((0.0437091, -0.0701, 0.386638),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_upper_arm",
        position_file="right_arm.txt",
        mesh_files=("sim_sea_2-5_r_elbow_prt-binary.stl",),
        mesh_pos=((0.131934, 0.0, -0.0353516),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=True,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, -32.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
            ((-0.08, 0.0015, 0.012), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_forearm",
        position_file="right_forearm_V2.txt",
        mesh_files=("sim_sea_2-5_r_forearm_prt-binary.stl",),
        mesh_pos=((0.296887, 0.0, -0.0795506),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=True,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 90.0)),
            ((0.0, 0.0, 0.0), (0.0, 78.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
            ((-0.05, 0.0, -0.0015), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_upper_arm",
        position_file="left_arm.txt",
        mesh_files=("sim_sea_2-5_l_elbow_prt-binary.stl",),
        mesh_pos=((-0.131901, 0.0, -0.0353428),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=True,
        manual_steps=(
            ((0.0, 0.0, 0.0), (-270.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, -148.0, 0.0)),
            ((0.08, 0.0015, 0.012), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_forearm",
        position_file="left_forearm_V2.txt",
        mesh_files=("sim_sea_2-5_l_forearm_prt-binary.stl",),
        mesh_pos=((-0.296887, 0.0, -0.0795506),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=True,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, -90.0)),
            ((0.0, 0.0, 0.0), (0.0, 102.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
            ((0.05, -0.001, 0.0), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="torso",
        position_file="torso.txt",
        mesh_files=("chest_hull_3.stl",),
        mesh_pos=((0.0, 0.0928, -0.024189),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=True,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 90.0, 0.0)),
            ((0.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
            ((0.0, 0.06, 0.068), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_palm",
        position_file="right_hand_V2_1.txt",
        mesh_files=("col_RightHandPalm.stl",),
        mesh_pos=((0.00271607, -0.0015568, -0.00248235),),
        mesh_quat_wxyz=((0.701057, -0.701057, 0.0922959, 0.0922959),),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 180.0)),
            ((0.0, 0.0, 0.0), (90.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, 15.0, 0.0)),
            ((-0.055, -0.005, 0.02), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_palm",
        position_file="left_hand_V2_1.txt",
        mesh_files=("col_LeftHandPalm.stl",),
        mesh_pos=((-0.00271607, -0.0015568, -0.00248235),),
        mesh_quat_wxyz=((0.701057, 0.701057, -0.0922959, 0.0922959),),
        rebase=False,
        manual_steps=(
            ((0.0, 0.0, 0.0), (-90.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, -15.0, 0.0)),
            ((0.055, -0.005, 0.02), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="l_foot",
        position_file="left_foot.txt",
        mesh_files=("sim_sea_2-5_l_sole_prt-binary.stl",),
        mesh_pos=((0.0, 0.0, 0.0),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=False,
        include_to_model=False,
        optimizer_method=SOLE_2D_OPTIMIZER_METHOD,
        manual_steps=(
            ((0.0, 0.0, 0.0), (90.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, -90.0, 0.0)),
            ((0.0, -0.65, 0.0), (0.0, 0.0, 0.0)),
            ((0.07, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, -0.05), (0.0, 0.0, 0.0)),
        ),
    ),
    PartConfig(
        part_name="r_foot",
        position_file="right_foot.txt",
        mesh_files=("sim_sea_2-5_r_sole_prt-binary.stl",),
        mesh_pos=((0.0, 0.0, 0.0),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=False,
        include_to_model=False,
        optimizer_method=SOLE_2D_OPTIMIZER_METHOD,
        manual_steps=(
            ((0.0, 0.0, 0.0), (90.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (0.0, -90.0, 0.0)),
            ((0.0, -0.65, 0.0), (0.0, 0.0, 0.0)),
            ((-0.07, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, -0.05), (0.0, 0.0, 0.0)),
        ),
    ),
)

PARTS_BY_NAME: dict[str, PartConfig] = {cfg.part_name: cfg for cfg in PARTS}


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
            pos = rotate_position(pos=pos, offsets=offsets, angle_degrees=angles)
        pos = apply_quaternion_transform(pos=pos, offsets=delta_offsets_m, quat_wxyz=delta_quat_wxyz)
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
    taxels = validate_taxel_data(calibration, taxel2repr if taxel2repr else None)
    if rebase:
        taxels = rebase_coordinate_system(taxels)
    points = np.array([np.array(t[0], dtype=float) for t in taxels], dtype=float)
    return points


def load_mesh(
    mesh_files: tuple[str, ...],
    mesh_pos: tuple[Vector3, ...],
    mesh_quat_wxyz: tuple[QuaternionWXYZ, ...],
) -> trimesh.Trimesh:
    """Load one or more mesh files into the common body-part reference frame."""

    if not (len(mesh_files) == len(mesh_pos) == len(mesh_quat_wxyz)):
        raise ValueError("mesh_files, mesh_pos, and mesh_quat_wxyz must have the same length")

    meshes: list[trimesh.Trimesh] = []
    for mesh_file in mesh_files:
        mesh_path = MESH_DIR / mesh_file
        loaded = trimesh.load(mesh_path, force="mesh")
        if isinstance(loaded, trimesh.Trimesh):
            meshes.append(loaded)
        else:
            scene = cast(trimesh.Scene, loaded)
            meshes.extend([m for m in scene.dump() if isinstance(m, trimesh.Trimesh)])
    if not meshes:
        raise RuntimeError(f"No valid meshes loaded from: {mesh_files}")

    # Most sim_* meshes are exported in millimeters, while taxel coordinates are in meters.
    # The palm collision meshes in this repo are already in meters.
    scales = []
    scaled_meshes: list[trimesh.Trimesh] = []
    for mesh_file, mesh, pos, quat in zip(mesh_files, meshes, mesh_pos, mesh_quat_wxyz):
        mesh_local = mesh.copy()
        # Auto-detect mm-exported meshes by raw coordinate magnitude.
        max_abs = float(np.max(np.abs(mesh_local.vertices)))
        scale = 0.001 if max_abs > 2.0 else 1.0
        mesh_local.apply_scale(scale)

        # Apply geom orientation/translation from the MuJoCo model body frame.
        rot4 = trimesh.transformations.quaternion_matrix(np.array(quat, dtype=float))
        mesh_local.apply_transform(rot4)
        mesh_local.apply_translation(np.array(pos, dtype=float))

        scales.append(scale)
        scaled_meshes.append(mesh_local)

    print(f"mesh unit scales for {mesh_files}: {scales}")

    if len(meshes) == 1:
        return scaled_meshes[0]
    return trimesh.util.concatenate(scaled_meshes)


def build_distance_fn(mesh: trimesh.Trimesh) -> Callable[[np.ndarray], np.ndarray]:
    """Build a robust point-to-mesh distance function.

    Trimesh proximity queries are preferred when available. A KDTree built on a
    dense surface sample is kept as a fallback because proximity queries can be
    fragile for some meshes and environments.
    """

    sampled = mesh.sample(120000)
    kdtree = KDTree(sampled)

    def dist(points: np.ndarray) -> np.ndarray:
        try:
            _, distances, _ = trimesh.proximity.closest_point(mesh, points)
            return np.asarray(distances, dtype=float)
        except Exception:
            d, _ = kdtree.query(points, k=1)
            return np.asarray(d, dtype=float)

    return dist


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
            quat = normalize_quaternion_wxyz(np.array(quat_values, dtype=float))
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

    for idx, angle_seed in enumerate(config.delta_angle_seed_candidates):
        add_candidate(f"configured:{idx}", np.array(angle_seed, dtype=float), zero_offsets)

    if previous_seed is None or not config.include_previous_seed:
        return candidates

    prev_angles = previous_seed["angles"]
    prev_offsets = previous_seed["offsets"]
    add_candidate("previous:optimized", prev_angles, prev_offsets)

    if polish:
        return candidates

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

    points = load_taxel_points(POSITIONS_DIR / config.position_file, config.rebase)
    mesh = load_mesh(config.mesh_files, config.mesh_pos, config.mesh_quat_wxyz)
    if config.optimizer_method == SOLE_2D_OPTIMIZER_METHOD:
        return optimize_foot_sole_part(config, points, mesh, previous_seed_map, polish=polish)

    dist_fn = build_distance_fn(mesh)

    # We optimize a final quaternion perturbation applied on top of the exact manual chain.
    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    angles0 = quat_wxyz_to_euler_xyz_deg(quat0)
    offsets0 = np.zeros(3, dtype=float)
    x0 = np.hstack([quat0, offsets0])

    angle_bounds = np.array(config.delta_angle_bounds_deg, dtype=float)
    translation_bounds = np.array(config.delta_translation_bounds_m, dtype=float)

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
        return float(np.mean(distances))

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
        return {
            "mean_m": float(np.mean(distances)),
            "median_m": float(np.median(distances)),
            "p90_m": float(np.quantile(distances, 0.9)),
        }

    def objective(params: np.ndarray, quat_center: np.ndarray, offset_center: np.ndarray) -> float:
        """Robust objective with soft regularization around the current seed hypothesis."""

        quat, offsets, _ = unpack_params(params)
        transformed = apply_part_transform(
            points,
            config.part_name,
            delta_offsets_m=offsets,
            delta_quat_wxyz=quat,
        )
        distances = dist_fn(transformed)
        q = np.quantile(distances, 0.9)
        trimmed = distances[distances <= q]
        mean_trimmed = float(np.mean(trimmed)) if len(trimmed) else float(np.mean(distances))

        # Soft regularization keeps each run near its local seed hypothesis.
        rot_dist_deg = quaternion_geodesic_distance_deg(quat, quat_center)
        rot_reg = (rot_dist_deg / 10.0) ** 2
        trans_reg = np.sum(((offsets - offset_center) / 0.01) ** 2)
        return float(mean_trimmed + 0.001 * rot_reg + 0.001 * trans_reg)

    initial_stats = distance_stats(x0)
    initial_mean = float(initial_stats["mean_m"])
    print(f"[{config.part_name}] initial mean distance={initial_mean:.6f} m")

    previous_seed = previous_seed_map.get(config.part_name)
    seed_candidates = build_seed_candidates(config, previous_seed, polish=polish)

    seed_results: list[tuple[float, np.ndarray, Any, np.ndarray, np.ndarray, np.ndarray, str]] = []

    for seed_idx, (seed_label, seed_angles, seed_offsets) in enumerate(seed_candidates):
        print(f"\n[{config.part_name}] Trying seed {seed_idx}/{len(seed_candidates)}: {seed_label}")
        print(f"    angles: {seed_angles.tolist()} offsets: {seed_offsets.tolist()}", flush=True)
        seed_quat = euler_xyz_deg_to_quat_wxyz(seed_angles)
        x_seed = np.hstack([seed_quat, seed_offsets])
        bounds = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (seed_offsets[0] - translation_bounds[0], seed_offsets[0] + translation_bounds[0]),
            (seed_offsets[1] - translation_bounds[1], seed_offsets[1] + translation_bounds[1]),
            (seed_offsets[2] - translation_bounds[2], seed_offsets[2] + translation_bounds[2]),
        ]

        x_start = x_seed

        iteration = {"count": 0}
        best_so_far = {"score": mean_distance(x_start), "params": x_start.copy()}

        def callback(xk: np.ndarray) -> None:
            iteration["count"] += 1
            md = mean_distance(xk)
            print(
                f"[{config.part_name}] seed={seed_idx} ({seed_label}) iter={iteration['count']:02d} mean_distance={md:.6f} m",
                flush=True,
            )
            if md < best_so_far["score"]:
                best_so_far["score"] = md
                best_so_far["params"] = xk.copy()

        primary_maxiter = 500 if polish else 200
        primary_ftol = 1e-16 if polish else 1e-12
        result = minimize(
            lambda p: objective(p, seed_quat, seed_offsets),
            x_start,
            method=LOCAL_OPTIMIZER_METHOD,
            bounds=bounds,
            callback=callback,
            options={"maxiter": primary_maxiter, "ftol": primary_ftol, "gtol": 1e-12},
        )

        # Use best-so-far parameters, not just final iterate
        x_candidate = best_so_far["params"]
        score = best_so_far["score"]
        seed_results.append((score, x_candidate, result, seed_angles, seed_quat, seed_offsets, seed_label))

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
        angle_jitters = tuple(np.array(j, dtype=float) for j in config.local_refine_angle_jitter_candidates)

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
                (x_final[4] - translation_window_local[0], x_final[4] + translation_window_local[0]),
                (x_final[5] - translation_window_local[1], x_final[5] + translation_window_local[1]),
                (x_final[6] - translation_window_local[2], x_final[6] + translation_window_local[2]),
            ]

            fine_maxiter = 1000 if polish else 200
            fine_ftol = 1e-20 if polish else 1e-15
            result_local = minimize(
                lambda p: objective(p, normalize_quaternion_wxyz(x_final[:4]), x_final[4:]),
                x_seed_local,
                method=LOCAL_OPTIMIZER_METHOD,
                bounds=bounds_local,
                options={"maxiter": fine_maxiter, "ftol": fine_ftol, "gtol": 1e-16},
            )
            x_candidate_local = result_local.x if result_local.success else x_seed_local
            score_local = mean_distance(x_candidate_local)
            local_results.append((score_local, x_candidate_local, result_local, angle_jitter))
            print(
                f"[{config.part_name}] fine seed={local_idx} jitter={angle_jitter.tolist()} mean_distance={score_local:.6f} m"
            )

        local_results.sort(key=lambda x: x[0])
        best_local_score, best_local_x, best_local_result, best_local_jitter = local_results[0]
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
        translation_window_polish = np.minimum(translation_bounds, np.array([0.001, 0.001, 0.001], dtype=float))
        bounds_polish = [
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (x_final[4] - translation_window_polish[0], x_final[4] + translation_window_polish[0]),
            (x_final[5] - translation_window_polish[1], x_final[5] + translation_window_polish[1]),
            (x_final[6] - translation_window_polish[2], x_final[6] + translation_window_polish[2]),
        ]

        polish_before = mean_distance(x_final)
        result_polish = minimize(
            lambda p: objective(p, normalize_quaternion_wxyz(x_final[:4]), x_final[4:]),
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
        },
        "optimized": {
            "delta_angles_deg": final_angles.tolist(),
            "delta_quaternion_wxyz": final_quat.tolist(),
            "delta_offsets_m": final_offsets.tolist(),
            "mean_distance_m": final_stats["mean_m"],
            "median_distance_m": final_stats["median_m"],
            "p90_distance_m": final_stats["p90_m"],
        },
        "delta": {
            "angles_deg": (final_angles - angles0).tolist(),
            "quaternion_wxyz": final_quat.tolist(),
            "rotation_distance_deg": quaternion_geodesic_distance_deg(final_quat, quat0),
            "offsets_m": (final_offsets - offsets0).tolist(),
            "mean_distance_m": float(final_mean - initial_mean),
            "mean_distance_improvement_pct": float((initial_mean - final_mean) / max(initial_mean, 1e-12) * 100.0),
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
        lines.append(f"  delta angles deg (around manual): {init['delta_angles_deg']} -> {opt['delta_angles_deg']}")
        if opt.get("delta_quaternion_wxyz") is not None:
            lines.append(f"  delta quaternion wxyz (around manual): {opt['delta_quaternion_wxyz']}")
        lines.append(f"  delta offsets m (around manual): {init['delta_offsets_m']} -> {opt['delta_offsets_m']}")
        lines.append(f"  delta angles deg: {delta['angles_deg']}")
        if delta.get("rotation_distance_deg") is not None:
            lines.append(f"  delta rotation distance deg: {delta['rotation_distance_deg']}")
        lines.append(f"  delta offsets m: {delta['offsets_m']}")
        optimizer = cast(dict[str, Any], r["optimizer"])
        lines.append(f"  optimizer method: {optimizer.get('optimizer_method', LOCAL_OPTIMIZER_METHOD)}")
        lines.append(f"  solver: {optimizer.get('solver', LOCAL_OPTIMIZER_METHOD)}")
        lines.append(f"  seed count: {optimizer['seed_count']}")
        lines.append(f"  selected seed label: {optimizer['selected_seed_label']}")
        lines.append(f"  selected seed angles deg: {optimizer['selected_seed_angles_deg']}")
        if optimizer.get("selected_seed_quaternion_wxyz") is not None:
            lines.append(f"  selected seed quaternion wxyz: {optimizer['selected_seed_quaternion_wxyz']}")
        lines.append(f"  selected seed offsets m: {optimizer['selected_seed_offsets_m']}")
        lines.append(f"  angle bounds deg: {optimizer['angle_bounds_deg']}")
        lines.append(f"  translation bounds m: {optimizer['translation_bounds_m']}")
        if optimizer.get("fine_refinement") is not None:
            fr = cast(dict[str, Any], optimizer["fine_refinement"])
            lines.append(f"  fine refine candidates: {fr['candidate_count']}")
            lines.append(f"  fine refine selected jitter deg: {fr['selected_jitter_deg']}")
            lines.append(
                f"  fine refine mean distance: {fr['pre_refine_mean_distance_m']:.6f} -> {fr['post_refine_mean_distance_m']:.6f} m"
            )
            lines.append(f"  fine refine angle window deg: {fr['angle_window_deg']}")
            lines.append(f"  fine refine translation window m: {fr['translation_window_m']}")
        lines.append(f"  polish mode: {optimizer.get('polish_mode', False)}")
        if optimizer.get("polish_refinement") is not None:
            pr = cast(dict[str, Any], optimizer["polish_refinement"])
            lines.append(
                f"  polish mean distance: {pr['pre_polish_mean_distance_m']:.6f} -> {pr['post_polish_mean_distance_m']:.6f} m"
            )
            if pr.get("angle_window_deg") is not None:
                lines.append(f"  polish angle window deg: {pr['angle_window_deg']}")
            if pr.get("rotation_space") is not None:
                lines.append(f"  polish rotation space: {pr['rotation_space']}")
            lines.append(f"  polish translation window m: {pr['translation_window_m']}")
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selective optimization and polishing."""

    parser = argparse.ArgumentParser(description="Optimize taxel patch alignment against body meshes.")
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
    if "inlcude_to_model" not in entry:
        entry["inlcude_to_model"] = bool(cfg.include_to_model)
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
    selected_parts = set(args.parts) if args.parts else None
    previous_seed_map = load_previous_optimized_seeds()
    existing_results = load_existing_results()
    existing_results = [_upgrade_result_entry_schema(entry) for entry in existing_results]
    result_by_part = {cast(str, entry["part"]): entry for entry in existing_results}

    for part_cfg in PARTS:
        if selected_parts is not None and part_cfg.part_name not in selected_parts:
            continue
        print(f"\n=== Optimizing {part_cfg.part_name} ===")
        new_result = optimize_part(part_cfg, previous_seed_map, polish=args.polish)

        # Only update if new result is better than existing best (including initialized values)
        new_mean = _result_mean_distance(new_result)
        if part_cfg.part_name not in result_by_part:
            result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(new_result)
        else:
            best_mean = _result_mean_distance(result_by_part[part_cfg.part_name])
            if new_mean < best_mean:
                result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(new_result)
                print(f"  improved mean_distance {best_mean:.6f} m -> {new_mean:.6f} m, updating stored result.")
            else:
                result_by_part[part_cfg.part_name] = _upgrade_result_entry_schema(result_by_part[part_cfg.part_name])
                print(f"  mean_distance {new_mean:.6f} m >= best {best_mean:.6f} m, keeping previous result.")

    results = [result_by_part[part_cfg.part_name] for part_cfg in PARTS if part_cfg.part_name in result_by_part]

    write_reports(results)
    print("\nOptimization finished.")
    print(f"JSON report: {REPORT_JSON}")
    print(f"Text report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
