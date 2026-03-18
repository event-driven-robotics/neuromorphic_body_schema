"""
Optimize taxel patch alignment against body meshes.

This script starts from the current hand-tuned transforms used in
include_skin_to_mujoco_model.py and refines them by minimizing mean
point-to-surface distance to selected STL meshes.

Run:
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import KDTree

from include_skin_to_mujoco_model import (
    read_calibration_data,
    read_taxel2repr_data,
    rebase_coordinate_system,
    rotate_position,
    validate_taxel_data,
)

@dataclass(frozen=True)
class PartConfig:
    part_name: str
    position_file: str
    mesh_files: tuple[str, ...]
    mesh_pos: tuple[tuple[float, float, float], ...]
    mesh_quat_wxyz: tuple[tuple[float, float, float, float], ...]
    rebase: bool
    delta_angle_seed_candidates: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    delta_angle_bounds_deg: tuple[float, float, float] = (25.0, 25.0, 25.0)
    delta_translation_bounds_m: tuple[float, float, float] = (0.03, 0.03, 0.03)


ROOT = Path(__file__).resolve().parents[2]
POSITIONS_DIR = ROOT / "neuromorphic_body_schema" / "include_taxels" / "positions"
MESH_DIR = ROOT / "neuromorphic_body_schema" / "meshes" / "iCub"
REPORT_JSON = ROOT / "neuromorphic_body_schema" / "include_taxels" / "taxel_alignment_optimization_report.json"
REPORT_TXT = ROOT / "neuromorphic_body_schema" / "include_taxels" / "taxel_alignment_optimization_report.txt"



def _angle_seed_grid(
    x_values: tuple[float, ...],
    y_values: tuple[float, ...] = (0.0,),
    z_values: tuple[float, ...] = (0.0,),
) -> tuple[tuple[float, float, float], ...]:
    return tuple((x, y, z) for x in x_values for y in y_values for z in z_values)

# Expanded grid for upper arms: 0, 45, 90 degrees for each axis
UPPER_ARM_ANGLE_GRID = _angle_seed_grid((0.0, 45.0, 90.0), (0.0, 45.0, 90.0), (0.0, 45.0, 90.0))

MANUAL_STEPS: dict[str, tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]] = {
    "r_upper_leg": (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ),
    "r_lower_leg": (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ),
    "l_upper_leg": (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ),
    "l_lower_leg": (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ),
    "r_upper_arm": (
        ((0.0, 0.0, 0.0), (0.0, -32.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
        ((-0.08, 0.0015, 0.012), (0.0, 0.0, 0.0)),
    ),
    "r_forearm": (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 90.0)),
        ((0.0, 0.0, 0.0), (0.0, 78.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
        ((-0.05, 0.0, -0.0015), (0.0, 0.0, 0.0)),
    ),
    "l_upper_arm": (
        ((0.0, 0.0, 0.0), (-270.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, -148.0, 0.0)),
        ((0.08, 0.0015, 0.012), (0.0, 0.0, 0.0)),
    ),
    "l_forearm": (
        ((0.0, 0.0, 0.0), (0.0, 0.0, -90.0)),
        ((0.0, 0.0, 0.0), (0.0, 102.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
        ((0.05, -0.001, 0.0), (0.0, 0.0, 0.0)),
    ),
    "torso": (
        ((0.0, 0.0, 0.0), (0.0, 90.0, 0.0)),
        ((0.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
        ((0.0, 0.06, 0.068), (0.0, 0.0, 0.0)),
    ),
    "r_palm": (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 180.0)),
        ((0.0, 0.0, 0.0), (90.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 15.0, 0.0)),
        ((-0.055, -0.005, 0.02), (0.0, 0.0, 0.0)),
    ),
    "l_palm": (
        ((0.0, 0.0, 0.0), (-90.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, -15.0, 0.0)),
        ((0.055, -0.005, 0.02), (0.0, 0.0, 0.0)),
    ),
}


PARTS: tuple[PartConfig, ...] = (
    PartConfig(
        part_name="r_upper_leg",
        position_file="right_leg_upper.txt",
        mesh_files=("sim_sea_2-5_r_thigh_prt-binary.stl",),
        mesh_pos=((0.043709, 0.0701, 0.240813),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
    ),
    PartConfig(
        part_name="r_lower_leg",
        position_file="right_leg_lower.txt",
        mesh_files=("sim_sea_2-5_r_shank_prt-binary.stl",),
        mesh_pos=((0.043709, 0.0701, 0.386638),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
    ),
    PartConfig(
        part_name="l_upper_leg",
        position_file="left_leg_upper.txt",
        mesh_files=("sim_sea_2-5_l_thigh_prt-binary.stl",),
        mesh_pos=((0.0437091, -0.0701, 0.240813),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
    ),
    PartConfig(
        part_name="l_lower_leg",
        position_file="left_leg_lower.txt",
        mesh_files=("sim_sea_2-5_l_shank_prt-binary.stl",),
        mesh_pos=((0.0437091, -0.0701, 0.386638),),
        mesh_quat_wxyz=((0.5, 0.5, 0.5, 0.5),),
        rebase=False,
    ),
    PartConfig(
        part_name="r_upper_arm",
        position_file="right_arm_mesh.txt",
        mesh_files=("sim_sea_2-5_r_upper_arm_prt-binary.stl", "sim_sea_2-5_r_elbow_prt-binary.stl"),
        mesh_pos=((0.131934, 0.0, -0.0353516), (0.131934, 0.0, -0.0353516)),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        rebase=True,
        delta_angle_seed_candidates=UPPER_ARM_ANGLE_GRID,
    ),
    PartConfig(
        part_name="r_forearm",
        position_file="right_forearm_V2.txt",
        mesh_files=("sim_sea_2-5_r_forearm_prt-binary.stl",),
        mesh_pos=((0.296887, 0.0, -0.0795506),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=True,
        # Sweep a wider x-flip family and let prior optimized results contribute
        # additional local starts in the same region.
        delta_angle_seed_candidates=_angle_seed_grid(
            (130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0)
        ),
        delta_angle_bounds_deg=(15.0, 8.0, 8.0),
        delta_translation_bounds_m=(0.008, 0.008, 0.008),
    ),
    PartConfig(
        part_name="l_upper_arm",
        position_file="left_arm_mesh.txt",
        mesh_files=("sim_sea_2-5_l_upper_arm_prt-binary.stl", "sim_sea_2-5_l_elbow_prt-binary.stl"),
        mesh_pos=((-0.131901, 0.0, -0.0353428), (-0.131901, 0.0, -0.0353428)),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        rebase=True,
        delta_angle_seed_candidates=UPPER_ARM_ANGLE_GRID,
    ),
    PartConfig(
        part_name="l_forearm",
        position_file="left_forearm_V2.txt",
        mesh_files=("sim_sea_2-5_l_forearm_prt-binary.stl",),
        mesh_pos=((-0.296887, 0.0, -0.0795506),),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0),),
        rebase=True,
    ),
    PartConfig(
        part_name="torso",
        position_file="torso.txt",
        mesh_files=("chest_hull_1.stl", "chest_hull_2.stl", "chest_hull_3.stl"),
        mesh_pos=((0.0, 0.0928, -0.024189), (0.0, 0.0928, -0.024189), (0.0, 0.0928, -0.024189)),
        mesh_quat_wxyz=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        rebase=True,
    ),
    PartConfig(
        part_name="r_palm",
        position_file="right_hand_V2_1.txt",
        mesh_files=("col_RightHandPalm.stl",),
        mesh_pos=((0.00271607, -0.0015568, -0.00248235),),
        mesh_quat_wxyz=((0.701057, -0.701057, 0.0922959, 0.0922959),),
        rebase=False,
    ),
    PartConfig(
        part_name="l_palm",
        position_file="left_hand_V2_1.txt",
        mesh_files=("col_LeftHandPalm.stl",),
        mesh_pos=((-0.00271607, -0.0015568, -0.00248235),),
        mesh_quat_wxyz=((0.701057, 0.701057, -0.0922959, 0.0922959),),
        rebase=False,
    ),
)


def apply_part_transform(
    points: np.ndarray,
    part_name: str,
    delta_angles_deg: np.ndarray | None = None,
    delta_offsets_m: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the same transform pipeline as include_skin_to_mujoco_model.py, plus an optional final delta."""
    if delta_angles_deg is None:
        delta_angles_deg = np.zeros(3, dtype=float)
    if delta_offsets_m is None:
        delta_offsets_m = np.zeros(3, dtype=float)

    out = np.empty_like(points)
    steps = MANUAL_STEPS[part_name]
    for i in range(points.shape[0]):
        pos = np.array(points[i], dtype=float)
        for offsets, angles in steps:
            pos = rotate_position(pos=pos, offsets=offsets, angle_degrees=angles)
        pos = rotate_position(
            pos=pos,
            offsets=delta_offsets_m.tolist(),
            angle_degrees=delta_angles_deg.tolist(),
        )
        out[i] = pos
    return out


def load_taxel_points(position_file: Path, rebase: bool) -> np.ndarray:
    calibration = read_calibration_data(str(position_file))
    taxel2repr = read_taxel2repr_data(str(position_file))
    taxels = validate_taxel_data(calibration, taxel2repr if taxel2repr else None)
    if rebase:
        taxels = rebase_coordinate_system(taxels)
    points = np.array([np.array(t[0], dtype=float) for t in taxels], dtype=float)
    return points


def load_mesh(
    mesh_files: tuple[str, ...],
    mesh_pos: tuple[tuple[float, float, float], ...],
    mesh_quat_wxyz: tuple[tuple[float, float, float, float], ...],
) -> trimesh.Trimesh:
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
    """Use trimesh proximity when possible, otherwise sampled-surface KDTree."""
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


def load_previous_optimized_seeds() -> dict[str, dict[str, np.ndarray]]:
    if not REPORT_JSON.exists():
        return {}

    try:
        raw = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}

    seeds: dict[str, dict[str, np.ndarray]] = {}
    for entry in raw:
        if not isinstance(entry, dict) or "part" not in entry or "optimized" not in entry:
            continue

        part = str(entry["part"])
        optimized = cast(dict[str, Any], entry["optimized"])
        angle_values = optimized.get("delta_angles_deg")
        offset_values = optimized.get("delta_offsets_m")
        if angle_values is None or offset_values is None:
            continue

        seeds[part] = {
            "angles": np.array(angle_values, dtype=float),
            "offsets": np.array(offset_values, dtype=float),
        }

    return seeds


def build_seed_candidates(
    config: PartConfig,
    previous_seed: dict[str, np.ndarray] | None,
    polish: bool = False,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()

    def add_candidate(label: str, angles: np.ndarray, offsets: np.ndarray) -> None:
        key = tuple(np.round(np.hstack([angles, offsets]), 6).tolist())
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, angles.copy(), offsets.copy()))

    zero_offsets = np.zeros(3, dtype=float)

    if polish and previous_seed is not None:
        # In polish mode, start from the best known solution first.
        add_candidate("previous:optimized", previous_seed["angles"], previous_seed["offsets"])


    # Special logic for l_upper_arm: focused grid around previous best with specified rotations and translations
    if config.part_name == "l_upper_arm" and previous_seed is not None and not polish:
        prev_angles = previous_seed["angles"]
        prev_offsets = previous_seed["offsets"]
        # Best so far
        # rot_x_vals = [-15.0]
        # rot_y_vals = [95.0]
        # rot_z_vals = [170.0]
        # trans_x_vals = [-0.04]  # moves along z
        # trans_y_vals = [0.0]
        # trans_z_vals = [-0.09]  # moves along x

        # Experimental
        rot_x_vals = [-6.000012628466136]
        rot_y_vals = [114.99998344324177]
        rot_z_vals = [168.2978092632269]
        trans_x_vals = [-0.07, -0.06, -0.05, -0.04, -0.03]  # moves along z
        trans_y_vals = [-0.0032137035524437718]
        trans_z_vals = [-0.08003560586316362]  # moves along x

        for dx in rot_x_vals:
            for dy in rot_y_vals:
                for dz in rot_z_vals:
                    for tx in trans_x_vals:
                        for ty in trans_y_vals:
                            for tz in trans_z_vals:
                                # use previous best as starting point
                                # angles = prev_angles + np.array([dx, dy, dz], dtype=float)
                                # offsets = prev_offsets + np.array([tx, ty, tz], dtype=float)
                                # always start from original orientation
                                angles = np.array([dx, dy, dz], dtype=float)
                                offsets = np.array([tx, ty, tz], dtype=float)
                                add_candidate(
                                    f"l_upper_arm:refine_rotx{dx}_roty{dy}_rotz{dz}_x{tx}_y{ty}_z{tz}",
                                    angles,
                                    offsets
                                )
    else:
        for idx, angle_seed in enumerate(config.delta_angle_seed_candidates):
            add_candidate(f"configured:{idx}", np.array(angle_seed, dtype=float), zero_offsets)

    if previous_seed is not None and config.part_name == "r_forearm":
        prev_angles = previous_seed["angles"]
        prev_offsets = previous_seed["offsets"]
        add_candidate("previous:optimized", prev_angles, prev_offsets)

        for delta_x in (-30.0, -20.0, -10.0, 10.0, 20.0, 30.0):
            add_candidate(
                f"previous:x{delta_x:+.0f}",
                prev_angles + np.array([delta_x, 0.0, 0.0], dtype=float),
                prev_offsets,
            )
        for delta_y in (-10.0, -5.0, 5.0, 10.0):
            add_candidate(
                f"previous:y{delta_y:+.0f}",
                prev_angles + np.array([0.0, delta_y, 0.0], dtype=float),
                prev_offsets,
            )
        for delta_z in (-10.0, -5.0, 5.0, 10.0):
            add_candidate(
                f"previous:z{delta_z:+.0f}",
                prev_angles + np.array([0.0, 0.0, delta_z], dtype=float),
                prev_offsets,
            )
        for delta_xyz in (
            np.array([10.0, -5.0, 0.0], dtype=float),
            np.array([10.0, 5.0, 0.0], dtype=float),
            np.array([-10.0, -5.0, 0.0], dtype=float),
            np.array([-10.0, 5.0, 0.0], dtype=float),
            np.array([0.0, -5.0, -5.0], dtype=float),
            np.array([0.0, -5.0, 5.0], dtype=float),
            np.array([0.0, 5.0, -5.0], dtype=float),
            np.array([0.0, 5.0, 5.0], dtype=float),
        ):
            add_candidate(
                f"previous:combo_{delta_xyz[0]:+.0f}_{delta_xyz[1]:+.0f}_{delta_xyz[2]:+.0f}",
                prev_angles + delta_xyz,
                prev_offsets,
            )

        for delta_offset in (
            np.array([0.002, 0.0, 0.0], dtype=float),
            np.array([-0.002, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.002, 0.0], dtype=float),
            np.array([0.0, -0.002, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.002], dtype=float),
            np.array([0.0, 0.0, -0.002], dtype=float),
            np.array([0.002, 0.002, 0.0], dtype=float),
            np.array([0.0, 0.002, -0.002], dtype=float),
        ):
            add_candidate(
                f"previous:offset_{delta_offset[0]:+.3f}_{delta_offset[1]:+.3f}_{delta_offset[2]:+.3f}",
                prev_angles,
                prev_offsets + delta_offset,
            )

    return candidates


def optimize_part(
    config: PartConfig,
    previous_seed_map: dict[str, dict[str, np.ndarray]],
    polish: bool = False,
) -> dict[str, Any]:
    points = load_taxel_points(POSITIONS_DIR / config.position_file, config.rebase)
    mesh = load_mesh(config.mesh_files, config.mesh_pos, config.mesh_quat_wxyz)
    dist_fn = build_distance_fn(mesh)

    # We optimize a final perturbation applied on top of the exact manual chain.
    angles0 = np.zeros(3, dtype=float)
    offsets0 = np.zeros(3, dtype=float)
    x0 = np.zeros(6, dtype=float)

    angle_bounds = np.array(config.delta_angle_bounds_deg, dtype=float)
    translation_bounds = np.array(config.delta_translation_bounds_m, dtype=float)

    def mean_distance(params: np.ndarray) -> float:
        transformed = apply_part_transform(points, config.part_name, params[:3], params[3:])
        distances = dist_fn(transformed)
        return float(np.mean(distances))

    def distance_stats(params: np.ndarray) -> dict[str, float]:
        transformed = apply_part_transform(points, config.part_name, params[:3], params[3:])
        distances = dist_fn(transformed)
        return {
            "mean_m": float(np.mean(distances)),
            "median_m": float(np.median(distances)),
            "p90_m": float(np.quantile(distances, 0.9)),
        }

    def objective(params: np.ndarray, angle_center: np.ndarray, offset_center: np.ndarray) -> float:
        transformed = apply_part_transform(points, config.part_name, params[:3], params[3:])
        distances = dist_fn(transformed)
        q = np.quantile(distances, 0.9)
        trimmed = distances[distances <= q]
        mean_trimmed = float(np.mean(trimmed)) if len(trimmed) else float(np.mean(distances))

        # Soft regularization keeps each run near its local seed hypothesis.
        angle_reg = np.sum(((params[:3] - angle_center) / 10.0) ** 2)
        trans_reg = np.sum(((params[3:] - offset_center) / 0.01) ** 2)
        return float(mean_trimmed + 0.001 * angle_reg + 0.001 * trans_reg)

    initial_stats = distance_stats(x0)
    initial_mean = float(initial_stats["mean_m"])
    print(f"[{config.part_name}] initial mean distance={initial_mean:.6f} m")

    previous_seed = previous_seed_map.get(config.part_name)
    if polish and previous_seed is not None:
        # In polish mode, only use the previous optimized result as the seed
        seed_candidates = [("previous:optimized", previous_seed["angles"], previous_seed["offsets"])]
    else:
        # In normal mode, use all seed candidates
        seed_candidates = build_seed_candidates(config, previous_seed, polish=polish)

    seed_results: list[tuple[float, np.ndarray, Any, np.ndarray, np.ndarray, str]] = []


    for seed_idx, (seed_label, seed_angles, seed_offsets) in enumerate(seed_candidates):
        print(f"\n[{config.part_name}] Trying seed {seed_idx}/{len(seed_candidates)}: {seed_label}")
        print(f"    angles: {seed_angles.tolist()} offsets: {seed_offsets.tolist()}")
        x_seed = np.hstack([seed_angles, seed_offsets])
        bounds = [
            (seed_angles[0] - angle_bounds[0], seed_angles[0] + angle_bounds[0]),
            (seed_angles[1] - angle_bounds[1], seed_angles[1] + angle_bounds[1]),
            (seed_angles[2] - angle_bounds[2], seed_angles[2] + angle_bounds[2]),
            (seed_offsets[0] - translation_bounds[0], seed_offsets[0] + translation_bounds[0]),
            (seed_offsets[1] - translation_bounds[1], seed_offsets[1] + translation_bounds[1]),
            (seed_offsets[2] - translation_bounds[2], seed_offsets[2] + translation_bounds[2]),
        ]

        iteration = {"count": 0}
        best_so_far = {"score": float("inf"), "params": x_seed.copy()}

        def callback(xk: np.ndarray) -> None:
            iteration["count"] += 1
            md = mean_distance(xk)
            print(
                f"[{config.part_name}] seed={seed_idx} ({seed_label}) iter={iteration['count']:02d} mean_distance={md:.6f} m"
            )
            if md < best_so_far["score"]:
                best_so_far["score"] = md
                best_so_far["params"] = xk.copy()

        primary_maxiter = 220 if polish else 120
        primary_ftol = 1e-14 if polish else 1e-10
        result = minimize(
            lambda p: objective(p, seed_angles, seed_offsets),
            x_seed,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={"maxiter": primary_maxiter, "ftol": primary_ftol, "gtol": 1e-12},
        )

        # Use best-so-far parameters, not just final iterate
        x_candidate = best_so_far["params"]
        score = best_so_far["score"]
        seed_results.append((score, x_candidate, result, seed_angles, seed_offsets, seed_label))

    seed_results.sort(key=lambda x: x[0])
    _, x_final, result, selected_seed_angles, selected_seed_offsets, selected_seed_label = seed_results[0]

    fine_refinement: dict[str, Any] | None = None
    if config.part_name == "r_forearm":
        angle_window_local = np.array([6.0, 6.0, 6.0], dtype=float)
        translation_window_local = np.minimum(translation_bounds, np.array([0.002, 0.002, 0.002], dtype=float))
        angle_jitters = (
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([5.0, 0.0, 0.0], dtype=float),
            np.array([-5.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 5.0, 0.0], dtype=float),
            np.array([0.0, -5.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 5.0], dtype=float),
            np.array([0.0, 0.0, -5.0], dtype=float),
            np.array([5.0, -5.0, 0.0], dtype=float),
            np.array([5.0, 5.0, 0.0], dtype=float),
            np.array([0.0, -5.0, -5.0], dtype=float),
            np.array([0.0, -5.0, 5.0], dtype=float),
        )

        local_results: list[tuple[float, np.ndarray, Any, np.ndarray]] = []
        for local_idx, angle_jitter in enumerate(angle_jitters):
            x_seed_local = x_final.copy()
            x_seed_local[:3] = x_seed_local[:3] + angle_jitter

            bounds_local = [
                (x_seed_local[0] - angle_window_local[0], x_seed_local[0] + angle_window_local[0]),
                (x_seed_local[1] - angle_window_local[1], x_seed_local[1] + angle_window_local[1]),
                (x_seed_local[2] - angle_window_local[2], x_seed_local[2] + angle_window_local[2]),
                (x_final[3] - translation_window_local[0], x_final[3] + translation_window_local[0]),
                (x_final[4] - translation_window_local[1], x_final[4] + translation_window_local[1]),
                (x_final[5] - translation_window_local[2], x_final[5] + translation_window_local[2]),
            ]

            fine_maxiter = 140 if polish else 80
            fine_ftol = 1e-13 if polish else 1e-11
            result_local = minimize(
                lambda p: objective(p, x_final[:3], x_final[3:]),
                x_seed_local,
                method="L-BFGS-B",
                bounds=bounds_local,
                options={"maxiter": fine_maxiter, "ftol": fine_ftol, "gtol": 1e-12},
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
        angle_window_polish = np.minimum(angle_bounds, np.array([2.0, 2.0, 2.0], dtype=float))
        translation_window_polish = np.minimum(translation_bounds, np.array([0.001, 0.001, 0.001], dtype=float))
        bounds_polish = [
            (x_final[0] - angle_window_polish[0], x_final[0] + angle_window_polish[0]),
            (x_final[1] - angle_window_polish[1], x_final[1] + angle_window_polish[1]),
            (x_final[2] - angle_window_polish[2], x_final[2] + angle_window_polish[2]),
            (x_final[3] - translation_window_polish[0], x_final[3] + translation_window_polish[0]),
            (x_final[4] - translation_window_polish[1], x_final[4] + translation_window_polish[1]),
            (x_final[5] - translation_window_polish[2], x_final[5] + translation_window_polish[2]),
        ]

        polish_before = mean_distance(x_final)
        result_polish = minimize(
            lambda p: objective(p, x_final[:3], x_final[3:]),
            x_final,
            method="L-BFGS-B",
            bounds=bounds_polish,
            options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-13},
        )
        x_polished = result_polish.x if result_polish.success else x_final
        polish_after = mean_distance(x_polished)
        if polish_after < polish_before:
            x_final = x_polished
            result = result_polish

        polish_refinement = {
            "pre_polish_mean_distance_m": float(polish_before),
            "post_polish_mean_distance_m": float(mean_distance(x_final)),
            "angle_window_deg": angle_window_polish.tolist(),
            "translation_window_m": translation_window_polish.tolist(),
            "success": bool(result_polish.success),
        }

    final_stats = distance_stats(x_final)
    final_mean = float(final_stats["mean_m"])

    return {
        "part": config.part_name,
        "position_file": config.position_file,
        "mesh_files": list(config.mesh_files),
        "manual_steps": [
            {"offsets_m": list(step[0]), "angles_deg": list(step[1])}
            for step in MANUAL_STEPS[config.part_name]
        ],
        "initial": {
            "delta_angles_deg": angles0.tolist(),
            "delta_offsets_m": offsets0.tolist(),
            "mean_distance_m": initial_stats["mean_m"],
            "median_distance_m": initial_stats["median_m"],
            "p90_distance_m": initial_stats["p90_m"],
        },
        "optimized": {
            "delta_angles_deg": x_final[:3].tolist(),
            "delta_offsets_m": x_final[3:].tolist(),
            "mean_distance_m": final_stats["mean_m"],
            "median_distance_m": final_stats["median_m"],
            "p90_distance_m": final_stats["p90_m"],
        },
        "delta": {
            "angles_deg": (x_final[:3] - angles0).tolist(),
            "offsets_m": (x_final[3:] - offsets0).tolist(),
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
            "seed_count": len(seed_candidates),
            "selected_seed_label": selected_seed_label,
            "selected_seed_angles_deg": selected_seed_angles.tolist(),
            "selected_seed_offsets_m": selected_seed_offsets.tolist(),
            "angle_bounds_deg": angle_bounds.tolist(),
            "translation_bounds_m": translation_bounds.tolist(),
            "fine_refinement": fine_refinement,
            "polish_mode": bool(polish),
            "polish_refinement": polish_refinement,
        },
    }


def write_reports(results: list[dict[str, Any]]) -> None:
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
        lines.append(f"  delta offsets m (around manual): {init['delta_offsets_m']} -> {opt['delta_offsets_m']}")
        lines.append(f"  delta angles deg: {delta['angles_deg']}")
        lines.append(f"  delta offsets m: {delta['offsets_m']}")
        optimizer = cast(dict[str, Any], r["optimizer"])
        lines.append(f"  seed count: {optimizer['seed_count']}")
        lines.append(f"  selected seed label: {optimizer['selected_seed_label']}")
        lines.append(f"  selected seed angles deg: {optimizer['selected_seed_angles_deg']}")
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
            lines.append(f"  polish angle window deg: {pr['angle_window_deg']}")
            lines.append(f"  polish translation window m: {pr['translation_window_m']}")
            lines.append(f"  polish success: {pr['success']}")
        lines.append("")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def load_existing_results() -> list[dict[str, Any]]:
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


def main() -> None:
    args = parse_args()
    selected_parts = set(args.parts) if args.parts else None
    previous_seed_map = load_previous_optimized_seeds()
    existing_results = load_existing_results()
    result_by_part = {cast(str, entry["part"]): entry for entry in existing_results}

    for part_cfg in PARTS:
        if selected_parts is not None and part_cfg.part_name not in selected_parts:
            continue
        print(f"\n=== Optimizing {part_cfg.part_name} ===")
        result = optimize_part(part_cfg, previous_seed_map, polish=args.polish)
        result_by_part[part_cfg.part_name] = result

    results = [result_by_part[part_cfg.part_name] for part_cfg in PARTS if part_cfg.part_name in result_by_part]

    write_reports(results)
    print("\nOptimization finished.")
    print(f"JSON report: {REPORT_JSON}")
    print(f"Text report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
