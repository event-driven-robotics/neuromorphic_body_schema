"""Project legacy fingertip taxel layouts onto MuJoCo collision meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
MODEL_XML = ROOT / "neuromorphic_body_schema" / "models" / "icub_v2_full_body_improved.xml"

MjModel = getattr(mujoco, "MjModel")
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")
mjtGeom = getattr(mujoco, "mjtGeom")


LEGACY_FINGERTIP_TAXELS = np.array(
    [
        [12e-3, 3.25e-3, -3.25e-3],
        [12e-3, 6.5e-3, 0.0],
        [12e-3, 3.25e-3, 3.25e-3],
        [13.5e-3, -1.5e-3, 6.5e-3],
        [13.5e-3, -1.5e-3, -6.5e-3],
        [15e-3, 3.25e-3, 3.25e-3],
        [15e-3, 6.5e-3, 0.0],
        [15e-3, 3.25e-3, -3.25e-3],
        [19.625e-3, 1.625e-3, -3.25e-3],
        [21.25e-3, 3.25e-3, 0.0],
        [19.625e-3, 1.625e-3, 3.25e-3],
        [24.5e-3, 0.0, 0.0],
    ],
    dtype=float,
)

FINGERTIP_BODY_TO_PART = {
    "r_hand_thumb_3": "r_hand_thumb",
    "r_hand_index_3": "r_hand_index",
    "r_hand_middle_3": "r_hand_middle",
    "r_hand_ring_3": "r_hand_ring",
    "r_hand_little_3": "r_hand_little",
    "l_hand_thumb_3": "l_hand_thumb",
    "l_hand_index_3": "l_hand_index",
    "l_hand_middle_3": "l_hand_middle",
    "l_hand_ring_3": "l_hand_ring",
    "l_hand_little_3": "l_hand_little",
}

FINGERTIP_BODY_TO_COLLISION_MESH = {
    "r_hand_thumb_3": "col_RightThumb3",
    "r_hand_index_3": "col_RightIndex3",
    "r_hand_middle_3": "col_RightMiddle3",
    "r_hand_ring_3": "col_RightRing3",
    "r_hand_little_3": "col_RightLittle3",
    "l_hand_thumb_3": "col_LeftThumb3",
    "l_hand_index_3": "col_LeftIndex3",
    "l_hand_middle_3": "col_LeftMiddle3",
    "l_hand_ring_3": "col_LeftRing3",
    "l_hand_little_3": "col_LeftLittle3",
}


@dataclass(frozen=True)
class ProjectionResult:
    body_name: str
    part_name: str
    mesh: trimesh.Trimesh
    original_points: np.ndarray
    projected_points: np.ndarray
    movement_m: np.ndarray


_MODEL_CACHE: dict[Path, Any] = {}
_MESH_CACHE: dict[tuple[Path, str], trimesh.Trimesh] = {}


def _load_model(model_xml: Path = MODEL_XML) -> Any:
    model_xml = model_xml.resolve()
    if model_xml not in _MODEL_CACHE:
        _MODEL_CACHE[model_xml] = MjModel.from_xml_path(str(model_xml))
    return _MODEL_CACHE[model_xml]


def _quat_wxyz_to_rotation_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    q = np.array(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.eye(3, dtype=float)
    q = q / norm
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def load_fingertip_collision_mesh(
    body_name: str,
    model_xml: Path = MODEL_XML,
) -> trimesh.Trimesh:
    """Load a fingertip collision mesh in the local frame of its body."""

    model_xml = model_xml.resolve()
    cache_key = (model_xml, body_name)
    if cache_key in _MESH_CACHE:
        return _MESH_CACHE[cache_key].copy()

    mesh_name = FINGERTIP_BODY_TO_COLLISION_MESH.get(body_name)
    if mesh_name is None:
        raise ValueError(f"Unknown fingertip body: {body_name}")

    model = _load_model(model_xml)
    body_id = int(mj_name2id(model, mjtObj.mjOBJ_BODY, body_name))
    mesh_id = int(mj_name2id(model, mjtObj.mjOBJ_MESH, mesh_name))
    if body_id < 0:
        raise RuntimeError(f"MuJoCo body not found: {body_name}")
    if mesh_id < 0:
        raise RuntimeError(f"MuJoCo mesh not found: {mesh_name}")

    geom_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_type[geom_id]) == int(mjtGeom.mjGEOM_MESH)
        and int(model.geom_bodyid[geom_id]) == body_id
        and int(model.geom_dataid[geom_id]) == mesh_id
    ]
    if not geom_ids:
        raise RuntimeError(f"No collision geom for body={body_name}, mesh={mesh_name}")

    geom_id = int(geom_ids[0])
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = np.array(model.mesh_vert[vert_start:vert_start + vert_count], dtype=float)
    faces = np.array(model.mesh_face[face_start:face_start + face_count], dtype=int)
    rot = _quat_wxyz_to_rotation_matrix(model.geom_quat[geom_id])
    pos = np.array(model.geom_pos[geom_id], dtype=float)

    mesh = trimesh.Trimesh(vertices=(rot @ vertices.T).T + pos, faces=faces, process=False)
    _MESH_CACHE[cache_key] = mesh.copy()
    return mesh


def estimate_legacy_fingertip_normal(point: Sequence[float]) -> np.ndarray:
    """Estimate the old cylinder/sphere layout normal for a legacy taxel point."""

    p = np.array(point, dtype=float)
    yz_radius = float(np.linalg.norm(p[1:3]))
    if yz_radius <= 1e-9:
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
    elif p[0] > 0.018:
        normal = np.array([max(0.0, p[0] - 0.018), p[1], p[2]], dtype=float)
    else:
        normal = np.array([0.0, p[1], p[2]], dtype=float)

    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return normal / norm


def _project_one_point(
    mesh: trimesh.Trimesh,
    point: np.ndarray,
    normal_preference: np.ndarray,
    distance_slack_m: float,
) -> np.ndarray:
    triangles = mesh.triangles
    points = np.repeat(point.reshape(1, 3), len(triangles), axis=0)
    closest = trimesh.triangles.closest_point(triangles, points)
    distances = np.linalg.norm(closest - point, axis=1)
    min_distance = float(np.min(distances))

    candidate_mask = distances <= min_distance + distance_slack_m
    candidate_indices = np.flatnonzero(candidate_mask)
    normals = np.array(mesh.face_normals[candidate_indices], dtype=float)
    normal_scores = np.abs(normals @ normal_preference)
    candidate_distances = distances[candidate_indices]

    # Prefer the most normal-compatible nearby surface, then the closest point.
    order = np.lexsort((candidate_distances, -normal_scores))
    return np.array(closest[candidate_indices[int(order[0])]], dtype=float)


def project_points_to_fingertip_surface(
    body_name: str,
    points: np.ndarray,
    model_xml: Path = MODEL_XML,
    distance_slack_m: float = 0.0015,
) -> ProjectionResult:
    mesh = load_fingertip_collision_mesh(body_name, model_xml=model_xml)
    original = np.array(points, dtype=float)
    projected = np.vstack(
        [
            _project_one_point(
                mesh=mesh,
                point=point,
                normal_preference=estimate_legacy_fingertip_normal(point),
                distance_slack_m=distance_slack_m,
            )
            for point in original
        ]
    )
    movement = np.linalg.norm(projected - original, axis=1)
    return ProjectionResult(
        body_name=body_name,
        part_name=FINGERTIP_BODY_TO_PART[body_name],
        mesh=mesh,
        original_points=original,
        projected_points=projected,
        movement_m=movement,
    )


def project_legacy_fingertip_taxels(
    body_name: str,
    model_xml: Path = MODEL_XML,
    distance_slack_m: float = 0.0015,
) -> ProjectionResult:
    return project_points_to_fingertip_surface(
        body_name=body_name,
        points=LEGACY_FINGERTIP_TAXELS,
        model_xml=model_xml,
        distance_slack_m=distance_slack_m,
    )
