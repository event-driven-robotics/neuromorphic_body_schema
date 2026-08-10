"""Project a manually tuned fingertip taxel baseline onto detailed MuJoCo meshes."""

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
        [13.0e-3, 3.25e-3, -3.25e-3],
        [13.0e-3, 6.5e-3, 0.0],
        [13.0e-3, 3.25e-3, 3.25e-3],
        [14.75e-3, -1.5e-3, 6.5e-3],
        [14.75e-3, -1.5e-3, -6.5e-3],
        [16.5e-3, 3.25e-3, 3.25e-3],
        [16.5e-3, 6.5e-3, 0.0],
        [16.5e-3, 3.25e-3, -3.25e-3],
        [19.625e-3, 1.625e-3, -3.25e-3],
        [21.25e-3, 3.25e-3, 0.0],
        [19.625e-3, 1.625e-3, 3.25e-3],
        [24.5e-3, 0.0, 0.0],
    ],
    dtype=float,
)
# This is the manually maintained, unprojected baseline. Every run starts from
# these coordinates; projected output is never fed back in as a new seed.

# The legacy 12-taxel layout is bilaterally symmetric about the fingertip-local
# z=0 plane. Indices are zero-based and preserve the sensor/channel ordering.
FINGERTIP_MIRRORED_PAIRS = ((0, 2), (3, 4), (5, 7), (8, 10))
FINGERTIP_CENTERLINE_INDICES = (1, 6, 9, 11)
FINGERTIP_SYMMETRY_AXIS = 2
FINGERTIP_X_ALIGNED_GROUPS = ((0, 1, 2), (3, 4), (5, 6, 7), (8, 10))
FINGERTIP_FREE_X_INDICES = (9, 11)
FINGERTIP_ROUNDED_TIP_CENTER = np.array([18e-3, 0.0, 0.0], dtype=float)

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

FINGERTIP_BODY_TO_PROJECTION_MESH = {
    "r_hand_thumb_3": "full_RightThumb3",
    "r_hand_index_3": "full_RightIndex3",
    "r_hand_middle_3": "full_RightMiddle3",
    "r_hand_ring_3": "full_RightRing3",
    "r_hand_little_3": "full_RightLittle3",
    "l_hand_thumb_3": "full_LeftThumb3",
    "l_hand_index_3": "full_LeftIndex3",
    "l_hand_middle_3": "full_LeftMiddle3",
    "l_hand_ring_3": "full_LeftRing3",
    "l_hand_little_3": "full_LeftLittle3",
}


@dataclass(frozen=True)
class ProjectionResult:
    body_name: str
    part_name: str
    mesh: trimesh.Trimesh
    original_points: np.ndarray
    surface_projected_points: np.ndarray
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


def load_fingertip_projection_mesh(
    body_name: str,
    model_xml: Path = MODEL_XML,
) -> trimesh.Trimesh:
    """Load a detailed fingertip mesh in the local frame of its body."""

    model_xml = model_xml.resolve()
    cache_key = (model_xml, body_name)
    if cache_key in _MESH_CACHE:
        return _MESH_CACHE[cache_key].copy()

    mesh_name = FINGERTIP_BODY_TO_PROJECTION_MESH.get(body_name)
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
        raise RuntimeError(f"No projection geom for body={body_name}, mesh={mesh_name}")

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


def _project_one_point_radially(
    mesh: trimesh.Trimesh,
    point: np.ndarray,
    allow_x_movement: bool = False,
) -> np.ndarray:
    """Project to the outermost surface along an outward ray.

    Grouped taxels use a ray in their constant-x yz cross-section. Unique-x
    centerline taxels instead use a ray from the rounded-tip center, allowing
    movement in xy while retaining z=0. If the detailed model exposes both an
    inner and an outer shell, the furthest positive intersection is selected.
    """

    if allow_x_movement:
        ray_origin = FINGERTIP_ROUNDED_TIP_CENTER
        radial_direction = np.array(point - ray_origin, dtype=float)
    else:
        ray_origin = np.array([point[0], 0.0, 0.0], dtype=float)
        radial_direction = np.array([0.0, point[1], point[2]], dtype=float)

    direction_norm = float(np.linalg.norm(radial_direction))
    if direction_norm <= 1e-12:
        raise ValueError(f"Fingertip taxel has no outward direction: {point.tolist()}")
    radial_direction /= direction_norm

    triangles = np.asarray(mesh.triangles, dtype=float)
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    repeated_direction = np.broadcast_to(radial_direction, edge_2.shape)
    direction_cross_edge_2 = np.cross(repeated_direction, edge_2)
    determinant = np.einsum("ij,ij->i", edge_1, direction_cross_edge_2)

    non_parallel = np.abs(determinant) > 1e-12
    inverse_determinant = np.zeros_like(determinant)
    inverse_determinant[non_parallel] = 1.0 / determinant[non_parallel]

    origin_from_vertex = ray_origin - triangles[:, 0]
    barycentric_u = inverse_determinant * np.einsum(
        "ij,ij->i", origin_from_vertex, direction_cross_edge_2
    )
    origin_cross_edge_1 = np.cross(origin_from_vertex, edge_1)
    barycentric_v = inverse_determinant * np.einsum(
        "ij,j->i", origin_cross_edge_1, radial_direction
    )
    ray_distance = inverse_determinant * np.einsum(
        "ij,ij->i", edge_2, origin_cross_edge_1
    )

    tolerance = 1e-9
    valid = (
        non_parallel
        & (barycentric_u >= -tolerance)
        & (barycentric_v >= -tolerance)
        & (barycentric_u + barycentric_v <= 1.0 + tolerance)
        & (ray_distance >= 0.0)
    )
    intersections = ray_distance[valid]
    if len(intersections) == 0:
        raise RuntimeError(
            "No outward radial mesh intersection for fingertip taxel at "
            f"{point.tolist()}"
        )

    selected_distance = float(np.max(intersections))
    return ray_origin + selected_distance * radial_direction


def project_points_radially_to_fingertip_surface(
    body_name: str,
    points: np.ndarray,
    model_xml: Path = MODEL_XML,
    free_x_indices: Sequence[int] = FINGERTIP_FREE_X_INDICES,
) -> ProjectionResult:
    """Project points onto the outer shell while preserving x-row structure."""

    mesh = load_fingertip_projection_mesh(body_name, model_xml=model_xml)
    original = np.array(points, dtype=float)
    free_x = set(free_x_indices)
    projected = np.vstack(
        [
            _project_one_point_radially(
                mesh=mesh,
                point=point,
                allow_x_movement=index in free_x,
            )
            for index, point in enumerate(original)
        ]
    )
    return ProjectionResult(
        body_name=body_name,
        part_name=FINGERTIP_BODY_TO_PART[body_name],
        mesh=mesh,
        original_points=original,
        surface_projected_points=projected,
        projected_points=projected,
        movement_m=np.linalg.norm(projected - original, axis=1),
    )


def enforce_bilateral_symmetry(
    points: np.ndarray,
    mirrored_pairs: Sequence[tuple[int, int]] = FINGERTIP_MIRRORED_PAIRS,
    centerline_indices: Sequence[int] = FINGERTIP_CENTERLINE_INDICES,
    symmetry_axis: int = FINGERTIP_SYMMETRY_AXIS,
) -> np.ndarray:
    """Return the least-change exactly symmetric version of a point layout.

    One point in each pair is reflected into the coordinate frame of the other
    before their positions are averaged. Centerline points retain their other
    coordinates and are placed exactly on the symmetry plane.
    """

    symmetric = np.array(points, dtype=float, copy=True)
    if symmetric.ndim != 2:
        raise ValueError("points must be a two-dimensional array")
    if symmetry_axis < 0 or symmetry_axis >= symmetric.shape[1]:
        raise ValueError(f"Invalid symmetry axis: {symmetry_axis}")

    reflection = np.ones(symmetric.shape[1], dtype=float)
    reflection[symmetry_axis] = -1.0
    point_count = len(symmetric)

    for first, second in mirrored_pairs:
        if not (0 <= first < point_count and 0 <= second < point_count):
            raise ValueError(f"Mirrored pair {(first, second)} is out of range")
        representative = 0.5 * (
            symmetric[first] + reflection * symmetric[second]
        )
        symmetric[first] = representative
        symmetric[second] = reflection * representative

    for index in centerline_indices:
        if not 0 <= index < point_count:
            raise ValueError(f"Centerline index {index} is out of range")
        symmetric[index, symmetry_axis] = 0.0

    return symmetric


def project_points_to_fingertip_surface(
    body_name: str,
    points: np.ndarray,
    model_xml: Path = MODEL_XML,
    distance_slack_m: float = 0.0015,
) -> ProjectionResult:
    mesh = load_fingertip_projection_mesh(body_name, model_xml=model_xml)
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
        surface_projected_points=projected,
        projected_points=projected,
        movement_m=movement,
    )


def project_legacy_fingertip_taxels(
    body_name: str,
    model_xml: Path = MODEL_XML,
    distance_slack_m: float = 0.0015,
) -> ProjectionResult:
    surface_projection = project_points_radially_to_fingertip_surface(
        body_name=body_name,
        points=LEGACY_FINGERTIP_TAXELS,
        model_xml=model_xml,
    )
    symmetric = enforce_bilateral_symmetry(
        surface_projection.surface_projected_points
    )
    return ProjectionResult(
        body_name=surface_projection.body_name,
        part_name=surface_projection.part_name,
        mesh=surface_projection.mesh,
        original_points=surface_projection.original_points,
        surface_projected_points=surface_projection.surface_projected_points,
        projected_points=symmetric,
        movement_m=np.linalg.norm(
            symmetric - surface_projection.original_points, axis=1
        ),
    )


def project_all_legacy_fingertip_taxels(
    model_xml: Path = MODEL_XML,
    distance_slack_m: float = 0.0015,
) -> dict[str, np.ndarray]:
    """Return symmetric fingertip-local taxel positions keyed by skin part."""

    return {
        part_name: project_legacy_fingertip_taxels(
            body_name=body_name,
            model_xml=model_xml,
            distance_slack_m=distance_slack_m,
        ).projected_points
        for body_name, part_name in FINGERTIP_BODY_TO_PART.items()
    }
