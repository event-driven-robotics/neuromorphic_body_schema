"""
Create interactive HTML visualization of taxel alignment before/after optimization.

Usage:
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots
from shapely.geometry import GeometryCollection, MultiPoint, Polygon
from shapely.ops import unary_union

from optimize_taxel_alignment import (
    PARTS,
    PART_MODEL_BODY,
    REPORT_JSON,
    ROOT,
    _load_mj_model_data,
    apply_manual_steps_with_euler_delta,
    load_mesh,
    load_taxel_points,
    resolve_mesh_transform_from_model,
)

OUT_DIR = ROOT / "neuromorphic_body_schema" / "include_taxels" / "visualizations"
INDEX_HTML = OUT_DIR / "index.html"
CONTACT_MODEL_XML = (
    ROOT
    / "neuromorphic_body_schema"
    / "models"
    / "icub_v2_full_body_improved_contact_sensors.xml"
)
MODEL_XML = ROOT / "neuromorphic_body_schema" / "models" / "icub_v2_full_body_improved.xml"

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
mj_forward = getattr(mujoco, "mj_forward")
mj_name2id = getattr(mujoco, "mj_name2id")
mjtObj = getattr(mujoco, "mjtObj")

_WORLD_MODEL_CACHE: Any = None
_WORLD_DATA_CACHE: Any = None


def _polygon_xy_coords(geom: Any) -> tuple[np.ndarray, np.ndarray]:
    if geom.is_empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    if geom.geom_type == "Polygon":
        coords = np.asarray(geom.exterior.coords, dtype=float)
        return coords[:, 0], coords[:, 1]
    if geom.geom_type == "LineString":
        coords = np.asarray(geom.coords, dtype=float)
        return coords[:, 0], coords[:, 1]
    point = np.asarray(geom.coords[0], dtype=float)
    return np.array([point[0]], dtype=float), np.array([point[1]], dtype=float)


def _project_mesh_footprint_xz(mesh, bottom_slice_height_m: float = 0.002) -> Any:
    """Return a strict top-down XZ footprint for the bottom sole contact patch.

    The footprint is built from faces whose three vertices lie within a thin
    slice above the minimum Y coordinate. This excludes side walls and upper
    geometry that would make a full-mesh projection too wide.
    """

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
        triangle_xz = vertices[face][:, [0, 2]]
        polygon = Polygon(triangle_xz)
        if polygon.is_valid and polygon.area > 0.0:
            projected_triangles.append(polygon)

    if not projected_triangles:
        return MultiPoint(vertices[bottom_faces.reshape(-1), :][:, [0, 2]]).convex_hull

    footprint = unary_union(projected_triangles)
    if isinstance(footprint, GeometryCollection) and len(footprint.geoms) == 0:
        return MultiPoint(vertices[bottom_faces.reshape(-1), :][:, [0, 2]]).convex_hull
    return footprint


def _project_points_hull_xz(points: np.ndarray) -> Any:
    """Return the top-down XZ convex hull of the point cloud."""

    return MultiPoint(points[:, [0, 2]]).convex_hull


def _mesh_trace(mesh, scene_name: str) -> go.Mesh3d:
    vertices = mesh.vertices
    faces = mesh.faces
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="lightgray",
        opacity=0.6,
        flatshading=True,
        name="mesh",
        hoverinfo="skip",
        scene=scene_name,
    )


def _taxel_trace(points: np.ndarray, color: str, name: str, scene_name: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker={"size": 4, "color": color, "opacity": 0.95},
        name=name,
        scene=scene_name,
    )


def _load_world_model_data() -> tuple[Any, Any]:
    global _WORLD_MODEL_CACHE, _WORLD_DATA_CACHE
    if _WORLD_MODEL_CACHE is None or _WORLD_DATA_CACHE is None:
        _WORLD_MODEL_CACHE = MjModel.from_xml_path(str(MODEL_XML))
        _WORLD_DATA_CACHE = MjData(_WORLD_MODEL_CACHE)
        mj_forward(_WORLD_MODEL_CACHE, _WORLD_DATA_CACHE)
    return _WORLD_MODEL_CACHE, _WORLD_DATA_CACHE


def _body_world_transform(part: str) -> tuple[np.ndarray, np.ndarray]:
    model, data = _load_world_model_data()
    body_name = PART_MODEL_BODY[part]
    body_id = int(mj_name2id(model, mjtObj.mjOBJ_BODY, body_name))
    pos = np.array(data.xpos[body_id], dtype=float)
    rot = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)
    return pos, rot


def _points_to_world(points: np.ndarray, part: str) -> np.ndarray:
    pos, rot = _body_world_transform(part)
    return (rot @ np.asarray(points, dtype=float).T).T + pos


def _mesh_to_world(mesh: Any, part: str) -> Any:
    pos, rot = _body_world_transform(part)
    mesh_world = mesh.copy()
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    mesh_world.apply_transform(transform)
    return mesh_world


def _load_compiled_mesh(part_name: str, mesh_files: tuple[str, ...]) -> Any:
    """Load MuJoCo compiled mesh geometry in the same local frame used by the simulator."""

    model, _ = _load_mj_model_data()
    compiled_meshes: list[Any] = []

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
        quat_xyzw = np.array([quat_arr[1], quat_arr[2], quat_arr[3], quat_arr[0]], dtype=float)

        mesh_local = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
        rot = np.array(
            [
                [1.0 - 2.0 * (quat_xyzw[1] ** 2 + quat_xyzw[2] ** 2), 2.0 * (quat_xyzw[0] * quat_xyzw[1] - quat_xyzw[2] * quat_xyzw[3]), 2.0 * (quat_xyzw[0] * quat_xyzw[2] + quat_xyzw[1] * quat_xyzw[3])],
                [2.0 * (quat_xyzw[0] * quat_xyzw[1] + quat_xyzw[2] * quat_xyzw[3]), 1.0 - 2.0 * (quat_xyzw[0] ** 2 + quat_xyzw[2] ** 2), 2.0 * (quat_xyzw[1] * quat_xyzw[2] - quat_xyzw[0] * quat_xyzw[3])],
                [2.0 * (quat_xyzw[0] * quat_xyzw[2] - quat_xyzw[1] * quat_xyzw[3]), 2.0 * (quat_xyzw[1] * quat_xyzw[2] + quat_xyzw[0] * quat_xyzw[3]), 1.0 - 2.0 * (quat_xyzw[0] ** 2 + quat_xyzw[1] ** 2)],
            ],
            dtype=float,
        )
        mesh_local.vertices = (rot @ mesh_local.vertices.T).T + pos_arr
        compiled_meshes.append(mesh_local)

    if not compiled_meshes:
        raise RuntimeError(f"No compiled meshes resolved for part={part_name}: {mesh_files}")
    if len(compiled_meshes) == 1:
        return compiled_meshes[0]
    return trimesh.util.concatenate(compiled_meshes)


def _read_results() -> dict[str, dict[str, Any]]:
    if not REPORT_JSON.exists():
        raise FileNotFoundError(
            f"Optimization report not found at {REPORT_JSON}. Run optimize_taxel_alignment.py first."
        )

    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    return {entry["part"]: entry for entry in data}


def _manual_steps_from_report(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = result.get("manual_steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("Report field 'manual_steps' must be a list")
    steps: list[dict[str, Any]] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        offsets = step.get("offsets_m")
        angles = step.get("angles_deg")
        if not (isinstance(offsets, list) and isinstance(angles, list) and len(offsets) == 3 and len(angles) == 3):
            continue
        steps.append({
            "offsets_m": [float(offsets[0]), float(offsets[1]), float(offsets[2])],
            "angles_deg": [float(angles[0]), float(angles[1]), float(angles[2])],
        })
    return steps


def _read_contact_model_sites(part: str, world_frame: bool = False) -> np.ndarray | None:
    if not CONTACT_MODEL_XML.exists():
        return None
    try:
        tree = ET.parse(CONTACT_MODEL_XML)
    except Exception:
        return None

    prefix = f"{part}_taxel_"
    indexed_positions: list[tuple[int, np.ndarray]] = []
    for site in tree.iterfind(".//site"):
        name = site.attrib.get("name", "")
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if not suffix.isdigit():
            continue
        pos_text = site.attrib.get("pos")
        if pos_text is None:
            continue
        values = [float(v) for v in re.split(r"\s+", pos_text.strip()) if v]
        if len(values) != 3:
            continue
        indexed_positions.append((int(suffix), np.array(values, dtype=float)))

    if not indexed_positions:
        return None
    indexed_positions.sort(key=lambda item: item[0])
    points = np.vstack([pos for _, pos in indexed_positions])
    if world_frame:
        return _points_to_world(points, part)
    return points


def _write_part_figure(
    part: str,
    result: dict[str, Any],
    world_frame: bool = False,
    use_compiled_mesh: bool = False,
) -> Path:
    position_file = result.get("position_file")
    if not isinstance(position_file, str):
        raise ValueError(f"Missing or invalid position_file in report for part: {part}")
    rebase = bool(result.get("rebase", False))
    mesh_files_raw = result.get("mesh_files", [])
    if not isinstance(mesh_files_raw, list) or not mesh_files_raw:
        raise ValueError(f"Missing or invalid mesh_files in report for part: {part}")
    mesh_files = tuple(str(m) for m in mesh_files_raw)

    manual_steps = _manual_steps_from_report(result)
    if not manual_steps:
        raise ValueError(f"Missing manual_steps in report for part: {part}")

    base_points = load_taxel_points(
        ROOT / "neuromorphic_body_schema" / "include_taxels" / "positions" / position_file,
        rebase,
    )
    mesh = _load_compiled_mesh(part, mesh_files) if use_compiled_mesh else load_mesh(part, mesh_files)

    init_angles = np.array(result["initial"]["delta_angles_deg"], dtype=float)
    init_offsets = np.array(result["initial"]["delta_offsets_m"], dtype=float)
    opt_angles = np.array(result["optimized"]["delta_angles_deg"], dtype=float)
    opt_offsets = np.array(result["optimized"]["delta_offsets_m"], dtype=float)

    before_points = apply_manual_steps_with_euler_delta(
        base_points,
        manual_steps,
        delta_angles_deg=init_angles,
        delta_offsets_m=init_offsets,
    )
    after_points = apply_manual_steps_with_euler_delta(
        base_points,
        manual_steps,
        delta_angles_deg=opt_angles,
        delta_offsets_m=opt_offsets,
    )

    if world_frame:
        mesh = _mesh_to_world(mesh, part)
        before_points = _points_to_world(before_points, part)
        after_points = _points_to_world(after_points, part)

    model_sites = _read_contact_model_sites(part, world_frame=world_frame)
    consistency_note = ""
    if model_sites is None:
        consistency_note = " | model check: n/a"
    elif len(model_sites) != len(after_points):
        consistency_note = (
            f" | model check: count mismatch xml={len(model_sites)} vs report={len(after_points)}"
        )
    else:
        distances = np.linalg.norm(model_sites - after_points, axis=1)
        consistency_note = (
            f" | model check max={float(np.max(distances)):.6e} m rms={float(np.sqrt(np.mean(distances ** 2))):.6e} m"
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(f"{part}: Zero Baseline", f"{part}: Optimized"),
        horizontal_spacing=0.03,
    )

    fig.add_trace(_mesh_trace(mesh, "scene"), row=1, col=1)
    fig.add_trace(_taxel_trace(before_points, "red", "taxels before", "scene"), row=1, col=1)

    fig.add_trace(_mesh_trace(mesh, "scene2"), row=1, col=2)
    fig.add_trace(_taxel_trace(after_points, "limegreen", "taxels after", "scene2"), row=1, col=2)

    before_md = float(result["initial"]["mean_distance_m"])
    after_md = float(result["optimized"]["mean_distance_m"])
    improve_pct = float(result["delta"]["mean_distance_improvement_pct"])

    fig.update_layout(
        title=(
            f"Taxel Alignment | {part} | mean distance: "
            f"{before_md:.6f} -> {after_md:.6f} m ({improve_pct:.2f}% better)"
            f" | frame={'world' if world_frame else 'local'}"
            f" | mesh={'compiled' if use_compiled_mesh else 'raw-stl'}"
            f"{consistency_note}"
        ),
        template="plotly_white",
        height=760,
        width=1500,
        legend={"orientation": "h", "y": -0.03},
        margin={"l": 10, "r": 10, "t": 70, "b": 20},
    )

    # Equal aspect for consistent visual comparison.
    fig.update_scenes(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",
        camera={"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
    )

    output_path = OUT_DIR / f"{part}_before_after.html"
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path


def _write_footprint_preview(part_cfg) -> Path:
    part = part_cfg.part_name
    base_points = load_taxel_points(
        ROOT / "neuromorphic_body_schema" / "include_taxels" / "positions" / part_cfg.position_file,
        part_cfg.rebase,
    )
    baseline_points = apply_manual_steps_with_euler_delta(base_points, list(part_cfg.manual_steps))
    mesh = load_mesh(part_cfg.part_name, part_cfg.mesh_files)

    mesh_footprint = _project_mesh_footprint_xz(mesh)
    taxel_footprint = _project_points_hull_xz(baseline_points)
    mesh_x, mesh_z = _polygon_xy_coords(mesh_footprint)
    taxel_x, taxel_z = _polygon_xy_coords(taxel_footprint)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(f"{part}: Zero Baseline 3D", f"{part}: Detected Sole Footprint (XZ)"),
        horizontal_spacing=0.06,
    )

    fig.add_trace(_mesh_trace(mesh, "scene"), row=1, col=1)
    fig.add_trace(_taxel_trace(baseline_points, "red", "taxels baseline", "scene"), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 2],
            mode="markers",
            marker={"size": 3, "color": "rgba(120,120,120,0.25)"},
            name="mesh projection",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=mesh_x,
            y=mesh_z,
            mode="lines",
            line={"color": "royalblue", "width": 3},
            name="sole footprint hull",
            fill="toself",
            fillcolor="rgba(65,105,225,0.12)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_points[:, 0],
            y=baseline_points[:, 2],
            mode="markers",
            marker={"size": 7, "color": "firebrick", "opacity": 0.85},
            name="taxels baseline",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=taxel_x,
            y=taxel_z,
            mode="lines",
            line={"color": "firebrick", "width": 3},
            name="taxel hull",
            fill="toself",
            fillcolor="rgba(178,34,34,0.10)",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Footprint Preview | {part} | current zero baseline and detected sole footprint",
        template="plotly_white",
        height=760,
        width=1500,
        legend={"orientation": "h", "y": -0.03},
        margin={"l": 10, "r": 10, "t": 70, "b": 20},
    )
    fig.update_scenes(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",
        camera={"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
    )
    fig.update_xaxes(title_text="X (m)", scaleanchor="y", scaleratio=1.0, row=1, col=2)
    fig.update_yaxes(title_text="Z (m)", row=1, col=2)

    output_path = OUT_DIR / f"{part}_footprint_preview.html"
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path


def _write_index(part_files: list[tuple[str, Path]]) -> None:
    lines = [
        "<html>",
        "<head><meta charset='utf-8'><title>Taxel Alignment Visualizations</title></head>",
        "<body style='font-family:Arial,sans-serif;padding:24px'>",
        "<h2>Taxel Alignment Visualizations (Before vs After)</h2>",
        "<ul>",
    ]

    for part, file_path in part_files:
        rel = file_path.name
        lines.append(f"<li><a href='{rel}' target='_blank'>{part}</a></li>")

    lines.extend(["</ul>", "</body>", "</html>"])
    INDEX_HTML.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create interactive HTML visualization of taxel alignment.")
    parser.add_argument(
        "--parts",
        nargs="+",
        help="Optional part names to visualize, for example: --parts r_forearm",
    )
    parser.add_argument(
        "--preview-footprint",
        action="store_true",
        help="Show the current manual baseline and detected XZ footprint before running optimization.",
    )
    parser.add_argument(
        "--world-frame",
        action="store_true",
        help="Render mesh and taxels in MuJoCo world coordinates instead of anchor-body local coordinates.",
    )
    parser.add_argument(
        "--use-compiled-mesh",
        action="store_true",
        help="Render the compiled MuJoCo mesh geometry instead of the raw STL-derived mesh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_parts = set(args.parts) if args.parts else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, Path]] = []
    result_map: dict[str, dict[str, Any]] = {}
    if not args.preview_footprint:
        result_map = _read_results()
    if args.preview_footprint:
        for part_cfg in PARTS:
            part = part_cfg.part_name
            if selected_parts is not None and part not in selected_parts:
                continue
            output = _write_footprint_preview(part_cfg)
            generated.append((part, output))
            print(f"[ok] wrote {output}")
    else:
        ordered_parts = [cfg.part_name for cfg in PARTS]
        for part in sorted(result_map.keys()):
            if part not in ordered_parts:
                ordered_parts.append(part)

        for part in ordered_parts:
            if selected_parts is not None and part not in selected_parts:
                continue
            if part not in result_map:
                print(f"[warn] Missing optimized result for part: {part}")
                continue
            output = _write_part_figure(
                part,
                result_map[part],
                world_frame=bool(args.world_frame),
                use_compiled_mesh=bool(args.use_compiled_mesh),
            )
            generated.append((part, output))
            print(f"[ok] wrote {output}")

    _write_index(generated)
    print(f"[ok] index: {INDEX_HTML}")


if __name__ == "__main__":
    main()
