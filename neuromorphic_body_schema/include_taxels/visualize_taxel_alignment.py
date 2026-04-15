"""
Create interactive HTML visualization of taxel alignment before/after optimization.

Usage:
    /home/smullercleve/.virtualenvs/mujoco/bin/python \
        neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import GeometryCollection, MultiPoint, Polygon
from shapely.ops import unary_union

from optimize_taxel_alignment import (
    PARTS,
    REPORT_JSON,
    ROOT,
    apply_part_transform,
    load_mesh,
    load_taxel_points,
)

OUT_DIR = ROOT / "neuromorphic_body_schema" / "include_taxels" / "visualizations"
INDEX_HTML = OUT_DIR / "index.html"


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
        opacity=0.35,
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


def _read_results() -> dict[str, dict[str, Any]]:
    if not REPORT_JSON.exists():
        raise FileNotFoundError(
            f"Optimization report not found at {REPORT_JSON}. Run optimize_taxel_alignment.py first."
        )

    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    return {entry["part"]: entry for entry in data}


def _write_part_figure(part_cfg, result: dict[str, Any]) -> Path:
    part = part_cfg.part_name
    base_points = load_taxel_points(
        ROOT / "neuromorphic_body_schema" / "include_taxels" / "positions" / part_cfg.position_file,
        part_cfg.rebase,
    )
    mesh = load_mesh(part_cfg.mesh_files, part_cfg.mesh_pos, part_cfg.mesh_quat_wxyz)

    init_angles = np.array(result["initial"]["delta_angles_deg"], dtype=float)
    init_offsets = np.array(result["initial"]["delta_offsets_m"], dtype=float)
    opt_angles = np.array(result["optimized"]["delta_angles_deg"], dtype=float)
    opt_offsets = np.array(result["optimized"]["delta_offsets_m"], dtype=float)

    before_points = apply_part_transform(
        base_points,
        part,
        delta_angles_deg=init_angles,
        delta_offsets_m=init_offsets,
    )
    after_points = apply_part_transform(
        base_points,
        part,
        delta_angles_deg=opt_angles,
        delta_offsets_m=opt_offsets,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(f"{part}: Manual Baseline", f"{part}: Optimized"),
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
    baseline_points = apply_part_transform(base_points, part)
    mesh = load_mesh(part_cfg.mesh_files, part_cfg.mesh_pos, part_cfg.mesh_quat_wxyz)

    mesh_footprint = _project_mesh_footprint_xz(mesh)
    taxel_footprint = _project_points_hull_xz(baseline_points)
    mesh_x, mesh_z = _polygon_xy_coords(mesh_footprint)
    taxel_x, taxel_z = _polygon_xy_coords(taxel_footprint)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(f"{part}: Manual Baseline 3D", f"{part}: Detected Sole Footprint (XZ)"),
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
        title=f"Footprint Preview | {part} | current manual baseline and detected sole footprint",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_parts = set(args.parts) if args.parts else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, Path]] = []
    result_map: dict[str, dict[str, Any]] = {}
    if not args.preview_footprint:
        result_map = _read_results()
    for part_cfg in PARTS:
        part = part_cfg.part_name
        if selected_parts is not None and part not in selected_parts:
            continue
        if args.preview_footprint:
            output = _write_footprint_preview(part_cfg)
            generated.append((part, output))
            print(f"[ok] wrote {output}")
            continue
        if part not in result_map:
            print(f"[warn] Missing optimized result for part: {part}")
            continue

        output = _write_part_figure(part_cfg, result_map[part])
        generated.append((part, output))
        print(f"[ok] wrote {output}")

    _write_index(generated)
    print(f"[ok] index: {INDEX_HTML}")


if __name__ == "__main__":
    main()
