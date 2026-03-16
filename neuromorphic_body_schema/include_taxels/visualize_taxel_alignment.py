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

    before_points = apply_part_transform(base_points, part, init_angles, init_offsets)
    after_points = apply_part_transform(base_points, part, opt_angles, opt_offsets)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(f"{part}: Before", f"{part}: After"),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_parts = set(args.parts) if args.parts else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_map = _read_results()

    generated: list[tuple[str, Path]] = []
    for part_cfg in PARTS:
        part = part_cfg.part_name
        if selected_parts is not None and part not in selected_parts:
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
