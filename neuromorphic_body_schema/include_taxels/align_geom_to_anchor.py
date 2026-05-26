#!/usr/bin/env python3
"""
align_geom_to_anchor.py

Small utility to inspect and nudge one geom relative to an anchor geom
without manually editing MuJoCo XML.

Usage examples:

  # Inspect current left-hand alignment (no edits)
    python align_geom_to_anchor.py --side left --show

  # Move left cover forward 2 mm in palm-local frame
    python align_geom_to_anchor.py --side left --dx 0.002 --frame anchor

  # Rotate left cover by -10 deg around local Z axis of the cover
    python align_geom_to_anchor.py --side left --rz -10 --frame movable

  # Apply same delta to both hands and write to a different file
    python align_geom_to_anchor.py --side both --dy 0.001 --output /tmp/model.xml

    # Generic mode: tune any movable geom relative to any anchor geom
    python align_geom_to_anchor.py \
            --anchor-geom torso_shell_vis_0 \
            --movable-geom torso_cover_vis_0 \
            --show --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple
import xml.etree.ElementTree as ET
import tempfile
import webbrowser

import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "icub_v2_full_body_improved.xml"
)

HAND_CONFIG = {
    "left": {
        "anchor": "l_hand_vis_0",
        "movable": "l_hand_top_cover_vis_0",
    },
    "right": {
        "anchor": "r_hand_vis_0",
        "movable": "r_hand_top_cover_vis_0",
    },
}


def _resolve_geom_pairs(
    side: str,
    anchor_geom: str | None,
    movable_geom: str | None,
    pair_label: str | None,
) -> list[dict[str, str]]:
    """Resolve alignment targets.

    Default behavior uses hand anchor/movable pairs from --side.
    Generic behavior is enabled by passing both --anchor-geom and --movable-geom.
    """
    if bool(anchor_geom) != bool(movable_geom):
        raise ValueError("Provide both --anchor-geom and --movable-geom together")

    if anchor_geom and movable_geom:
        label = pair_label or f"custom:{anchor_geom}->{movable_geom}"
        return [{"label": label, "anchor": anchor_geom, "movable": movable_geom}]

    sides = ["left", "right"] if side == "both" else [side]
    pairs: list[dict[str, str]] = []
    for side_name in sides:
        cfg = HAND_CONFIG[side_name]
        pairs.append(
            {
                "label": side_name,
                "anchor": cfg["anchor"],
                "movable": cfg["movable"],
            }
        )
    return pairs

ATTR_RE = re.compile(r"(\w+)=\"([^\"]*)\"")


def _as_float_vec_attr(attrs: Dict[str, str], key: str, n: int) -> np.ndarray:
    if key not in attrs:
        if key == "quat" and n == 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if key == "pos" and n == 3:
            return np.zeros(3, dtype=float)
        raise ValueError(f"Missing required attribute '{key}'")
    return _parse_vec(attrs[key], n)


def _parse_vec(text: str, n: int) -> np.ndarray:
    vals = np.fromstring(text, sep=" ", dtype=float)
    if vals.size != n:
        raise ValueError(f"Expected {n} values, got: {text}")
    return vals


def _fmt_float(x: float) -> str:
    return f"{x:.17g}"


def _fmt_vec(v: np.ndarray) -> str:
    return " ".join(_fmt_float(float(x)) for x in v)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return q / n


def _quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=float)
    return _normalize_quat(q)


def _rot_from_euler_deg_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    ax = np.deg2rad(rx)
    ay = np.deg2rad(ry)
    az = np.deg2rad(rz)

    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)

    # Intrinsic XYZ
    return Rx @ Ry @ Rz


def _euler_deg_xyz_from_rot(R: np.ndarray) -> np.ndarray:
    """Inverse of _rot_from_euler_deg_xyz for intrinsic XYZ convention."""
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    y = np.arcsin(sy)
    cy = np.cos(y)

    # Handle near-gimbal-lock robustly.
    if abs(cy) > 1e-9:
        x = np.arctan2(-R[1, 2], R[2, 2])
        z = np.arctan2(-R[0, 1], R[0, 0])
    else:
        x = np.arctan2(R[2, 1], R[1, 1])
        z = 0.0

    return np.rad2deg(np.array([x, y, z], dtype=float))


def _relative_movable_in_anchor_frame(
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    movable_pos: np.ndarray,
    movable_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (relative position in anchor frame, relative Euler XYZ deg)."""
    R_anchor = _quat_to_mat(anchor_quat)
    R_movable = _quat_to_mat(movable_quat)

    rel_pos = R_anchor.T @ (movable_pos - anchor_pos)
    R_rel = R_anchor.T @ R_movable
    rel_euler = _euler_deg_xyz_from_rot(R_rel)
    return rel_pos, rel_euler


def _movable_from_relative_in_anchor_frame(
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    rel_pos: np.ndarray,
    rel_euler_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose world movable pose from anchor-fixed relative pose controls."""
    R_anchor = _quat_to_mat(anchor_quat)
    R_rel = _rot_from_euler_deg_xyz(*rel_euler_deg.tolist())

    movable_pos = anchor_pos + R_anchor @ rel_pos
    movable_quat = _mat_to_quat(R_anchor @ R_rel)
    return movable_pos, movable_quat


def _get_geom_line_index_and_attrs(lines: list[str], geom_name: str) -> Tuple[int, Dict[str, str]]:
    for idx, line in enumerate(lines):
        if "<geom" not in line or f'name="{geom_name}"' not in line:
            continue
        attrs = {k: v for k, v in ATTR_RE.findall(line)}
        if attrs.get("name") == geom_name:
            return idx, attrs
    raise ValueError(f"Geom not found: {geom_name}")


def _replace_attr(line: str, key: str, value: str) -> str:
    pattern = rf"\b{key}=\"[^\"]*\""
    if re.search(pattern, line):
        return re.sub(pattern, f'{key}="{value}"', line, count=1)
    return line.replace("/>", f' {key}="{value}"/>', 1)


def _apply_delta(
    movable_pos: np.ndarray,
    movable_quat: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    delta_pos_local: np.ndarray,
    delta_rot_xyz_deg: np.ndarray,
    frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    R_movable = _quat_to_mat(movable_quat)
    R_anchor = _quat_to_mat(anchor_quat)
    R_delta = _rot_from_euler_deg_xyz(*delta_rot_xyz_deg.tolist())

    if frame == "world":
        pos_new = movable_pos + delta_pos_local
        R_new = R_delta @ R_movable
    elif frame in {"movable", "cover"}:
        pos_new = movable_pos + R_movable @ delta_pos_local
        R_new = R_movable @ R_delta
    elif frame in {"anchor", "palm"}:
        pos_new = movable_pos + R_anchor @ delta_pos_local
        R_new = (R_anchor @ R_delta @ R_anchor.T) @ R_movable
    else:
        raise ValueError(f"Unknown frame: {frame}")

    return pos_new, _mat_to_quat(R_new)


def _print_alignment(
    label: str,
    anchor_name: str,
    movable_name: str,
    anchor_pos: np.ndarray,
    movable_pos: np.ndarray,
    anchor_q: np.ndarray,
    movable_q: np.ndarray,
) -> None:
    rel_pos_world = movable_pos - anchor_pos
    R_rel = _quat_to_mat(anchor_q).T @ _quat_to_mat(movable_q)
    rel_q = _mat_to_quat(R_rel)

    print(f"[{label}] anchor ({anchor_name}) pos : {_fmt_vec(anchor_pos)}")
    print(f"[{label}] movable ({movable_name}) pos: {_fmt_vec(movable_pos)}")
    print(f"[{label}] delta pos (world): {_fmt_vec(rel_pos_world)}")
    print(f"[{label}] anchor ({anchor_name}) quat : {_fmt_vec(anchor_q)}")
    print(f"[{label}] movable ({movable_name}) quat: {_fmt_vec(movable_q)}")
    print(f"[{label}] rel quat (anchor->movable): {_fmt_vec(rel_q)}")


def _load_mesh_assets(model_path: Path) -> Tuple[Path, Dict[str, Tuple[Path, np.ndarray]]]:
    tree = ET.parse(model_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    base_dir = model_path.parent
    mesh_base = (base_dir / meshdir).resolve() if meshdir else base_dir.resolve()

    assets: Dict[str, Tuple[Path, np.ndarray]] = {}
    asset_node = root.find("asset")
    if asset_node is None:
        return mesh_base, assets

    for mesh in asset_node.findall("mesh"):
        name = mesh.get("name")
        file_attr = mesh.get("file")
        if not name or not file_attr:
            continue
        scale_attr = mesh.get("scale")
        if scale_attr:
            scale = _parse_vec(scale_attr, 3)
        else:
            scale = np.ones(3, dtype=float)
        assets[name] = ((mesh_base / file_attr).resolve(), scale)

    return mesh_base, assets


def _build_visualization(
    lines: list[str],
    model_path: Path,
    pairs: list[dict[str, str]],
    title_label: str,
    html_path: Path | None = None,
    open_browser: bool = False,
) -> Path:
    try:
        import trimesh
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Visualization requires 'trimesh' and 'plotly'. Install with: pip install trimesh plotly"
        ) from exc

    _, assets = _load_mesh_assets(model_path)

    fig = go.Figure()

    colors = {
        "anchor": "rgb(70,130,180)",
        "movable": "rgb(220,120,30)",
    }

    for pair in pairs:
        label = pair["label"]
        anchor_name = pair["anchor"]
        movable_name = pair["movable"]

        _, anchor_attrs = _get_geom_line_index_and_attrs(lines, anchor_name)
        _, movable_attrs = _get_geom_line_index_and_attrs(lines, movable_name)

        anchor_mesh_name = anchor_attrs.get("mesh")
        movable_mesh_name = movable_attrs.get("mesh")
        if not anchor_mesh_name or not movable_mesh_name:
            raise RuntimeError(f"Missing mesh attributes for pair '{label}'")

        for part, geom_attrs, mesh_name in [
            ("anchor", anchor_attrs, anchor_mesh_name),
            ("movable", movable_attrs, movable_mesh_name),
        ]:
            if mesh_name not in assets:
                raise RuntimeError(f"Mesh asset not found in XML asset section: {mesh_name}")

            mesh_path, scale = assets[mesh_name]
            if not mesh_path.exists():
                raise RuntimeError(f"Mesh file does not exist: {mesh_path}")

            mesh = trimesh.load(mesh_path, force="mesh")
            vertices = np.asarray(getattr(mesh, "vertices"), dtype=float) * scale[None, :]
            faces = np.asarray(getattr(mesh, "faces"), dtype=int)

            pos = _as_float_vec_attr(geom_attrs, "pos", 3)
            quat = _as_float_vec_attr(geom_attrs, "quat", 4)
            R = _quat_to_mat(quat)
            vertices_world = (R @ vertices.T).T + pos[None, :]

            fig.add_trace(
                go.Mesh3d(
                    x=vertices_world[:, 0],
                    y=vertices_world[:, 1],
                    z=vertices_world[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.38 if part == "movable" else 0.58,
                    color=colors[part],
                    name=f"{label}_{part}",
                )
            )

    fig.update_layout(
        title=f"Geom Alignment Preview ({title_label})",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if html_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"geom_alignment_{title_label.replace(' ', '_')}_", suffix=".html", delete=False
        )
        html_path = Path(tmp.name)
        tmp.close()
    else:
        html_path = html_path.resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(html_path), include_plotlyjs="cdn")

    if open_browser:
        webbrowser.open_new_tab(html_path.as_uri())

    return html_path


def _build_mesh_plot_figure(
    pair_label: str,
    assets: Dict[str, Tuple[Path, np.ndarray]],
    anchor_attrs: Dict[str, str],
    movable_attrs: Dict[str, str],
    rel_pos: np.ndarray,
    rel_euler_deg: np.ndarray,
):
    import trimesh
    import plotly.graph_objects as go

    anchor_mesh_name = anchor_attrs.get("mesh")
    movable_mesh_name = movable_attrs.get("mesh")
    if not anchor_mesh_name or not movable_mesh_name:
        raise RuntimeError("Anchor or movable mesh attribute missing")
    if anchor_mesh_name not in assets or movable_mesh_name not in assets:
        raise RuntimeError("Anchor or movable mesh asset not found in <asset>")

    anchor_pos = _as_float_vec_attr(anchor_attrs, "pos", 3)
    anchor_quat = _as_float_vec_attr(anchor_attrs, "quat", 4)
    movable_pos, movable_quat = _movable_from_relative_in_anchor_frame(
        anchor_pos, anchor_quat, rel_pos, rel_euler_deg
    )

    def _mesh_trace(mesh_name: str, pos: np.ndarray, quat: np.ndarray, color: str, opacity: float, name: str):
        mesh_path, scale = assets[mesh_name]
        if not mesh_path.exists():
            raise RuntimeError(f"Mesh file does not exist: {mesh_path}")
        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.asarray(getattr(mesh, "vertices"), dtype=float) * scale[None, :]
        faces = np.asarray(getattr(mesh, "faces"), dtype=int)

        R = _quat_to_mat(quat)
        vertices_world = (R @ vertices.T).T + pos[None, :]
        return go.Mesh3d(
            x=vertices_world[:, 0],
            y=vertices_world[:, 1],
            z=vertices_world[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=opacity,
            color=color,
            name=name,
        )

    fig = go.Figure()
    fig.add_trace(
        _mesh_trace(
            anchor_mesh_name,
            anchor_pos,
            anchor_quat,
            color="rgb(70,130,180)",
            opacity=0.58,
            name=f"{pair_label}_anchor",
        )
    )
    fig.add_trace(
        _mesh_trace(
            movable_mesh_name,
            movable_pos,
            movable_quat,
            color="rgb(220,120,30)",
            opacity=0.38,
            name=f"{pair_label}_movable",
        )
    )
    fig.update_layout(
        title=f"Geom Alignment ({pair_label})",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _run_interactive_ui(
    model_path: Path,
    pair_label: str,
    anchor_name: str,
    movable_name: str,
    save_path: Path | None,
    port: int,
    start_zero: bool,
) -> None:
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
    except ImportError as exc:
        raise RuntimeError(
            "Interactive mode requires 'dash'. Install with: pip install dash"
        ) from exc

    lines = model_path.read_text(encoding="utf-8").splitlines(keepends=True)
    _, assets = _load_mesh_assets(model_path)
    _, anchor_attrs = _get_geom_line_index_and_attrs(lines, anchor_name)
    _, movable_attrs = _get_geom_line_index_and_attrs(lines, movable_name)

    anchor_pos = _as_float_vec_attr(anchor_attrs, "pos", 3)
    anchor_quat = _as_float_vec_attr(anchor_attrs, "quat", 4)
    movable_pos = _as_float_vec_attr(movable_attrs, "pos", 3)
    movable_quat = _as_float_vec_attr(movable_attrs, "quat", 4)
    rel_pos0, rel_euler0 = _relative_movable_in_anchor_frame(
        anchor_pos, anchor_quat, movable_pos, movable_quat
    )
    if start_zero:
        rel_pos0 = np.zeros(3, dtype=float)
        rel_euler0 = np.zeros(3, dtype=float)

    app = dash.Dash(__name__)

    slider_specs = [
        ("dx", "dX (m)", -0.08, 0.08, float(rel_pos0[0]), 0.0005),
        ("dy", "dY (m)", -0.08, 0.08, float(rel_pos0[1]), 0.0005),
        ("dz", "dZ (m)", -0.08, 0.08, float(rel_pos0[2]), 0.0005),
        ("rx", "rX (deg)", -180.0, 180.0, float(rel_euler0[0]), 0.25),
        ("ry", "rY (deg)", -180.0, 180.0, float(rel_euler0[1]), 0.25),
        ("rz", "rZ (deg)", -180.0, 180.0, float(rel_euler0[2]), 0.25),
    ]

    controls = []
    for key, label, vmin, vmax, v0, step in slider_specs:
        controls.append(html.Div(label, style={"marginTop": "8px", "fontWeight": "bold"}))
        controls.append(
            dcc.Slider(
                id=f"slider-{key}",
                min=vmin,
                max=vmax,
                step=step,
                value=v0,
                tooltip={"always_visible": True, "placement": "bottom"},
            )
        )

    app.layout = html.Div(
        [
            html.H3(
                f"Geom Alignment UI ({pair_label}) - Anchor fixed reference frame"
            ),
            dcc.Graph(id="alignment-graph", style={"height": "70vh"}),
            html.Div(controls, style={"width": "95%", "margin": "auto"}),
            html.Button("Save to XML", id="save-btn", n_clicks=0, style={"marginTop": "14px"}),
            html.Div(id="save-status", style={"marginTop": "8px", "fontWeight": "bold"}),
        ],
        style={"padding": "12px"},
    )

    @app.callback(
        Output("alignment-graph", "figure"),
        [Input("slider-dx", "value"), Input("slider-dy", "value"), Input("slider-dz", "value"),
         Input("slider-rx", "value"), Input("slider-ry", "value"), Input("slider-rz", "value")],
    )
    def _update_figure(dx, dy, dz, rx, ry, rz):
        rel_pos = np.array([dx, dy, dz], dtype=float)
        rel_euler = np.array([rx, ry, rz], dtype=float)
        return _build_mesh_plot_figure(
            pair_label=pair_label,
            assets=assets,
            anchor_attrs=anchor_attrs,
            movable_attrs=movable_attrs,
            rel_pos=rel_pos,
            rel_euler_deg=rel_euler,
        )

    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        [State("slider-dx", "value"), State("slider-dy", "value"), State("slider-dz", "value"),
         State("slider-rx", "value"), State("slider-ry", "value"), State("slider-rz", "value")],
    )
    def _save_xml(n_clicks, dx, dy, dz, rx, ry, rz):
        if not n_clicks:
            return ""

        rel_pos = np.array([dx, dy, dz], dtype=float)
        rel_euler = np.array([rx, ry, rz], dtype=float)
        new_pos, new_quat = _movable_from_relative_in_anchor_frame(
            anchor_pos, anchor_quat, rel_pos, rel_euler
        )

        local_lines = model_path.read_text(encoding="utf-8").splitlines(keepends=True)
        idx, _ = _get_geom_line_index_and_attrs(local_lines, movable_name)
        updated = local_lines[idx]
        updated = _replace_attr(updated, "pos", _fmt_vec(new_pos))
        updated = _replace_attr(updated, "quat", _fmt_vec(new_quat))
        local_lines[idx] = updated

        target = save_path.resolve() if save_path else model_path
        target.write_text("".join(local_lines), encoding="utf-8")
        return (
            f"Saved {movable_name} to {target} | pos={_fmt_vec(new_pos)} | quat={_fmt_vec(new_quat)}"
        )

    print(f"Launching interactive alignment UI on http://127.0.0.1:{port}")
    app.run(debug=False, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nudge movable geom alignment relative to an anchor geom in MuJoCo XML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Input model XML")
    parser.add_argument("--output", type=Path, default=None, help="Output XML (default: overwrite --model)")
    parser.add_argument("--side", choices=["left", "right", "both"], required=True, help="Which hand to edit")
    parser.add_argument(
        "--anchor-geom",
        type=str,
        default=None,
        help="Anchor geom name (generic mode; requires --movable-geom)",
    )
    parser.add_argument(
        "--movable-geom",
        type=str,
        default=None,
        help="Movable geom name (generic mode; requires --anchor-geom)",
    )
    parser.add_argument(
        "--pair-label",
        type=str,
        default=None,
        help="Optional label shown in prints/UI for generic mode",
    )
    parser.add_argument("--dx", type=float, default=0.0, help="Translation X (meters)")
    parser.add_argument("--dy", type=float, default=0.0, help="Translation Y (meters)")
    parser.add_argument("--dz", type=float, default=0.0, help="Translation Z (meters)")
    parser.add_argument("--rx", type=float, default=0.0, help="Rotation X (degrees)")
    parser.add_argument("--ry", type=float, default=0.0, help="Rotation Y (degrees)")
    parser.add_argument("--rz", type=float, default=0.0, help="Rotation Z (degrees)")
    parser.add_argument(
        "--frame",
        choices=["anchor", "movable", "world", "palm", "cover"],
        default="anchor",
        help="Frame in which delta translation/rotation is interpreted",
    )
    parser.add_argument("--show", action="store_true", help="Print alignment information")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print but do not write XML")
    parser.add_argument("--visualize", action="store_true", help="Generate a 3D anchor/movable preview HTML")
    parser.add_argument(
        "--visualize-html",
        type=Path,
        default=None,
        help="Output HTML path for --visualize (default: temporary file)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open generated HTML in your default graphical browser",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive slider UI (anchor fixed, movable adjustable)",
    )
    parser.add_argument(
        "--interactive-port",
        type=int,
        default=8057,
        help="Port for --interactive Dash UI",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Target XML path used by Save button in --interactive mode (default: --model)",
    )
    parser.add_argument(
        "--start-zero",
        action="store_true",
        help="Initialize interactive sliders at zero instead of current relative pose",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    out_path = args.output.resolve() if args.output else model_path

    lines = model_path.read_text(encoding="utf-8").splitlines(keepends=True)
    pairs = _resolve_geom_pairs(
        side=args.side,
        anchor_geom=args.anchor_geom,
        movable_geom=args.movable_geom,
        pair_label=args.pair_label,
    )

    if args.interactive:
        if len(pairs) != 1:
            raise RuntimeError(
                "Interactive mode requires exactly one anchor/movable pair "
                "(use --side left|right or explicit --anchor-geom/--movable-geom)."
            )
        pair = pairs[0]
        _run_interactive_ui(
            model_path=model_path,
            pair_label=pair["label"],
            anchor_name=pair["anchor"],
            movable_name=pair["movable"],
            save_path=args.save_model,
            port=args.interactive_port,
            start_zero=args.start_zero,
        )
        return

    do_edit = any(
        abs(v) > 0.0
        for v in [args.dx, args.dy, args.dz, args.rx, args.ry, args.rz]
    )

    for pair in pairs:
        label = pair["label"]
        anchor_name = pair["anchor"]
        movable_name = pair["movable"]

        _, anchor_attrs = _get_geom_line_index_and_attrs(lines, anchor_name)
        movable_idx, movable_attrs = _get_geom_line_index_and_attrs(lines, movable_name)

        anchor_pos = _parse_vec(anchor_attrs["pos"], 3)
        anchor_quat = _parse_vec(anchor_attrs["quat"], 4)
        movable_pos = _parse_vec(movable_attrs["pos"], 3)
        movable_quat = _parse_vec(movable_attrs["quat"], 4)

        if args.show or not do_edit:
            _print_alignment(
                label=label,
                anchor_name=anchor_name,
                movable_name=movable_name,
                anchor_pos=anchor_pos,
                movable_pos=movable_pos,
                anchor_q=anchor_quat,
                movable_q=movable_quat,
            )

        if do_edit:
            pos_new, quat_new = _apply_delta(
                movable_pos=movable_pos,
                movable_quat=movable_quat,
                anchor_pos=anchor_pos,
                anchor_quat=anchor_quat,
                delta_pos_local=np.array([args.dx, args.dy, args.dz], dtype=float),
                delta_rot_xyz_deg=np.array([args.rx, args.ry, args.rz], dtype=float),
                frame=args.frame,
            )

            updated = lines[movable_idx]
            updated = _replace_attr(updated, "pos", _fmt_vec(pos_new))
            updated = _replace_attr(updated, "quat", _fmt_vec(quat_new))
            lines[movable_idx] = updated

            print(
                f"[{label}] updated {movable_name}: pos={_fmt_vec(pos_new)} quat={_fmt_vec(quat_new)}"
            )

    if do_edit and not args.dry_run:
        out_path.write_text("".join(lines), encoding="utf-8")
        print(f"Wrote updated XML to: {out_path}")
    elif do_edit and args.dry_run:
        print("Dry run enabled: no file written")

    if args.visualize:
        # Visualize the current in-memory state (including dry-run edits).
        html_path = _build_visualization(
            lines,
            model_path=model_path,
            pairs=pairs,
            title_label=args.pair_label or args.side,
            html_path=args.visualize_html,
            open_browser=args.open_browser,
        )
        print(f"Wrote visualization HTML to: {html_path}")
        if not args.open_browser:
            print("Open this file in a graphical browser to inspect alignment.")


if __name__ == "__main__":
    main()
