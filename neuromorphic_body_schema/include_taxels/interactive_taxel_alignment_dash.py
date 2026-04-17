"""
interactive_taxel_alignment_dash.py

A fully interactive Dash GUI for manual alignment of a taxel patch to a robot part using sliders for rotation (x, y, z) and translation (x, y, z).

- Loads the part and taxel patch meshes using the same logic as optimize_taxel_alignment.py
- Initializes sliders to the best solution found so far (if available)
- Updates the visualization in real time as sliders are moved
- Allows exporting the current slider values for use as a new starting point for optimization

Usage:
    python interactive_taxel_alignment_dash.py --part l_upper_arm

Dependencies:
    - dash
    - plotly
    - numpy
    - trimesh
    - (your existing code for mesh loading and transformation)

To install dependencies:
    pip install dash plotly trimesh
"""

import argparse
import numpy as np
import sys
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

# Import mesh loading and transformation logic from your codebase
sys.path.append(os.path.dirname(__file__))
from neuromorphic_body_schema.include_taxels.optimize_taxel_alignment import (
    load_taxel_points,
    load_mesh,
    apply_part_transform,
    PARTS,
    POSITIONS_DIR,
)

def get_initial_params(part_name):
    # Try to load the best solution from previous optimization JSON
    import json
    from pathlib import Path
    from neuromorphic_body_schema.include_taxels.optimize_taxel_alignment import REPORT_JSON
    try:
        if Path(REPORT_JSON).exists():
            with open(REPORT_JSON, 'r', encoding='utf-8') as f:
                results = json.load(f)
            for entry in results:
                if entry.get('part') == part_name:
                    return entry['optimized']['delta_angles_deg'] + entry['optimized']['delta_offsets_m']
    except Exception:
        pass
    # Default: zeros
    return [0, 0, 0, 0, 0, 0]

def create_mesh_traces(part_mesh, taxel_points, part_name, params):
    # Ensure all params are floats (handle dicts from Dash slider)
    # Use initial values if any slider param is None
    def to_float(val, fallback):
        if val is None:
            return float(fallback)
        if isinstance(val, dict) and 'value' in val:
            return float(val['value'])
        return float(val)
    # Use the initial values from get_initial_params
    initial_params = get_initial_params(part_name)
    params = [to_float(p, fallback) for p, fallback in zip(params, initial_params)]
    # Apply transformation to taxel points
    rot = np.array(params[:3], dtype=float)
    trans = np.array(params[3:], dtype=float)
    taxel_points_t = apply_part_transform(taxel_points, part_name, rot, trans)
    # Create plotly traces
    part_trace = go.Mesh3d(
        x=part_mesh.vertices[:,0], y=part_mesh.vertices[:,1], z=part_mesh.vertices[:,2],
        i=part_mesh.faces[:,0], j=part_mesh.faces[:,1], k=part_mesh.faces[:,2],
        color='lightblue', opacity=0.5, name='Part'
    )
    taxel_trace = go.Scatter3d(
        x=taxel_points_t[:,0], y=taxel_points_t[:,1], z=taxel_points_t[:,2],
        mode='markers', marker=dict(size=4, color='orange'), name='Taxel Patch'
    )
    return [part_trace, taxel_trace]

def main():
    parser = argparse.ArgumentParser(description='Interactive taxel alignment tool (Dash GUI)')
    parser.add_argument('--part', type=str, required=True, help='Name of the part (e.g., l_upper_arm)')
    args = parser.parse_args()
    global part_name
    part_name = args.part
    config = next(cfg for cfg in PARTS if cfg.part_name == part_name)
    part_mesh = load_mesh(config.mesh_files, config.mesh_pos, config.mesh_quat_wxyz)
    taxel_points = load_taxel_points(POSITIONS_DIR / config.position_file, config.rebase)
    init_params = get_initial_params(part_name)

    param_names = ['rot_x (deg)', 'rot_y (deg)', 'rot_z (deg)', 'trans_x (m)', 'trans_y (m)', 'trans_z (m)']
    param_bounds = [
        (-360, 360),  # rot_x
        (-360, 360),  # rot_y
        (-360, 360),  # rot_z
        (-0.2, 0.2),  # trans_x
        (-0.2, 0.2),  # trans_y
        (-0.2, 0.2),  # trans_z
    ]
    param_min = [b[0] for b in param_bounds]
    param_max = [b[1] for b in param_bounds]

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2(f'Interactive Alignment: {part_name}'),
        dcc.Graph(id='mesh-figure', style={'height': '70vh'}),
        html.Div([
            html.Div([
                html.Label(name),
                dcc.Slider(
                    id=f'slider-{i}',
                    min=param_min[i],
                    max=param_max[i],
                    step=(param_max[i]-param_min[i])/200 if param_max[i] != param_min[i] else 0.01,
                    value=init_params[i],
                    marks={float(f'{param_min[i]:.2f}'): f'{param_min[i]:.2f}', float(f'{param_max[i]:.2f}'): f'{param_max[i]:.2f}'},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'margin': '10px'})
            for i, name in enumerate(param_names)
        ], style={'display': 'flex', 'flexDirection': 'column', 'width': '60%', 'margin': 'auto'}),
        html.Button('Export Params', id='export-btn', n_clicks=0, style={'margin': '20px'}),
        html.Div(id='export-output', style={'margin': '10px', 'fontWeight': 'bold'})
    ])

    @app.callback(
        Output('mesh-figure', 'figure'),
        [Input(f'slider-{i}', 'value') for i in range(6)],
        [State('mesh-figure', 'relayoutData')]
    )
    def update_figure(*params, relayoutData=None):
        traces = create_mesh_traces(part_mesh, taxel_points, part_name, list(params))
        fig = go.Figure(data=traces)
        camera = None
        if relayoutData and 'scene.camera' in relayoutData:
            camera = relayoutData['scene.camera']
        elif relayoutData:
            # Sometimes the camera dict is nested
            for k, v in relayoutData.items():
                if k.endswith('camera') and isinstance(v, dict):
                    camera = v
                    break
        fig.update_layout(
            scene=dict(aspectmode='data', camera=camera) if camera else dict(aspectmode='data'),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True
        )
        return fig

    @app.callback(
        Output('export-output', 'children'),
        [Input('export-btn', 'n_clicks')],
        [State(f'slider-{i}', 'value') for i in range(6)]
    )
    def export_params(n_clicks, *params):
        if n_clicks > 0:
            out = f'Exported params: {dict(zip(param_names, params))}'
            print(out)
            return out
        return ''

    app.run(debug=True)

if __name__ == '__main__':
    main()
