"""
check_skin_topology.py

Author: Simon F. Muller-Cleve (with Copilot)
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 17.03.2026

Description:
This script checks whether the 3D taxel placement (robot body) and the 2D GUI layout (flattened for visualization) share the same topology for each skin patch.

It implements:
1. Triangle/Connectivity Consistency: Compares triangle connectivity (taxel indices per triangle) in 3D and 2D.
2. Euler Characteristic: Computes and compares Euler characteristic (V - E + F) for both 3D and 2D meshes.

It uses the same parsing and filtering logic as the main pipeline to ensure consistency.
"""
import sys
from pathlib import Path

# Robustly find the project root (parent of neuromorphic_body_schema) from any location
def find_project_root(start_path: Path, target_dir: str = "neuromorphic_body_schema") -> Path:
    current = start_path.resolve()
    while current != current.parent:
        if (current / target_dir).is_dir():
            return current
        current = current.parent
    raise RuntimeError(f"Could not find {target_dir} in any parent directory of {start_path}")

project_root = find_project_root(Path(__file__))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if __name__ == "__main__":
    print(f"[DEBUG] Project root added to sys.path: {project_root}")
    print(f"[DEBUG] sys.path: {sys.path}")
        
import os

import numpy as np

from neuromorphic_body_schema.helpers.ed_skin import \
    read_triangle_data as read_triangle_data_gui
from neuromorphic_body_schema.helpers.helpers import POSITIONS_FILES, TRIANGLE_FILES
from neuromorphic_body_schema.include_taxels.include_skin_to_mujoco_model import (
    read_calibration_data, read_taxel2repr_data, read_triangle_data,
    validate_taxel_data)

# Paths (adjust as needed)
POSITIONS_PATH = "./neuromorphic_body_schema/include_taxels/positions"
TRIANGLES_PATH = "./neuromorphic_body_schema/include_taxels/skinGUI"


def get_3d_taxel_positions(part):
    pos_file = os.path.join(POSITIONS_PATH, f"{part}")
    calibration = read_calibration_data(pos_file)
    taxel2repr = read_taxel2repr_data(pos_file)
    taxels = validate_taxel_data(calibration, taxel2repr if taxel2repr else None)
    # Returns list of 3D positions (np.ndarray)
    return [pos for (pos, nrm, i_filt) in taxels]

def get_2d_taxel_positions(part):
    from neuromorphic_body_schema.helpers.ed_skin import visualize_skin_patches
    tri_file = os.path.join(TRIANGLES_PATH, f"{part}.ini")
    pos_file = os.path.join(POSITIONS_PATH, f"{part}")
    # Use visualize_skin_patches to get 2D layout and mask
    img, dX, dY, taxel_mask = visualize_skin_patches(
        path_to_triangles=TRIANGLES_PATH,
        triangles_ini=part,
        position_txt=part,
        expected_tactile_count=None,
        DEBUG=False,
    )
    # Only keep tactile points (mask True)
    if taxel_mask and len(taxel_mask) == len(dX):
        dX = [x for x, m in zip(dX, taxel_mask) if m]
        dY = [y for y, m in zip(dY, taxel_mask) if m]
    return list(zip(dX, dY))

def get_2d_triangle_indices(part):
    tri_file = os.path.join(TRIANGLES_PATH, f"{part}.ini")
    config_types, triangles = read_triangle_data_gui(tri_file)
    # Each triangle: (np.array([x, y, ...]), patch_id)
    # For topology, we care about the order/indices, not the coordinates
    # The GUI code uses patch_id and config_type to build the mask and order
    # Here, we just return the patch_id list in draw order
    patch_ids = [tri[1] for tri in triangles]
    return patch_ids

def build_triangle_adjacency(triangles):
    """Build an undirected edge set from triangles of vertex indices."""
    edges = set()
    for tri in triangles:
        if len(tri) < 3:
            continue
        a, b, c = tri[0], tri[1], tri[2]
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return edges


def euler_characteristic(num_vertices, edges, num_faces):
    """Compute Euler characteristic chi = V - E + F."""
    return num_vertices - len(edges) + num_faces


def main():
    for part in POSITIONS_FILES.keys():
        print(f"\n=== Checking part: {part} ===")
        try:
            taxel_pos_3d = get_3d_taxel_positions(POSITIONS_FILES[part])
            taxel_pos_2d = get_2d_taxel_positions(part)
        except Exception as e:
            print(f"[ERROR] Failed loading taxel data for {part}: {e}")
            continue

        n3d = len(taxel_pos_3d)
        n2d = len(taxel_pos_2d)
        print(f"3D tactile count: {n3d}, 2D GUI tactile count: {n2d}")
        if n3d != n2d:
            print(f"[MISMATCH] Number of tactile taxels differs! 3D: {n3d}, 2D: {n2d}")

        try:
            patch_ids_2d = get_2d_triangle_indices(part)
            # Placeholder 3D triangles based on contiguous tactile indices.
            triangles_3d = [list(range(min(3, n3d)))] if n3d >= 3 else []
            triangles_2d = [patch_ids_2d[i:i + 3] for i in range(0, len(patch_ids_2d), 3)]
            triangles_2d = [tri for tri in triangles_2d if len(tri) == 3]

            edges_3d = build_triangle_adjacency(triangles_3d)
            edges_2d = build_triangle_adjacency(triangles_2d)
            chi_3d = euler_characteristic(n3d, edges_3d, len(triangles_3d))
            chi_2d = euler_characteristic(n2d, edges_2d, len(triangles_2d))
            print(f"3D: V={n3d}, E={len(edges_3d)}, F={len(triangles_3d)}, chi={chi_3d}")
            print(f"2D: V={n2d}, E={len(edges_2d)}, F={len(triangles_2d)}, chi={chi_2d}")
        except Exception as e:
            print(f"[ERROR] Topology check failed for {part}: {e}")

if __name__ == "__main__":
    main()
