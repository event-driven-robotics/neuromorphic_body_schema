"""Tests for mesh-projected, symmetric fingertip taxel placement."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from neuromorphic_body_schema.include_taxels.fingertip_taxel_projection import (
    FINGERTIP_BODY_TO_PART,
    FINGERTIP_BODY_TO_PROJECTION_MESH,
    FINGERTIP_CENTERLINE_INDICES,
    FINGERTIP_FREE_X_INDICES,
    FINGERTIP_MIRRORED_PAIRS,
    FINGERTIP_ROUNDED_TIP_CENTER,
    FINGERTIP_X_ALIGNED_GROUPS,
    LEGACY_FINGERTIP_TAXELS,
    _project_one_point_radially,
    enforce_bilateral_symmetry,
    project_all_legacy_fingertip_taxels,
    project_legacy_fingertip_taxels,
)


def _nearest_surface_distances(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> np.ndarray:
    """Compute point-to-triangle distances without optional rtree."""

    distances = []
    for point in points:
        repeated = np.repeat(point.reshape(1, 3), len(mesh.triangles), axis=0)
        closest = trimesh.triangles.closest_point(mesh.triangles, repeated)
        distances.append(float(np.min(np.linalg.norm(closest - point, axis=1))))
    return np.asarray(distances)


def test_legacy_fingertip_symmetry_topology_covers_all_taxels() -> None:
    paired = {index for pair in FINGERTIP_MIRRORED_PAIRS for index in pair}
    centerline = set(FINGERTIP_CENTERLINE_INDICES)

    assert paired.isdisjoint(centerline)
    assert paired | centerline == set(range(len(LEGACY_FINGERTIP_TAXELS)))
    assert FINGERTIP_X_ALIGNED_GROUPS == (
        (0, 1, 2),
        (3, 4),
        (5, 6, 7),
        (8, 10),
    )
    assert FINGERTIP_FREE_X_INDICES == (9, 11)


def test_projection_targets_detailed_full_fingertip_meshes() -> None:
    assert set(FINGERTIP_BODY_TO_PROJECTION_MESH) == set(FINGERTIP_BODY_TO_PART)
    assert all(
        mesh_name.startswith("full_") and mesh_name.endswith("3")
        for mesh_name in FINGERTIP_BODY_TO_PROJECTION_MESH.values()
    )


def test_enforce_bilateral_symmetry_uses_reflected_pair_mean() -> None:
    points = np.array(
        [
            [1.0, 2.0, -4.0],
            [3.0, 6.0, 5.0],
            [7.0, 8.0, 9.0],
        ]
    )

    symmetric = enforce_bilateral_symmetry(
        points,
        mirrored_pairs=((0, 2),),
        centerline_indices=(1,),
        symmetry_axis=2,
    )

    np.testing.assert_allclose(symmetric[0], [4.0, 5.0, -6.5])
    np.testing.assert_allclose(symmetric[2], [4.0, 5.0, 6.5])
    np.testing.assert_allclose(symmetric[1], [3.0, 6.0, 0.0])


def test_radial_projection_selects_outermost_mesh_intersection() -> None:
    vertices = np.array(
        [
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 2.0, -1.0],
            [1.0, 2.0, -1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]])

    projected = _project_one_point_radially(
        mesh=mesh, point=np.array([0.0, 1.1, 0.0])
    )

    np.testing.assert_allclose(projected, [0.0, 2.0, 0.0])


def test_projected_fingertip_taxels_preserve_x_alignment_and_symmetry() -> None:
    reflection = np.array([1.0, 1.0, -1.0])

    for body_name in FINGERTIP_BODY_TO_PART:
        result = project_legacy_fingertip_taxels(body_name)
        points = result.projected_points

        fixed_x_indices = sorted(
            set(range(len(points))) - set(FINGERTIP_FREE_X_INDICES)
        )
        np.testing.assert_array_equal(
            points[fixed_x_indices, 0], LEGACY_FINGERTIP_TAXELS[fixed_x_indices, 0]
        )
        for group in FINGERTIP_X_ALIGNED_GROUPS:
            assert len(set(points[list(group), 0])) == 1

        assert np.array_equal(
            points[list(FINGERTIP_CENTERLINE_INDICES), 2], np.zeros(4)
        )
        for first, second in FINGERTIP_MIRRORED_PAIRS:
            np.testing.assert_array_equal(points[first], reflection * points[second])

        # Grouped taxels remain on constant-x yz rays. The two unique-x
        # centerline taxels may follow xy rays around the rounded tip.
        for index, (original, projected) in enumerate(
            zip(LEGACY_FINGERTIP_TAXELS, points)
        ):
            if index in FINGERTIP_FREE_X_INDICES:
                baseline_ray = original - FINGERTIP_ROUNDED_TIP_CENTER
                projected_ray = projected - FINGERTIP_ROUNDED_TIP_CENTER
                assert float(np.linalg.norm(np.cross(baseline_ray, projected_ray))) <= 1e-12
                assert projected[2] == 0.0
                continue
            cross_2d = original[1] * projected[2] - original[2] * projected[1]
            assert abs(float(cross_2d)) <= 1e-12
            assert float(original[1:] @ projected[1:]) >= 0.0

        # Pair averaging introduces only a tiny offset from the detailed mesh.
        distances = _nearest_surface_distances(result.mesh, points)
        assert float(np.max(distances)) <= 0.0001


def test_project_all_uses_requested_model_and_part_names() -> None:
    model_xml = Path(
        "neuromorphic_body_schema/models/icub_v2_full_body_improved.xml"
    )
    projected = project_all_legacy_fingertip_taxels(model_xml=model_xml)

    assert set(projected) == set(FINGERTIP_BODY_TO_PART.values())
    assert all(points.shape == (12, 3) for points in projected.values())


def test_generated_model_uses_projected_positions_in_fingertip_local_frame() -> None:
    source_model = Path(
        "neuromorphic_body_schema/models/icub_v2_full_body_improved.xml"
    )
    generated_model = source_model.with_name(
        f"{source_model.stem}_contact_sensors{source_model.suffix}"
    )
    expected = project_all_legacy_fingertip_taxels(model_xml=source_model)
    root = ET.parse(generated_model).getroot()
    bodies = {body.get("name"): body for body in root.iter("body")}

    for body_name, part_name in FINGERTIP_BODY_TO_PART.items():
        body = bodies[body_name]
        sites = {site.get("name"): site for site in body.findall("site")}
        for index, expected_position in enumerate(expected[part_name]):
            site = sites[f"{part_name}_taxel_{index}"]
            actual_position = np.fromstring(site.get("pos", ""), sep=" ")
            np.testing.assert_array_equal(actual_position, expected_position)
