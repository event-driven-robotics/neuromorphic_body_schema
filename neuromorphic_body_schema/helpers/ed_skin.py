"""
ed_skin.py

Author: Simon F. Muller-Cleve
Affiliation: Istituto Italiano di Tecnologia (IIT)
Department: Event-Driven Perception for Robotics (EDPR)
Date: 29.04.2025

Description:
This module provides functionality for simulating event-based tactile sensors and integrating them with the iCub robot's
skin system. It includes classes and functions for generating events based on changes in taxel intensity, visualizing
tactile data, and managing skin sensor configurations.

Classes:
- SkinEventSimulator: Simulates an event-based skin sensor by generating events based on taxel intensity changes.
- ICubSkin: Represents the iCub robot's skin system, integrating tactile sensors and visualization.

Functions:
- visualize_skin_patches: Visualizes the layout of skin patches based on triangle configurations.
- make_skin_event_frame: Updates a visual representation of skin events on a tactile sensor image.
- read_triangle_data: Reads triangle data from a given file path.

"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from .draw_pads import (
    fingertip3L,
    fingertip3R,
    palmL,
    palmR,
    triangle_10pad,
)
from .helpers import (
    ARM_PATCH_ID_REMAP,
    KEY_MAPPING,
    POSITIONS_FILES,
    TRIANGLE_FILES,
    TRIANGLE_INI_PATH,
)

# set background color to gray
background_color = (50, 50, 50)
# set taxel contours to gold
taxel_color = (0, 215, 255)
# non-tactile channels (thermal pads / unused slots) drawn as dark-grey outlines
non_tactile_color = (120, 120, 120)
line_color = (255, 255, 255)
# set the color for the events
red = (0, 0, 255)  # positive events
blue = (255, 0, 0)  # negative events


class SkinEventSimulator:
    """
    Simulates an event-based skin sensor by generating events based on changes in taxel (tactile pixel) intensity.

    Attributes:
        Cp (float): Positive contrast threshold.
        Cm (float): Negative contrast threshold.
        sigma_Cp (float): Standard deviation for noise in the positive contrast threshold.
        sigma_Cm (float): Standard deviation for noise in the negative contrast threshold.
        log_eps (float): Small constant added to avoid log(0) when using logarithmic data.
        refractory_period_ns (int): Minimum time (in nanoseconds) between consecutive events for the same taxel.
        last_data (np.array): The last processed taxel data.
        ref_values (np.array): Reference values for contrast threshold crossings.
        last_event_timestamp (np.array): Timestamps of the last event for each taxel.
        current_time (int): Current simulation time.
        size (int): Number of taxels in the sensor.
    """

    def __init__(
        self,
        data,
        time,
        Cp=0.5,
        Cm=0.5,
        sigma_Cp=0.01,
        sigma_Cm=0.01,
        log_eps=1e-6,
        refractory_period_ns=100,
    ):
        """Create an event-based skin simulator from initial taxel readings.

        Args:
            data (np.ndarray): Initial 1D taxel intensity/pressure values.
            time (int): Initial timestamp in nanoseconds.
            Cp (float): Positive contrast threshold for ON events.
            Cm (float): Negative contrast threshold for OFF events.
            sigma_Cp (float): Std-dev of Gaussian noise applied to ``Cp``.
            sigma_Cm (float): Std-dev of Gaussian noise applied to ``Cm``.
            log_eps (float): Reserved epsilon for log-domain workflows.
            refractory_period_ns (int): Minimum interval between two accepted
                events on the same taxel.
        """

        self.Cp = Cp
        self.Cm = Cm
        self.sigma_Cp = sigma_Cp
        self.sigma_Cm = sigma_Cm
        self.log_eps = log_eps
        self.refractory_period_ns = refractory_period_ns
        logging.info(
            f"Initialized event skin simulator with sensor size: {data.shape}")
        logging.info(
            f"and contrast thresholds: C+ = {self.Cp}, C- = {self.Cm}")

        self.last_data = data.copy()
        self.ref_values = data.copy()
        self.last_event_timestamp = np.zeros(data.shape, dtype=np.ulonglong)
        self.current_time = time
        self.size = data.shape[0]

    def skinCallback(self, data, time):
        """Process one taxel sample and emit events for threshold crossings.

        Args:
            data (np.ndarray): Current 1D taxel values.
            time (int): Current timestamp in nanoseconds. Must be strictly
                greater than the previous callback timestamp.

        Returns:
            np.ndarray: Event array of shape ``(N, 3)`` with rows
                ``(taxel_id, t_ns, polarity)`` where polarity is ``True`` for
                positive events and ``False`` for negative events. Returns an
                empty object array when no events are generated.

        Notes:
            - Events are sorted by timestamp before returning.
            - Refractory filtering is applied independently for each taxel.
        """
        assert time >= 0

        tolerance = 1e-6
        delta_t_ns = time - self.current_time
        assert delta_t_ns > 0

        itdt = np.asarray(data)
        it = self.last_data
        prev_cross = self.ref_values

        delta = itdt - it
        changed_mask = np.abs(delta) > tolerance
        changed_idx = np.where(changed_mask)[0]

        events = []

        # Only process changed taxels
        for idx in changed_idx:
            it0 = it[idx]
            it1 = itdt[idx]
            ref = prev_cross[idx]
            pol = 1.0 if it1 >= it0 else -1.0
            C = self.Cp if pol > 0 else self.Cm
            sigma_C = self.sigma_Cp if pol > 0 else self.sigma_Cm

            curr_cross = ref
            while True:
                # Add noise to threshold
                C_eff = C + (np.random.normal(0, sigma_C)
                             if sigma_C > 0 else 0)
                C_eff = max(0.01, C_eff)
                curr_cross += pol * C_eff

                # Check if crossing occurred in this interval
                if (pol > 0 and curr_cross > it0 and curr_cross <= it1) or (
                    pol < 0 and curr_cross < it0 and curr_cross >= it1
                ):

                    # Interpolate event time
                    edt = int(abs((curr_cross - it0) *
                              delta_t_ns / (it1 - it0)))
                    t_evt = self.current_time + edt

                    # Refractory check
                    last_stamp = self.last_event_timestamp[idx]
                    dt = t_evt - last_stamp
                    if last_stamp == 0 or dt >= self.refractory_period_ns:
                        events.append((idx, t_evt, pol > 0))
                        self.last_event_timestamp[idx] = t_evt
                        self.ref_values[idx] = curr_cross
                    else:
                        # Don't update ref_values if event is dropped
                        pass
                else:
                    break

        # Update state for next call
        self.current_time = time
        self.last_data = itdt.copy()

        # Sort events by timestamp
        if len(events):
            events = np.array(events)
            events = events[np.argsort(events[:, 1])]
        else:
            events = np.empty((0, 3), dtype=object)
        return events


def _build_taxel_mask(
    triangles_ini: str,
    patch_entries: list[tuple[int, str]],
) -> list[bool]:
    """Build a per-drawn-point boolean mask from ``taxel2Repr``.

    Drawn points map back to ``taxel2Repr`` slots according to config type:
    - ``triangle_10pad``: 1 module x 10 drawn points with offsets
      ``[0,1,2,3,4,5,7,8,9,11]``.
    - ``fingertip3R/L``: 1 module x 12 drawn points with offsets ``0..11``.
    - ``palmR/L``: 4 consecutive modules x 12 drawn points each.

    Returns a flat list of booleans, one per drawn point, True if the point
    corresponds to a tactile channel (``taxel2Repr >= 0``).
    """
    from pathlib import Path
    from neuromorphic_body_schema.include_taxels.include_skin_to_mujoco_model import (
        read_calibration_data,
        read_taxel2repr_data,
    )

    pos_fn = POSITIONS_FILES.get(triangles_ini)
    if pos_fn is None:
        # Unknown part - treat every point as tactile.
        return []

    pos_dir = Path(__file__).parent.parent / "include_taxels" / "positions"
    pos_path = pos_dir / pos_fn
    if not pos_path.exists():
        return []

    repr_vals = read_taxel2repr_data(str(pos_path))
    if not repr_vals:
        return []

    calibration = read_calibration_data(str(pos_path))

    def _is_tactile_from_calibration(slot_idx: int) -> bool:
        """Fallback tactile check from 6D calibration row.

        A channel is considered tactile if either its 3D position or normal
        vector is non-zero. This mirrors validate_taxel_data behavior.
        """
        if slot_idx < 0 or slot_idx >= len(calibration):
            return False
        row = calibration[slot_idx]
        if row.shape[0] < 6:
            return False
        pos = row[:3]
        nrm = row[3:6]
        return np.linalg.norm(pos) != 0 or np.linalg.norm(nrm) != 0

    # Within each 12-slot block, the drawn-point intra-module order after
    # triangle_10pad drops positions 6 and 10 is:
    # slot offsets: 0,1,2,3,4,5, 7,8,9, 11
    drawn_offsets = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]

    full_offsets = list(range(12))
    patch_id_remap = ARM_PATCH_ID_REMAP.get(triangles_ini, {})

    mask = []
    for pid, config_type in patch_entries:
        # Some arm triangle files require a deterministic patch-id translation
        # so geometry patch IDs map to the tactile module IDs used in
        # positions/taxel2Repr.
        mapped_pid = patch_id_remap.get(pid, pid)
        if config_type == "triangle_10pad":
            module_offsets = [drawn_offsets]
        elif config_type in {"fingertip3R", "fingertip3L"}:
            module_offsets = [full_offsets]
        elif config_type in {"palmR", "palmL"}:
            # Palm occupies four consecutive 12-slot modules.
            module_offsets = [full_offsets, full_offsets, full_offsets, full_offsets]
        else:
            module_offsets = [drawn_offsets]

        for module_idx, offs in enumerate(module_offsets):
            base = (mapped_pid + module_idx) * 12
            if base + 11 >= len(repr_vals):
                # taxel2Repr coverage can be shorter than calibration. In this
                # case, fall back to the calibration row content.
                for off in offs:
                    mask.append(_is_tactile_from_calibration(base + off))
                continue
            for off in offs:
                mask.append(repr_vals[base + off] >= 0)
    return mask


def visualize_skin_patches(
    path_to_triangles,
    triangles_ini,
    expected_tactile_count: int | None = None,
    DEBUG=False,
):
    """Build a drawable taxel layout image for one skin patch configuration.

    Args:
        path_to_triangles (str): Directory containing ``*.ini`` patch layout
            files.
        triangles_ini (str): Basename of the patch configuration file
            (without ``.ini``).
        expected_tactile_count (int | None): Optional expected number of
            tactile channels for this patch (for example length of incoming
            sensor vector). If provided, the generated mask is reconciled to
            this count.
        DEBUG (bool): If True, also saves diagnostic geometry figures.

    Returns:
        tuple[np.ndarray, list[int], list[int], list[bool]]:
            - ``img`` — base BGR background with taxel/triangle outlines.  Non-
              tactile taxels (thermal pads, unused slots) are drawn with
              ``non_tactile_color`` so the user can see their position but
              understands they never respond.
            - ``dX``, ``dY`` — integer pixel coordinates for **all** drawn
              points (tactile and non-tactile).
            - ``taxel_mask`` — list of booleans, one per drawn point, True if
                            the point is a genuine tactile channel.
    """

    config_types, triangles = read_triangle_data(
        f"{path_to_triangles}/{triangles_ini}.ini"
    )
    patch_ID = []
    dX = []
    dY = []
    dXv = []
    dYv = []
    patch_entries_ordered = []  # (patch_id, config_type) in draw order
    scale = 3.0
    for tri, config_type in zip(triangles, config_types):
        cx, cy, th, lr_mirror = tri[0][0], tri[0][1], tri[0][2], int(tri[0][4])
        patch_ID.append(tri[1])
        patch_entries_ordered.append((tri[1], config_type))
        if config_type == "triangle_10pad":
            to_draw = triangle_10pad(cx=cx, cy=cy, th=th, lr_mirror=lr_mirror)
            # remove the thermal pads
            to_remove = [1, 5]  # always at the same position
            to_draw = (
                [v for idx, v in enumerate(to_draw[0]) if idx not in to_remove],
                [v for idx, v in enumerate(to_draw[1]) if idx not in to_remove],
                list(to_draw[2]),
                list(to_draw[3]),
            )

        elif config_type == "fingertip3R":
            to_draw = fingertip3R(cx=cx, cy=cy, th=th, lr_mirror=lr_mirror)
        elif config_type == "fingertip3L":
            to_draw = fingertip3L(cx=cx, cy=cy, th=th, lr_mirror=lr_mirror)
        elif config_type == "palmR":
            to_draw = palmR(cx=cx, cy=cy, th=th, lr_mirror=1)
            for i in range(len(to_draw[0])):
                to_draw[0][i] = to_draw[0][i] + 20
        elif config_type == "palmL":
            to_draw = palmL(cx=cx, cy=cy, th=th, lr_mirror=0)
            for i in range(len(to_draw[0])):
                to_draw[0][i] = to_draw[0][i] - 20
        else:
            logging.error("Unknown config type")

        # Normalize to float lists so incremental geometric shifts stay type-consistent.
        to_draw = (
            [float(v) for v in to_draw[0]],
            [float(v) for v in to_draw[1]],
            [float(v) for v in to_draw[2]],
            [float(v) for v in to_draw[3]],
        )
        # rearrange some triangles to make the fig more compact
        if (
            triangles_ini == "left_forearm_V2" or triangles_ini == "right_forearm_V2"
        ) and tri[1] in [16, 17, 19, 22, 24, 25, 28, 29]:
            for i in range(len(to_draw[1])):
                to_draw[1][i] += 40.0
            for i in range(len(to_draw[3])):
                to_draw[3][i] += 40.0
        if triangles_ini == "left_leg_upper" and tri[1] in [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            19,
            20,
            21,
            22,
            25,
            26,
            27,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            44,
            45,
            53,
            54,
            56,
            57,
            58,
            64,
            65,
            66,
            76,
            78,
            79,
        ]:
            for i in range(len(to_draw[1])):
                to_draw[1][i] += 83.0
            for i in range(len(to_draw[3])):
                to_draw[3][i] += 83.0
        if triangles_ini == "right_leg_upper" and tri[1] in [
            0,
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            19,
            20,
            21,
            22,
            25,
            26,
            27,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            44,
            45,
            51,
            53,
            54,
            57,
            58,
            59,
            64,
            65,
            66,
            67,
            76,
        ]:
            for i in range(len(to_draw[1])):
                to_draw[1][i] += 20.0
            for i in range(len(to_draw[3])):
                to_draw[3][i] += 20.0
        dX.extend(to_draw[0])
        dY.extend(to_draw[1])
        dXv.append(to_draw[2])
        dYv.append(to_draw[3])

    # let's set everything with repect to 0,0
    dX_min = np.min(dX)
    dY_min = np.min(dY)

    dX -= dX_min
    dY -= dY_min

    if not "hand" in triangles_ini:
        for i in range(len(dXv)):
            for j in range(len(dXv[i])):
                dXv[i][j] -= dX_min
                dYv[i][j] -= dY_min

    if DEBUG:
        # now we can draw the triangles
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dX, dY, marker="o")
        if not "hand" in triangles_ini:
            # draw the triangles
            for i in range(len(dXv)):
                for j in range(len(dXv[i])):
                    ax.plot(
                        [dXv[i][j - 1], dXv[i][j]],
                        [dYv[i][j - 1], dYv[i][j]],
                        linewidth=0.2,
                        color="black",
                    )
                ax.plot(
                    [dXv[i][-1], dXv[i][0]],
                    [dYv[i][-1], dYv[i][0]],
                    linewidth=0.2,
                    color="black",
                )  # Close the triangle
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(
            f"./neuromorphic_body_schema/figures/{triangles_ini}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # scale
    dX = dX * scale
    dY = dY * scale
    if not "hand" in triangles_ini:
        for i in range(len(dXv)):
            for j in range(len(dXv[i])):
                dXv[i][j] = dXv[i][j] * scale
                dYv[i][j] = dYv[i][j] * scale

    # flip axis and shift to positive values
    dY = -dY
    dYv_min = np.abs(np.min(dY))
    dY += dYv_min
    if not "hand" in triangles_ini:
        for i in range(len(dYv)):
            for j in range(len(dYv[i])):
                dYv[i][j] = -dYv[i][j] + dYv_min

    # find the width and height of the image
    width = np.max(dX)
    offset_x = width * 0.2
    width += offset_x
    width = int(width + 0.5)
    # offset_x = int((offset_x/2))

    height = np.max(dY)
    offset_y = height * 0.2
    height += offset_y
    height = int(height + 0.5)
    # offset_y = int((offset_y/2))

    # convert to int and apply offset to give somoe extra space at the corners
    dX = [int(x + 0.5 + offset_x / 2) for x in dX]
    dY = [int(y + 0.5 + offset_y / 2) for y in dY]
    if not "hand" in triangles_ini:
        for i in range(len(dYv)):
            for j in range(len(dYv[i])):
                dXv[i][j] = int(dXv[i][j] + 0.5 + offset_x / 2)
                dYv[i][j] = int(dYv[i][j] + 0.5 + offset_y / 2)

    if DEBUG:
        # now we can draw the triangles
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dX, dY, marker="o")
        if not "hand" in triangles_ini:
            # draw the triangles
            for i in range(len(dXv)):
                for j in range(len(dXv[i])):
                    ax.plot(
                        [dXv[i][j - 1], dXv[i][j]],
                        [dYv[i][j - 1], dYv[i][j]],
                        linewidth=0.2,
                        color="black",
                    )
                ax.plot(
                    [dXv[i][-1], dXv[i][0]],
                    [dYv[i][-1], dYv[i][0]],
                    linewidth=0.2,
                    color="black",
                )  # Close the triangle
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        fig.savefig(
            f"./neuromorphic_body_schema/figures/{triangles_ini}_processed.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # Build tactile mask from taxel2Repr before drawing
    taxel_mask = _build_taxel_mask(triangles_ini, patch_entries_ordered)
    if taxel_mask and len(taxel_mask) != len(dX):
        logging.warning(
            "Taxel mask length (%d) does not match drawn points (%d) for %s; "
            "falling back to all tactile.",
            len(taxel_mask),
            len(dX),
            triangles_ini,
        )
        taxel_mask = []
    if taxel_mask and expected_tactile_count is not None:
        tactile_count = int(sum(taxel_mask))
        if tactile_count != expected_tactile_count:
            logging.warning(
                "Taxel mask tactile count (%d) does not match expected (%d) for %s; "
                "reconciling mask.",
                tactile_count,
                expected_tactile_count,
                triangles_ini,
            )
            if tactile_count < expected_tactile_count:
                for idx, is_tactile in enumerate(taxel_mask):
                    if not is_tactile:
                        taxel_mask[idx] = True
                        tactile_count += 1
                        if tactile_count == expected_tactile_count:
                            break
            else:
                for idx in range(len(taxel_mask) - 1, -1, -1):
                    if taxel_mask[idx]:
                        taxel_mask[idx] = False
                        tactile_count -= 1
                        if tactile_count == expected_tactile_count:
                            break

    # Create a blank image
    img = np.full((height, width, 3), background_color, dtype=np.uint8)
    # Draw the triangles on the image; colour non-tactile points distinctly so
    # the user can see their position but understands they never respond.
    for taxel_counter in range(len(dX)):
        is_tactile = (not taxel_mask) or taxel_mask[taxel_counter]
        color = taxel_color if is_tactile else non_tactile_color
        cv2.circle(
            img, (dX[taxel_counter], dY[taxel_counter]), 5, color, 1
        )  # RGB color
    if not "hand" in triangles_ini:
        for i in range(len(dXv)):
            for j in range(len(dXv[i])):
                pt1 = (dXv[i][j - 1], dYv[i][j - 1])
                pt2 = (dXv[i][j], dYv[i][j])
                cv2.line(img, pt1, pt2, line_color, 1)
            pt1 = (dXv[i][-1], dYv[i][-1])
            pt2 = (dXv[i][0], dYv[i][0])
            cv2.line(img, pt1, pt2, line_color, 1)

    return (img, dX, dY, taxel_mask)


def read_triangle_data(file_path: str) -> tuple:
    """Parse a skin geometry ``.ini`` file into config labels and parameters.

    Args:
        file_path (str): Absolute or relative path to a triangle config file.

    Returns:
        tuple[list[str], list[tuple[np.ndarray, int]]]:
            - List of configuration tags (for example ``triangle_10pad``).
            - List of ``(triangle_params, patch_id)`` tuples.
    """
    triangles = []
    config_type = []
    start_found = False
    read_header = False
    with open(file_path, "r") as file:
        for line in file:
            if start_found and not read_header:
                stripped = line.strip()
                # Some skinGui files include comments/metadata lines in the
                # [SENSORS] block (for example "#robotPart ...").
                if not stripped or stripped.startswith("#") or stripped.startswith(";"):
                    continue
                entry = stripped.split()
                conf_type = entry[0]
                config_type.append(conf_type)
                triangle = list(map(float, entry[1:]))
                triangles.append((np.array(triangle[1:]), int(triangle[0])))
            if read_header:
                read_header = False
                continue
            if "[SENSORS]" in line:
                # TODO for the hand files the header comes before the [SENSORS] line, for now I change the ini files locally. Find a better solution later.
                # TODO also in torso.ini removes #leftlower
                start_found = True
                read_header = True
    return config_type, triangles


def make_skin_event_frame(
    img,
    events,
    locations,
    taxel_mask: list[bool] | None = None,
) -> np.ndarray:
    """Render an event-driven skin frame from taxel events.

    Args:
        img (np.ndarray): Base BGR image to draw onto (modified in-place).
        events (np.ndarray): Event array with rows
            ``(taxel_id, t_ns, polarity)``.
        locations (tuple[list[int], list[int]]): Taxel pixel coordinates as
            ``(x_coords, y_coords)``.
        taxel_mask (list[bool] | None): Per-point boolean mask where True means
            the point is a genuine tactile channel.  Non-tactile points are
            never coloured with event colours; they stay at background colour.
            Pass ``None`` or an empty list to treat all points as tactile
            (legacy behaviour / hand patches).

    Returns:
        np.ndarray: Updated BGR image where ON events are red and OFF events
            are blue.

    Notes:
        - When no events are present, tactile taxel centers are reset to
          background color for this frame.
    """

    # events = taxel_ID, t, pol
    event_by_taxel = {}
    if len(events):
        # Keep the most recent polarity per tactile taxel index.
        for evt in events:
            event_by_taxel[int(evt[0])] = bool(evt[-1] > 0)

    tactile_idx = 0
    for i, loc in enumerate(zip(locations[0], locations[1])):
        # Skip non-tactile positions: leave them as drawn by visualize_skin_patches.
        if taxel_mask and not taxel_mask[i]:
            continue

        if not len(events):
            # No event happened, return to blank for tactile points.
            cv2.circle(img, (int(loc[0]), int(loc[1])), 4, background_color, -1)
            tactile_idx += 1
            continue

        if tactile_idx in event_by_taxel:
            if event_by_taxel[tactile_idx]:
                cv2.circle(img, (int(loc[0]), int(loc[1])), 4, red, -1)
            else:
                cv2.circle(img, (int(loc[0]), int(loc[1])), 4, blue, -1)
        tactile_idx += 1
    return img


def make_skin_raw_frame(
    img,
    taxel_data,
    locations,
    taxel_mask: list[bool] | None = None,
) -> np.ndarray:
    """Render raw skin values as a per-taxel color map.

    Args:
        img (np.ndarray): Base BGR image to draw onto (modified in-place).
        taxel_data (np.ndarray | list[float]): Raw tactile-channel values for
            one patch.  The length must equal the number of **tactile** points
            in ``locations`` (i.e. points where ``taxel_mask[i]`` is True, or
            all points when ``taxel_mask`` is empty).
        locations (tuple[list[int], list[int]]): Pixel coordinates for **all**
            drawn points, including non-tactile ones.
        taxel_mask (list[bool] | None): Per-point boolean mask (same length as
            ``locations[0]``).  Non-tactile points are skipped; their
            appearance is preserved from the base image.  Pass ``None`` or an
            empty list to treat all points as tactile.

    Returns:
        np.ndarray: Updated BGR image with normalized low-to-high tactile
            values mapped from blue to red.
    """
    values = np.asarray(taxel_data, dtype=float)
    if values.size == 0:
        return img

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        norm_values = np.zeros_like(values)
    else:
        norm_values = (values - vmin) / (vmax - vmin)

    tactile_idx = 0  # index into taxel_data / norm_values
    for i, loc in enumerate(zip(locations[0], locations[1])):
        if taxel_mask and not taxel_mask[i]:
            continue  # non-tactile: leave dot as drawn by visualize_skin_patches
        if tactile_idx >= len(norm_values):
            break
        level = int(np.clip(norm_values[tactile_idx] * 255.0, 0, 255))
        # BGR: blue->red gradient for low->high pressure values.
        color = (255 - level, 0, level)
        cv2.circle(img, (int(loc[0]), int(loc[1])), 4, color, -1)
        tactile_idx += 1
    return img


class ICubSkin:
    """
    Represents the iCub robot's skin system, integrating event-based tactile sensors and visualization.

    Attributes:
        esim (list): A list of SkinEventSimulator instances for each skin patch.
        taxel_locs (dict): A dictionary mapping skin patches to their taxel locations (x, y coordinates).
        imgs (dict): A dictionary mapping skin patches to their visual representations.
        grouped_sensors (dict): A dictionary containing grouped sensor data for each skin patch.
        show_skin (bool): Whether to display the skin event visualizations.
        DEBUG (bool): Whether to enable debug logging.
    """

    def __init__(self, time, grouped_sensors, skin="all", skin_mode="frame_based", show_raw_feed=True, show_ed_feed=False, DEBUG=False):
        """Initialize skin patch renderers and event simulators.

        Args:
            time (int): Initial simulation timestamp in nanoseconds.
            grouped_sensors (dict): Dynamic sensor accessor/group mapping used
                to fetch current taxel values per patch.
            skin (str | list[str]): Skin selection identifier (reserved for
                filtering behavior).
            skin_mode (str): ``"frame_based"`` or ``"event_driven"`` mode tag.
            show_raw_feed (bool): If True, open and update raw skin windows.
            show_ed_feed (bool): If True, open and update event skin windows.
            DEBUG (bool): If True, enable additional logging.

        Notes:
            - For each configured patch in ``TRIANGLE_FILES``, a dedicated
              ``SkinEventSimulator`` is created.
            - If event feed visualization is enabled, each patch gets a
              threshold slider that updates ``Cp`` and ``Cm``.
        """

        self.esim = []
        self.taxel_locs = {}
        self.taxel_masks = {}   # per-patch bool list: True = tactile channel
        self.imgs = {}
        self.ed_imgs = {}
        self.latest_raw_frames = {}
        self.latest_ed_frames = {}
        self.grouped_sensors = grouped_sensors
        self.show_raw_feed = show_raw_feed
        self.show_ed_feed = show_ed_feed
        self.skin = skin  # list or str ofskin parts to visualize
        self.skin_mode = skin_mode
        self.show_raw_feed = show_raw_feed
        self.show_ed_feed = show_ed_feed
        self.DEBUG = DEBUG
        for triangle_ini in TRIANGLE_FILES:
            # TODO make sure we hand over the right data here
            if "right_hand" in triangle_ini:
                # TODO double check the order of the taxels!
                taxel_data = []
                for key in KEY_MAPPING["r_hand"]:
                    taxel_data.extend(self.grouped_sensors[key])
            elif "left_hand" in triangle_ini:
                taxel_data = []
                for key in KEY_MAPPING["l_hand"]:
                    taxel_data.extend(self.grouped_sensors[key])
            else:
                taxel_data = self.grouped_sensors[KEY_MAPPING[triangle_ini]]
            self.esim.append(SkinEventSimulator(np.array(taxel_data), time))
            if self.show_raw_feed or self.show_ed_feed:
                img, x, y, mask = visualize_skin_patches(
                    path_to_triangles=TRIANGLE_INI_PATH,
                    triangles_ini=triangle_ini,
                    expected_tactile_count=len(taxel_data),
                    DEBUG=DEBUG,
                )
                self.taxel_locs[triangle_ini] = [x, y]
                self.taxel_masks[triangle_ini] = mask
                self.imgs[triangle_ini] = img
                if self.show_ed_feed:
                    self.ed_imgs[triangle_ini] = img.copy()

                if self.show_raw_feed:
                    raw_window_name = f"{triangle_ini}_raw"
                    cv2.namedWindow(raw_window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(raw_window_name, self.imgs[triangle_ini])

                if self.show_ed_feed:
                    ed_window_name = f"{triangle_ini}_ed"
                    cv2.namedWindow(ed_window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(ed_window_name, self.ed_imgs[triangle_ini])

                    def on_thresh_slider(val):
                        val /= 100
                        self.esim[-1].Cm = val
                        self.esim[-1].Cp = val

                    cv2.createTrackbar(
                        "Threshold",
                        ed_window_name,
                        int(self.esim[-1].Cm * 100),
                        100,
                        on_thresh_slider,
                    )
                    cv2.setTrackbarMin("Threshold", ed_window_name, 1)
                    
        cv2.waitKey(50)
        if DEBUG:
            logging.info("All panels initialized.")
        # return esim, taxel_locs, imgs

    def update_skin(self, time):
        """Update all skin patches, generate events, and refresh feed images.

        Args:
            time (int): Current simulation timestamp in nanoseconds.

        Returns:
            list[np.ndarray]: Event arrays in the same order as
            ``TRIANGLE_FILES``. Each array contains rows
            ``(taxel_id, t_ns, polarity)``.

        Side Effects:
            - Updates ``latest_raw_frames`` and ``latest_ed_frames`` caches.
            - Refreshes OpenCV windows when corresponding ``show_*_feed`` flags
              are enabled.
        """

        all_events = []
        for triangle_ini, esim_single in zip(TRIANGLE_FILES, self.esim):
            # TODO make sure we hand over the right data here
            if "right_hand" in triangle_ini:
                # TODO double check the order of the taxels!
                taxel_data = []
                for key in KEY_MAPPING["r_hand"]:
                    taxel_data.extend(self.grouped_sensors[key])
            elif "left_hand" in triangle_ini:
                taxel_data = []
                for key in KEY_MAPPING["l_hand"]:
                    taxel_data.extend(self.grouped_sensors[key])
            else:
                taxel_data = self.grouped_sensors[KEY_MAPPING[triangle_ini]]

            events = esim_single.skinCallback(taxel_data, time)
            all_events.append(events)
            if self.DEBUG:
                if len(events):
                    logging.info(
                        f"{len(events)} events detected at {triangle_ini}.")

            if self.show_raw_feed:
                raw_img = make_skin_raw_frame(
                    img=self.imgs[triangle_ini].copy(),
                    taxel_data=taxel_data,
                    locations=self.taxel_locs[triangle_ini],
                    taxel_mask=self.taxel_masks[triangle_ini],
                )
                self.latest_raw_frames[triangle_ini] = raw_img
                cv2.imshow(f"{triangle_ini}_raw", raw_img)

            if self.show_ed_feed:
                ed_img = make_skin_event_frame(
                    img=self.ed_imgs[triangle_ini],
                    events=events,
                    locations=self.taxel_locs[triangle_ini],
                    taxel_mask=self.taxel_masks[triangle_ini],
                )
                self.latest_ed_frames[triangle_ini] = ed_img.copy()
                cv2.imshow(f"{triangle_ini}_ed", ed_img)

            if self.show_raw_feed or self.show_ed_feed:
                cv2.waitKey(1)

        return all_events

    def save_feed_images(
        self,
        output_dir: str,
        prefix: str = "",
        time_ns: int | None = None,
        include_raw: bool = True,
        include_ed: bool = True,
    ) -> list[str]:
        """Save one image per skin patch for raw and/or event-driven feeds.

        This method writes the latest rendered frames stored by ``update_skin``
        to disk. Use it each simulation step or with a lower save cadence
        (for example every N-th step) to control I/O volume.

        Args:
            output_dir: Directory where image files are written. It is created
                automatically if missing.
            prefix: Optional filename prefix (for example ``"frame_"``).
            time_ns: Optional timestamp appended to each filename as
                ``_<time_ns>``.
            include_raw: If True, save raw per-taxel color-map frames.
            include_ed: If True, save event-driven frames.

        Returns:
            list[str]: Paths of all files written during this call.

        Example:
            skin_object.update_skin(int(data.time * 1e9))
            skin_object.save_feed_images(
                output_dir="neuromorphic_body_schema/skin_frames",
                prefix="frame_",
                time_ns=int(data.time * 1e9),
                include_raw=True,
                include_ed=True,
            )
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        time_label = f"_{int(time_ns)}" if time_ns is not None else ""
        saved_files = []

        if include_raw:
            for patch_name, frame in self.latest_raw_frames.items():
                filename = f"{prefix}{patch_name}_raw{time_label}.png"
                file_path = out_path / filename
                cv2.imwrite(str(file_path), frame)
                saved_files.append(str(file_path))

        if include_ed:
            for patch_name, frame in self.latest_ed_frames.items():
                filename = f"{prefix}{patch_name}_ed{time_label}.png"
                file_path = out_path / filename
                cv2.imwrite(str(file_path), frame)
                saved_files.append(str(file_path))

        return saved_files
