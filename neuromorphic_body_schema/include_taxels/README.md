# include_taxels

This folder contains the tactile skin (taxel) alignment and MuJoCo model integration tooling.

It currently covers three main activities:

1. Optimize taxel patch alignment to robot body meshes.
2. Visualize optimization quality and geometric fit.
3. Inject taxel sites and touch sensors into a MuJoCo model.

It also contains targeted alignment utilities (for example geom-to-geom cover/palm tuning) and some legacy/experimental scripts.

## Quick Start

From repository root:

```bash
python neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --parts l_foot
python neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py --parts l_foot
python neuromorphic_body_schema/include_taxels/include_skin_to_mujoco_model.py --strict-report
python neuromorphic_body_schema/main.py
```

Quick palm-cover interactive tuning (right hand):

```bash
python neuromorphic_body_schema/include_taxels/align_geom_to_anchor.py \
  --side right \
  --anchor-geom r_hand_vis_0 \
  --movable-geom r_hand_top_cover_vis_0 \
  --pair-label right_palm_cover \
  --interactive \
  --interactive-port 8058 \
  --save-model neuromorphic_body_schema/models/icub_v2_full_body_improved.xml
```

## Folder Contents

- `align_geom_to_anchor.py`
  - Generic interactive and CLI alignment tool for one movable geom relative to one anchor geom.
  - Supports default hand presets (`--side left|right|both`) and explicit geom names (`--anchor-geom`, `--movable-geom`).
  - Can print, dry-run, edit XML, and generate Plotly HTML previews.

- `align_hand_cover_to_palm.py`
  - Backward-compatible wrapper that forwards to `align_geom_to_anchor.py`.
  - Keep using old commands if needed.

- `optimize_taxel_alignment.py`
  - Main optimization pipeline.
  - Uses MuJoCo compiled mesh geometry as optimization target (same representation as simulator and visualizer).
  - Core distance metric is mean absolute nearest-surface distance across all taxels.
  - Produces and updates:
    - `taxel_alignment_optimization_report.json`
    - `taxel_alignment_optimization_report.txt`

- `visualize_taxel_alignment.py`
  - Generates per-part before/after HTML reports from optimization results.
  - Writes output into `visualizations/` and creates `visualizations/index.html`.
  - Also supports footprint preview mode.

- `include_skin_to_mujoco_model.py`
  - Injects skin taxel sites and touch sensors into the MuJoCo XML model.
  - Uses **material-based styling by default** (`--style-mode material`).
  - Legacy per-geom RGBA styling remains available via `--style-mode rgba`.
  - Reads optimization report deltas when available.
  - Writes output model:
    - `../models/icub_v2_full_body_improved_contact_sensors.xml`
  - Writes taxel-count report:
    - `report_including_taxels.txt`

- `interactive_taxel_alignment_dash.py`
  - Older Dash-based per-part taxel patch alignment UI.
  - Useful as an auxiliary manual tool.

- `check_skin_topology.py`
  - Topology consistency checker between 3D taxel data and 2D GUI layout.
  - Experimental utility.

- `positions/`
  - Input taxel calibration and mapping text files (per part).

- `visualizations/`
  - Generated HTML visual output.

- `optimization_report_as_single_source_of_truth.md`
  - Design notes for report-driven workflow.

## Alignment Consistency (Important)

The current workflow is intentionally consistent across optimization, visualization, and XML insertion:

1. `optimize_taxel_alignment.py` evaluates distances against MuJoCo compiled meshes.
2. `visualize_taxel_alignment.py` renders MuJoCo compiled meshes.
3. `include_skin_to_mujoco_model.py` inserts taxel sites from the same report transform chain.

This avoids the old mismatch class where optimization was computed on raw STL geometry but validation/simulation used compiled MuJoCo geometry.

Report distance interpretation:

1. `initial.mean_distance_m` and `optimized.mean_distance_m` are mean absolute nearest-surface distances over all taxels.
2. Lower is better for this metric.
3. If additional diagnostic fields are present, compare runs primarily using this distance metric.

## Recommended Workflow

### 1) Optimize selected parts or whole body

```bash
python neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py
```

Examples:

```bash
python neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --parts l_foot
python neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --parts r_forearm torso
python neuromorphic_body_schema/include_taxels/optimize_taxel_alignment.py --polish
```

Notes:

- `--polish` runs stricter local refinement around current best.
- `--overwrite-results` forces replacement even if mean distance is worse.
- `--fresh-start` ignores previously saved seeds for this run.

### 2) Visualize alignment quality

```bash
python neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py
```

Examples:

```bash
python neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py --parts r_forearm torso
python neuromorphic_body_schema/include_taxels/visualize_taxel_alignment.py --preview-footprint --parts l_foot
```

Open:

- `neuromorphic_body_schema/include_taxels/visualizations/index.html`

### 3) Inject taxels and sensors into MuJoCo model

```bash
python neuromorphic_body_schema/include_taxels/include_skin_to_mujoco_model.py
```

By default this uses the material pipeline (recommended).

Use legacy mode only when needed:

```bash
python neuromorphic_body_schema/include_taxels/include_skin_to_mujoco_model.py --style-mode rgba
```

Optional strict validation:

```bash
python neuromorphic_body_schema/include_taxels/include_skin_to_mujoco_model.py --strict-report
```

Custom report path:

```bash
python neuromorphic_body_schema/include_taxels/include_skin_to_mujoco_model.py --report-path neuromorphic_body_schema/include_taxels/taxel_alignment_optimization_report.json
```

### 4) Run the model

Load or run the generated contact-sensor model via your normal runtime entrypoint.

## Geom Alignment Utility

Use `align_geom_to_anchor.py` to tune relative mesh/geoms in the XML while keeping one geom fixed.

### Common right-hand palm/cover workflow

```bash
python neuromorphic_body_schema/include_taxels/align_geom_to_anchor.py \
  --side right \
  --anchor-geom r_hand_vis_0 \
  --movable-geom r_hand_top_cover_vis_0 \
  --pair-label right_palm_cover \
  --interactive \
  --interactive-port 8058 \
  --save-model neuromorphic_body_schema/models/icub_v2_full_body_improved.xml
```

### Generic any-part usage

```bash
python neuromorphic_body_schema/include_taxels/align_geom_to_anchor.py \
  --side right \
  --anchor-geom <anchor_geom_name> \
  --movable-geom <movable_geom_name> \
  --pair-label <label> \
  --show --dry-run
```

### Useful options

- `--dx --dy --dz` translation deltas in meters
- `--rx --ry --rz` rotation deltas in degrees
- `--frame anchor|movable|world` interpretation frame (legacy aliases `palm|cover` also accepted)
- `--visualize` and `--visualize-html` for non-interactive Plotly output
- `--start-zero` to initialize sliders at zero instead of current relative pose

## VS Code Tasks

The workspace has task entries for optimization, visualization, and geom-alignment flows.

Tip: use the task variants labeled with palm-cover to keep anchor/movable targets explicit.

## Important Files Produced by This Folder

- `taxel_alignment_optimization_report.json`
- `taxel_alignment_optimization_report.txt`
- `report_including_taxels.txt`
- `visualizations/index.html`
- `../models/icub_v2_full_body_improved_contact_sensors.xml`

## Known Notes and Caveats

- `check_skin_topology.py` is useful but currently more experimental than the core optimization/inclusion pipeline.
- `include_skin_to_mujoco_model_bkp.py` is legacy backup and should not be the default script.
- `visualizations/` is generated content and can be regenerated at any time.

## Troubleshooting

### Dash UI exits with "Address already in use"

Cause:

- Another alignment Dash server is already running on the same port.

Fix:

```bash
lsof -ti:8058 | xargs -r kill
```

Then relaunch the interactive command or task.

### Alignment sliders start at unexpected values

Cause:

- `--start-zero` was used, or you are connected to an old/stale Dash session.

Fix:

1. Launch without `--start-zero` for current XML-relative initialization.
2. Stop stale port process and restart the UI.

### `visualize_taxel_alignment.py` warns about missing optimized result

Cause:

- That part has no entry yet in `taxel_alignment_optimization_report.json`.

Fix:

1. Run optimizer for that part: `--parts <part_name>`.
2. Re-run visualization.

### `include_skin_to_mujoco_model.py --strict-report` fails

Cause:

- Report schema is incomplete, malformed, or missing required parts.

Fix:

1. Re-run optimization (full body recommended at least once).
2. Inspect and repair `taxel_alignment_optimization_report.json` only if needed.
3. Re-run inclusion with `--strict-report`.

### Material vs legacy color mode

Cause:

- Visual output differs depending on style mode.

Fix:

1. Preferred/default mode: run without flag (material mode).
2. Legacy mode (optional): `--style-mode rgba`.
3. If unsure what was used, re-run inclusion explicitly with your intended mode.

### Geom exists in XML but appears "invisible"

Cause:

- Anchor and movable geoms overlap exactly, or mesh scale is mismatched.

Fix:

1. Run `align_geom_to_anchor.py --show --dry-run` to inspect relative pose.
2. Apply a small offset/rotation in interactive mode to confirm separation.
3. Verify mesh `scale` entries in XML asset definitions if extents look wrong.

## Future Folder Rename

If this folder is renamed later, update:

1. Import paths in scripts that reference `neuromorphic_body_schema.include_taxels`.
2. VS Code tasks in `.vscode/tasks.json`.
3. Any docs or CI commands that call script paths directly.
