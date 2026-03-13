# Skin Taxel Alignment: Reproducibility Notes

This document records the debugging and alignment steps used to make skin visualization, taxel placement, and MuJoCo sensor insertion consistent.

## Scope

The work aligns three data sources:

1. `skinGUI/*.ini` geometry patch definitions used by visualization
2. `include_taxels/positions/*.txt` position and `taxel2Repr` metadata used by model generation
3. `models/icub_v2_full_body_contact_sensors.xml` generated MuJoCo sensor model

## Semantics Used

`taxel2Repr` values:

- `>= 0`: tactile channel
- `-1`: unused/non-available channel
- `-2`: thermal/non-tactile channel

Calibration rows (6 values per channel):

- a channel is treated as tactile when either position or normal is non-zero
- all six values equal to `0.0` indicate a non-physical/non-tactile row

## Key Fixes Applied

### 1) taxel2Repr parsing and usage in model insertion

File: `neuromorphic_body_schema/include_taxels/include_skin_to_mujoco_model.py`

- added parsing of `taxel2Repr` blocks from position files
- validated tactile channels using `taxel2Repr >= 0`
- kept calibration non-zero filtering as a robust check

### 2) Visualization mask generation

File: `neuromorphic_body_schema/helpers/ed_skin.py`

- added `_build_taxel_mask(...)` to build per-point tactile masks
- made mapping config-aware:
  - `triangle_10pad` -> 10 offsets in 12-slot block
  - `fingertip3R/L` -> full 12-slot block
  - `palmR/L` -> 4 consecutive 12-slot blocks
- colored non-tactile channels with a dedicated dark-gray color

### 3) taxel2Repr-short fallback to calibration

When a geometry slot index exceeds `taxel2Repr` length:

- fallback now checks calibration rows for non-zero 6D content
- this is required for parts where `taxel2Repr` coverage is shorter than calibration coverage

### 4) Deterministic arm patch-id remap

File: `neuromorphic_body_schema/helpers/helpers.py`

- introduced `ARM_PATCH_ID_REMAP`:
  - `right_arm`: `7 -> 2`
  - `left_arm`: `7 -> 2`

Rationale:

- arm `skinGUI` draws patch IDs including `7`, while tactile `taxel2Repr` block `2` is the corresponding tactile module used by model/data
- remap avoids non-deterministic count reconciliation and yields stable tactile location assignment

### 5) Event rendering index correctness

File: `neuromorphic_body_schema/helpers/ed_skin.py`

- fixed event plotting to use tactile-channel index progression when mask is present
- prevents event/channel drift when non-tactile points are skipped in the drawing

## Validation Procedure

Use this command to compare per-patch mask counts against the generated MuJoCo model:

```bash
cd /home/smullercleve/code/neuromorphic_body_schema
/home/smullercleve/.virtualenvs/mujoco/bin/python - <<'PY'
import re
from collections import defaultdict
import mujoco
from neuromorphic_body_schema.helpers.helpers import TRIANGLE_FILES, TRIANGLE_INI_PATH, MODEL_PATH
from neuromorphic_body_schema.helpers.ed_skin import visualize_skin_patches

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
names = model.names.decode('utf-8').split('\x00')
g = defaultdict(int)
for n in names:
    if 'taxel' in n:
        g[re.sub(r'_\d+$', '', n)] += 1
for side in ['r', 'l']:
    g[f'{side}_hand_total'] = g[f'{side}_palm_taxel'] + sum(
        g[f'{side}_hand_{f}_taxel'] for f in ['thumb', 'index', 'middle', 'ring', 'little']
    )

model_key = {
    'right_arm':        'r_upper_arm_taxel',
    'left_arm':         'l_upper_arm_taxel',
    'torso':            'torso_taxel',
    'right_forearm_V2': 'r_forearm_taxel',
    'left_forearm_V2':  'l_forearm_taxel',
    'right_leg_upper':  'r_upper_leg_taxel',
    'left_leg_upper':   'l_upper_leg_taxel',
    'right_leg_lower':  'r_lower_leg_taxel',
    'left_leg_lower':   'l_lower_leg_taxel',
    'right_hand_V2_2':  'r_hand_total',
    'left_hand_V2_2':   'l_hand_total',
}

print(f"{'patch':<22} {'drawn':>6} {'tactile_mask':>13} {'model':>6} {'match':>6}")
for tri in TRIANGLE_FILES:
    expected = g.get(model_key.get(tri, ''), 0)
    _, x, _, mask = visualize_skin_patches(
        TRIANGLE_INI_PATH,
        tri,
        expected_tactile_count=expected,
        DEBUG=False,
    )
    tactile = sum(mask) if mask else len(x)
    print(f"{tri:<22} {len(x):>6} {tactile:>13} {expected:>6} {'OK' if tactile == expected else 'DIFF':>6}")
PY
```

## Test Command

If local `pytest.ini` includes options from optional plugins, run with an addopts override:

```bash
cd /home/smullercleve/code/neuromorphic_body_schema
/home/smullercleve/.virtualenvs/mujoco/bin/python -m pytest -q -o addopts=''
```

Expected result at time of writing:

- `19 passed`

## Notes for Future Changes

- Keep `include_taxels` generation and `helpers/ed_skin.py` masking semantics aligned.
- If arm geometry files are updated upstream, re-check whether `ARM_PATCH_ID_REMAP` is still needed.
- Prefer deterministic mapping over count-only reconciliation when a stable geometry-to-positions relationship is known.
