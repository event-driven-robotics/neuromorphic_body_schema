Goal

Every taxel placement decision used by model generation must come from taxel_alignment_optimization_report.json.
The insertion script must not use hard-coded per-part transforms, body mappings, or fallback defaults that can diverge.
Partial or stale reports must fail fast in strict mode.
Current divergence points to eliminate

Hard-coded body/group maps in insertion script: include_skin_to_mujoco_model.py:67 and include_skin_to_mujoco_model.py:83.
Large body-name switch mapping and file routing logic: include_skin_to_mujoco_model.py:592.
Legacy transform fallback block: include_skin_to_mujoco_model.py:870.
Report loader still allows defaults/fallback behavior: include_skin_to_mujoco_model.py:154.
Optimizer run can produce partial report when run on selected parts with fresh-start, leaving no canonical full manifest.
Target architecture

One canonical manifest/report with one entry per taxel-bearing part, always complete.
Insertion code consumes only that manifest plus position data files.
No duplication of part metadata in insertion script.
Explicit schema versioning and strict validation.
Optional compatibility mode during migration, then strict-only default.
Phase 1: Define and lock the report schema

Add report schema version field at top-level, for example schema_version.
Make required entry fields explicit: part, model_body_name, sensor_group, position_file, rebase, include_to_model, manual_steps, optimized delta fields.
Add placement_mode field for non-optimized but deterministic parts.
Add optional fields for fixed-pattern fingers if you want fingers in source-of-truth too.
Add strict JSON validation function in optimizer before writing and in insertion before reading.
Add schema doc section to SKIN_TAXEL_ALIGNMENT_REPRODUCIBILITY.md.
Phase 2: Ensure report completeness on every run

Change optimizer write behavior so report always contains all configured parts from optimize_taxel_alignment.py:1143.
For parts not optimized in a run, preserve previous optimized entry if present, else emit deterministic baseline entry from PartConfig.
Add explicit command flag for partial report if ever needed, and make it non-default.
Add a completeness validator: all expected parts present, no unknown parts, no duplicate parts.
Store generation metadata in report: generated_at, command, git revision if available.
Phase 3: Remove duplicate metadata from insertion

Delete PART_TO_MODEL_BODY and PART_TO_GROUP usage paths in insertion, replacing with report-only values.
Replace body-name switch routing logic with a report-driven lookup table built from report entries.
Keep only minimal model traversal logic in insertion; all policy decisions come from report.
Keep one well-defined fallback only during migration, behind compatibility flag.
Phase 4: Remove transform fallbacks

Delete the hard-coded per-part legacy transform block at include_skin_to_mujoco_model.py:870.
Always apply report manual_steps and optimized delta when present.
If a required field is missing, fail with a clear error listing missing keys and part names.
Add strict mode default true, compatibility mode optional.
Phase 5: Decide finger strategy explicitly

Option A preferred for strictness: add explicit finger entries to report with placement_mode fixed_pattern and fixed taxel list.
Option B transitional: keep finger constants in insertion but register them in report as externally managed and enforce explicit acknowledgment.
Whichever option you choose, remove silent behavior from insertion and make intent declarative in report.
Phase 6: Add strict validation gates

Pre-insertion gate: validate schema version, required fields, complete part set, and referenced position files exist.
Consistency gate: no contradictory include_to_model states for required bodies.
Structural gate: report parts map to existing MuJoCo body names.
Runtime gate: insertion aborts on first invalid entry with actionable message.
Phase 7: Testing plan

Unit test report schema validator with good and bad fixtures.
Unit test completeness validator for missing/extra/duplicate parts.
Unit test insertion planner from report metadata only.
Integration test: run optimizer then insertion and ensure expected taxel site counts per part.
Regression test for palms: both use report-declared side selection and resulting transforms.
Backward compatibility test for old reports only if compatibility mode retained.
Phase 8: Migration rollout

Milestone A: implement validators and add compatibility mode, keep fallback paths but log warnings.
Milestone B: switch default to strict mode in insertion, fail on fallback path usage.
Milestone C: remove compatibility and fallback code.
Milestone D: update docs and CI to enforce strict report-first pipeline.
Concrete code change checklist

Update optimizer report writer and upgrader in optimize_taxel_alignment.py:2190.
Add report schema/completeness validators in optimizer and insertion.
Refactor insertion mapping flow in include_skin_to_mujoco_model.py:505.
Remove legacy transform fallback in include_skin_to_mujoco_model.py:870.
Extend reproducibility doc at SKIN_TAXEL_ALIGNMENT_REPRODUCIBILITY.md.
Definition of done

Insertion script has zero hard-coded per-part transform rules.
Insertion script has zero hard-coded part-to-body and part-to-group defaults for taxel-bearing parts.
Strict mode passes with a full report and fails with partial report.
CI includes at least one end-to-end test optimizer to insertion.
Re-running with selected parts still leaves a complete canonical report.
If you want, I can implement this in incremental PR-style steps starting with Phase 1 and Phase 2 (schema plus completeness guarantees), then Phase 3 and Phase 4 (insertion refactor and fallback removal).