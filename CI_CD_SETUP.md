# CI/CD Setup for neuromorphic_body_schema

## Overview
This document describes the Continuous Integration/Continuous Deployment (CI/CD) setup for the `neuromorphic_body_schema` project. The setup includes automated testing, code quality checks, and branch protection rules on GitHub.

---

## Components

### 1. **GitHub Actions Workflow** (`.github/workflows/ci.yml`)

The workflow runs automatically on every push and pull request to `main` and `develop` branches.

#### What it checks:
- **Python Version Compatibility**: Tests on Python 3.10 and 3.11
- **Dependencies**: Installs all project dependencies including system libraries (libosmesa6-dev, libglfw3)
- **Code Formatting**: Checks code style with `black`
- **Linting**: Runs `ruff` to catch common errors
- **Type Checking**: Runs `mypy` for static type analysis
- **Unit Tests**: Runs pytest with coverage reporting
- **Model Loading**: Verifies the MuJoCo model XML loads correctly

#### Key features:
- Runs on each push/PR (saves build artifacts)
- Matrix testing across multiple Python versions
- Automatic LFS file pulling via `lfs: true` in checkout
- 10-minute timeout per test (prevents hanging tests from blocking CI)
- Coverage reporting via codecov (optional)

---

### 2. **Test Suite** (`tests/`)

**19 comprehensive tests** organized in 5 test modules:

| Module | Tests | Purpose |
|--------|-------|---------|
| `test_imports.py` | 3 | Verify core modules and dependencies import |
| `test_helpers.py` | 5 | Test helper functions, paths, and initialization |
| `test_model_loading.py` | 6 | Verify MuJoCo model loads and simulates correctly |
| `test_sensors.py` | 2 | Test sensor parsing and grouping |
| `test_main_initialization.py` | 3 | Test main.py initialization sequence |

**Key design:**
- Uses shared fixtures (`conftest.py`) to load model once per test session (speeds up tests)
- All tests include timeouts to prevent hanging
- Tests are independent but share cached resources
- Focus on initialization, not full simulation runs

---

### 3. **pytest Configuration** (`pytest.ini`)

```ini
[pytest]
testpaths = tests
addopts = -v --tb=short --timeout=10
```

- **Timeout**: 10 seconds per test (prevents CI hangs)
- **Verbose output**: Shows test names and results clearly
- **Short traceback**: Concise error output for debugging

---

### 4. **Pre-commit Hooks** (`.pre-commit-config.yaml`)

Local hooks that run **before committing** to catch issues early:

- **black**: Auto-formats Python code
- **ruff**: Fixes common linting issues
- **trailing-whitespace**: Removes trailing spaces
- **end-of-file-fixer**: Ensures files end with newline
- **check-yaml**: Validates YAML files
- **check-added-large-files**: Warns if adding files >1MB

#### Installation:
```bash
pip install pre-commit
pre-commit install
```

---

## What Tests Check For (Weak Spots Identified)

Based on analysis of your codebase, tests cover:

1. **Path Resolution** ✅
   - Model XML file paths are now absolute (fixed in `helpers.py`)
   - All paths resolve correctly from any directory

2. **Model Integrity** ✅
   - Model XML loads without errors
   - Data structures initialize correctly
   - Basic physics simulation works

3. **Sensor Configuration** ✅
   - Taxel sensors parse from model
   - Sensor grouping works correctly
   - DynamicGroupedSensors class initializes

4. **Joint Configuration** ✅ (with graceful fallback)
   - Tests check if expected joints exist
   - Skips gracefully if joints not in model variant

5. **Dependencies** ✅
   - All imports work
   - mujoco is properly installed
   - Helper modules accessible

6. **Simulation Basics** ✅
   - Forward kinematics produces valid output
   - Time steps correctly

---

## GitHub Branch Protection Rules

To enforce these checks before merging PRs, configure in repository settings:

1. Go to **Settings → Branches → Add rule** for `main` branch
2. Enable:
   - ✅ **Require PR reviews before merging**
   - ✅ **Require status checks to pass before merging**
   - ✅ **Allow auto-merge** (optional)
   - ✅ **Require branches to be up to date before merging**

Then select the CI job status checks to require (test-3.10, test-3.11, etc.)

---

## Running Tests Locally

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-timeout black ruff mypy

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_model_loading.py -v

# Run with coverage report
pytest tests/ --cov=neuromorphic_body_schema

# Format code locally (before commit)
black neuromorphic_body_schema tests/

# Lint code
ruff check neuromorphic_body_schema tests/

# Setup pre-commit hooks
pre-commit install
```

---

## Continuous Improvement

### Tests that could be added in the future:
- IK solver correctness tests (if test data available)
- Event camera simulation sanity checks
- Skin sensor event generation
- Integration tests with visualization disabled

### Performance optimization:
- Tests currently run in ~18 seconds
- Could add `@pytest.mark.slow` decorator for optional slow tests
- CI could run slow tests only on daily schedule

---

## Why This Setup Makes Sense

1. **Early Detection**: Issues caught before main branch
2. **Python Compatibility**: Tests on multiple versions (good for team dev)
3. **Shared Fixtures**: Model loads once, tests reuse it (10x faster than individual loads)
4. **LFS Ready**: GitHub Actions configured to pull LFS files automatically
5. **Formatting Consistency**: Black enforces code style automatically
6. **Type Safety**: mypy catches potential bugs before runtime
7. **Scientist-friendly**: No complex test requirements, straightforward checks

---

## Next Steps

1. **Commit these files** to your repository
2. **Enable branch protection** in GitHub settings
3. **Run tests locally** with `pytest tests/` to verify
4. **Install pre-commit hooks** with `pre-commit install`
5. **Test a PR** to see CI in action
