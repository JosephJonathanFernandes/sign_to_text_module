# UV Guide for ISL Sign-to-Text System

This document provides a comprehensive guide on using [`uv`](https://github.com/astral-sh/uv)—an extremely fast Python package and project manager written in Rust—with the ISL Sign-to-Text module.

Using `uv` accelerates dependency installation, environment creation, and execution by 10×–100× compared to traditional `pip` and `venv` workflows, without requiring any modifications to the core application source code.

---

## 🛠️ 1. How to Run the Project Using UV Commands

Because this repository contains standard PEP 621 metadata in [`pyproject.toml`](file:///c:/DEV/Project/Final_Year/Johnny's%20Model/sign_to_text_module/pyproject.toml), `uv` works natively with zero additional configuration.

### A. Environment Creation & Syncing
Instead of manually creating a virtual environment and running `pip install`, use `uv sync` to automatically create `.venv` and install all required dependencies:

```bash
# Sync core project dependencies into .venv
uv sync

# Sync both core dependencies and development tools (pytest, ruff, black, etc.)
uv sync --extra dev
```

### B. Running the API Server
`uv run` executes commands inside the project's virtual environment without needing to manually source or activate it (`activate.ps1` / `activate`):

```bash
# Start the FastAPI server
uv run python run_api.py

# Start the server in DEBUG mode (Windows PowerShell)
$env:PYTHONUTF8="1"; $env:DEBUG="true"; uv run python run_api.py
```
> [!TIP]
> On Windows PowerShell, setting `$env:PYTHONUTF8="1"` ensures rich ASCII/Unicode box-drawing characters (such as configuration tables) print cleanly without encoding errors.


### C. Running Tests
Run your pytest test suites instantly:

```bash
# Run unit tests
uv run pytest tests/unit/

# Run API tests
uv run pytest tests/api/

# Run full test suite with coverage
uv run pytest tests/ -v --cov=src --cov=api
```

### D. Running Utility Scripts & Benchmarks
Execute any project script directly:

```bash
# Run GNN synthetic smoke test
uv run python scripts/smoke_gnn_test.py

# Run GNN benchmark harness
uv run python scripts/benchmark_gnn.py --mode synthetic --use_gnn 1 --iters 100

# Launch live webcam inference test
uv run python webcam.py
```

---

## ⚡ 2. Helpful UV Commands & Use Cases

### A. Managing Python Versions
`uv` can automatically download and manage Python versions for you without requiring manual system installs:

```bash
# Install Python 3.10 (or 3.11, 3.12)
uv python install 3.10

# Pin the local directory to a specific Python version (.python-version file)
uv python pin 3.10

# Create a virtual environment using a specific Python version
uv venv --python 3.10
```

### B. Managing Dependencies (`uv add` / `uv remove` / `uv lock`)
Instead of manually editing `pyproject.toml` or `requirements.txt`:

```bash
# Add a new dependency to pyproject.toml and update uv.lock automatically
uv add scipy

# Add a development dependency
uv add --dev pytest-mock

# Remove a dependency
uv remove scipy

# Update and lock dependencies to deterministic exact versions
uv lock
```

### C. Dependency Tree Visualization
Inspect your installed dependencies and transitive sub-dependencies clearly:

```bash
uv tree
```

### D. On-the-Fly Tool Execution (`uvx` / `uv tool`)
Run standalone code quality tools without polluting your project environment:

```bash
# Run Ruff linter on-the-fly
uvx ruff check .

# Run Black code formatter on-the-fly
uvx black --check .

# Run MyPy type checker
uvx mypy src/
```

### E. Pip-Compatible Commands (`uv pip`)
If you prefer traditional `pip` flags, `uv` provides a dropped-in lightning fast replacement under `uv pip`:

```bash
# Install requirements.txt using UV speed
uv pip install -r requirements.txt

# Install dev requirements
uv pip install -r requirements-dev.txt

# List installed packages in current environment
uv pip list

# Compile requirements.txt from pyproject.toml
uv pip compile pyproject.toml -o requirements.lock
```

---

## 🚀 Summary of Workflow Comparison

| Workflow Step | Traditional `pip` / `venv` | `uv` Workflow |
| :--- | :--- | :--- |
| **Create Env** | `python -m venv venv` | `uv venv` |
| **Install Deps** | `pip install -r requirements.txt` | `uv sync` |
| **Run Scripts** | `venv\Scripts\activate` + `python script.py` | `uv run python script.py` |
| **Run Tests** | `pytest` (after activation) | `uv run pytest` |
| **Lock Deps** | `pip freeze > requirements.txt` | `uv lock` |
