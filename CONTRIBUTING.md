# Contributing to ISL Sign-to-Text

Thank you for your interest in contributing! This project is part of a Final Year Project (FYP) on Indian Sign Language recognition. Contributions, bug reports, and suggestions are all welcome.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sign_to_text_module.git
   cd sign_to_text_module
   ```
3. **Run the Setup Script** (This creates the virtual environment, installs all dependencies, and sets up git hooks):
   - **Windows:** `powershell -ExecutionPolicy Bypass -File scripts/setup.ps1`
   - **Linux / macOS:** `bash scripts/setup.sh`
4. **Activate the environment**:
   - **Windows:** `venv\Scripts\activate`
   - **Linux / macOS:** `source venv/bin/activate`
5. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Quality & Pre-commit Hooks

This project enforces strict code quality via `pre-commit`. When you run `git commit`, the following tools automatically check your code:
- `black` (Auto-formatting, line length 120)
- `isort` (Import sorting)
- `ruff` (Linting)
- File hygiene checks (trailing whitespace, mixed line endings)
- `detect-secrets` (Security scanning)

If a hook fails or modifies files, review the changes, `git add` them again, and re-run your commit.

You can also run checks manually:
```bash
pre-commit run --all-files
```

## Testing

All new features must include tests. We use `pytest`.

```bash
# Run unit tests
pytest tests/unit/

# Run API endpoint tests
pytest tests/api/

# Run specific file
pytest tests/unit/test_config.py
```

## Pull Request Process

1. Ensure the CI pipeline passes on your branch (GitHub Actions runs automatically on PRs).
2. Ensure you have added or updated tests in `tests/`.
3. Add a brief description of what you changed and why in the PR body.
4. Reference any related issues in the PR description.
5. If changing API endpoints, ensure `api/schemas.py` and `tests/api/test_endpoints.py` are updated.

## Reporting Issues

Open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce
- Your OS, Python version, and package versions (`pip freeze`)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
