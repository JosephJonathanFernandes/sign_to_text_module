# Contributing to ISL Sign-to-Text

Thank you for your interest in contributing! This project is part of a Final Year Project (FYP) on Indian Sign Language recognition. Contributions, bug reports, and suggestions are all welcome.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sign_to_text.git
   cd sign_to_text
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate

   pip install -r requirements.txt
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow **PEP 8** conventions
- Use **type hints** on all new functions
- Add **docstrings** (Google style) to all new public functions and classes
- Keep functions focused — aim for < 50 lines per function

## Areas to Contribute

| Area | Description |
|---|---|
| Sign classes | Add new ISL signs by recording, preprocessing, and training |
| Augmentation | New landmark-level augmentations in `src/preprocessing/augmentations.py` |
| Model | New architectural experiments in `src/training/model.py` |
| Inference | Latency improvements to `src/core/webcam.py` |
| Documentation | Improve docs, fix typos, add examples |
| Tests | Add unit tests to `tests/` |
| ONNX | Improve quantization or add TensorRT support |

## Pull Request Process

1. Ensure `python main.py -h` still works after your changes
2. Run the import check: `python -m flake8 src --select=E999,F821`
3. Add a brief description of what you changed and why
4. Reference any related issues in the PR description

## Reporting Issues

Open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce
- Your OS, Python version, and package versions (`pip freeze`)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
