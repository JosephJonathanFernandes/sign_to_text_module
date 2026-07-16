FROM python:3.11-slim-bookworm

# Set up a non-root user (Hugging Face requires UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy uv binary for fast package installation
COPY --chown=user --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files for caching
COPY --chown=user pyproject.toml uv.lock ./

# Install dependencies (frozen to lock file)
RUN uv sync --frozen --no-cache

# Copy the rest of the application files
COPY --chown=user . .

# Set Hugging Face's default port and run configurations
ENV PORT=7860
ENV PYTHONUTF8=1
ENV ENV=production

EXPOSE 7860

# Start the FastAPI server
CMD ["uv", "run", "python", "run_api.py"]
