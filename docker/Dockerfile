# Optional native mjlab runtime. Build from the repository root:
# docker build -f docker/Dockerfile -t supermimic:mjlab .
FROM ghcr.io/astral-sh/uv:0.11.21 AS uv
FROM python:3.13-slim-bookworm
COPY --from=uv /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux ffmpeg util-linux build-essential libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Register the NVIDIA EGL vendor (the driver library is injected by the NVIDIA container runtime);
# without it GLVND only knows Mesa and MuJoCo's offscreen renderer falls back to CPU rasterization.
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
WORKDIR /workspace/supermimic
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY supermimic ./supermimic
RUN uv sync --locked --no-dev
COPY tools ./tools
COPY run_train.sh ./
CMD ["bash"]
