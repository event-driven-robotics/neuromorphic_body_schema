# Runtime image for the neuromorphic_body_schema iCub MuJoCo simulation.
#
# Only system libraries and Python dependencies live in the image. The
# repository itself (code, models, meshes) is bind-mounted at /workspace by
# docker-compose.yml, so edits on the host take effect without rebuilding.
FROM ubuntu:24.04

ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 (Ubuntu 24.04 default) + OpenGL/GLFW for MuJoCo + X11/Qt libs for OpenCV windows.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git ca-certificates mesa-utils \
        libgl1 libegl1 libglvnd0 libglx-mesa0 libosmesa6 libglfw3 \
        libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 libdbus-1-3 fonts-dejavu-core \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0 libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# nvidia-container-toolkit injects libEGL_nvidia.so.0 but not its glvnd vendor
# file, so headless EGL (MUJOCO_GL=egl) would fall back to Mesa llvmpipe.
# On hosts without the NVIDIA driver glvnd skips this entry and uses Mesa.
RUN printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : { "library_path" : "libEGL_nvidia.so.0" }\n}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY requirements.txt /tmp/requirements.txt
# opencv-python's bundled Qt ships no fonts; link DejaVu in so imshow doesn't warn on every frame.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
        pytest pytest-timeout pytest-cov \
    && rm /tmp/requirements.txt \
    && ln -s /usr/share/fonts/truetype/dejavu \
        "$(/opt/venv/bin/python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')/qt/fonts"

# Run as a non-root user matching the host UID/GID so files written into the
# bind mount stay owned by the host user. Ubuntu 24.04 ships a default
# `ubuntu` user on UID 1000, which is removed first.
RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi \
    && if ! getent group "${USER_GID}" >/dev/null; then groupadd -g "${USER_GID}" icub; fi \
    && useradd -m -s /bin/bash -u "${USER_UID}" -g "${USER_GID}" icub

# No .pyc files or .pytest_cache/ are written into the mounted repository.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTEST_ADDOPTS="-p no:cacheprovider"

WORKDIR /workspace
USER icub
CMD ["bash"]
