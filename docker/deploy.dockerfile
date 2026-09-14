# syntax=docker/dockerfile:1.4

FROM --platform=linux/arm64 arm64v8/ros:noetic-perception-focal

LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV ROS_DISTRO=noetic
ENV ROS_PYTHON_VERSION=3

# Keep proxy config (override by --build-arg if needed)
ARG HTTP_PROXY=http://127.0.0.1:8889
ARG HTTPS_PROXY=http://127.0.0.1:8889
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV no_proxy=localhost,127.0.0.1,::1
ENV NO_PROXY=localhost,127.0.0.1,::1

SHELL ["/bin/bash", "-c"]

ARG APT_MAX_RETRIES=10
ARG APT_RETRY_SLEEP=12
ENV APT_MAX_RETRIES=${APT_MAX_RETRIES}
ENV APT_RETRY_SLEEP=${APT_RETRY_SLEEP}

RUN printf 'Acquire::Retries "8";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/99-network-retry
RUN cat >/usr/local/bin/apt-install-retry <<'EOF' && chmod +x /usr/local/bin/apt-install-retry
#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: apt-install-retry <pkg1> [pkg2 ...]" >&2
  exit 2
fi
max_retries="${APT_MAX_RETRIES:-10}"
sleep_base="${APT_RETRY_SLEEP:-12}"
for i in $(seq 1 "$max_retries"); do
  echo "[apt-install-retry] attempt ${i}/${max_retries}: $*"
  if apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 update && \
     DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 \
     install -y --no-install-recommends --fix-missing "$@"; then
    rm -rf /var/lib/apt/lists/*
    exit 0
  fi
  rc=$?
  echo "[apt-install-retry] failed with exit=${rc}, retrying..." >&2
  rm -rf /var/lib/apt/lists/*
  sleep "$(( sleep_base * i ))"
done
echo "[apt-install-retry] exhausted retries for: $*" >&2
exit 1
EOF

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone

RUN apt-install-retry \
    tzdata ca-certificates curl wget gnupg2 lsb-release software-properties-common \
    build-essential git git-lfs vim tmux unzip zip sudo \
    python3-dev python3-pip python3-setuptools python3-wheel python3-distutils

RUN python3 -m pip install --no-cache-dir --upgrade \
    "pip<25" \
    "setuptools<75" \
    wheel

# TensorRT + torch2trt (for ROS1 deployment side)
# Keep the requested pip command first; fallback for platforms where wheel is unavailable.
RUN set -eux; \
    if python3 -m pip install --no-cache-dir -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com; then \
      echo "Installed nvidia-tensorrt from NGC index."; \
    else \
      echo "nvidia-tensorrt wheel not available on this platform, trying apt TensorRT bindings..." >&2; \
      if DEBIAN_FRONTEND=noninteractive apt-get update && \
         DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
           python3-libnvinfer libnvinfer-dev libnvinfer-plugin-dev; then \
        rm -rf /var/lib/apt/lists/*; \
        echo "Installed TensorRT from apt packages."; \
      else \
        rm -rf /var/lib/apt/lists/*; \
        echo "TensorRT Python binding not installed during build; install it on target device runtime if needed." >&2; \
      fi; \
    fi

RUN git clone --depth 1 https://github.com/NVIDIA-AI-IOT/torch2trt /tmp/torch2trt && \
    cd /tmp/torch2trt && \
    if python3 setup.py install; then \
      echo "torch2trt installed."; \
    else \
      echo "torch2trt install skipped (TensorRT Python module unavailable during build)." >&2; \
    fi && \
    rm -rf /tmp/torch2trt

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

WORKDIR /workspace

# If proxy is not needed in your target machine, unset in runtime:
#   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

CMD ["/bin/bash"]
