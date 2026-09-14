#!/bin/bash
# OmniRisk install & build: apt deps, Python venv, Livox-SDK2, ros_ws, Controller, Simulator.
#
# Prerequisites: Ubuntu 20.04 + ROS Noetic + CUDA (x86), or JetPack 5.1.x (Jetson).
#
# Usage (run as a normal user; sudo is called where needed):
#   bash scripts/build.sh               # first time: install everything and build
#   bash scripts/build.sh --build-only  # rebuild catkin workspaces only
#   bash scripts/build.sh --no-sim      # skip Controller/Simulator (e.g. onboard)

set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="$ROOT_DIR/ros_ws"
VENV="$ROOT_DIR/omnirisk_env"
PY="$VENV/bin/python"
DEPS_DIR="$HOME/.cache/omnirisk"
NPROC=$(nproc)

JETSON_TORCH_WHL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"

BUILD_ONLY=0
NO_SIM=0
for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=1 ;;
        --no-sim)     NO_SIM=1 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if [[ $EUID -eq 0 ]]; then
    echo "Do not run as root; sudo is invoked where needed."
    exit 1
fi

IS_JETSON=0
[[ -f /etc/nv_tegra_release ]] && IS_JETSON=1

# Import from / so a source tree in cwd cannot shadow the installed package
py_has() { (cd / && "$PY" -c "import $1" >/dev/null 2>&1); }

echo "================================================================"
echo " OmniRisk Workspace: $ROOT_DIR"
echo " Platform: $([[ $IS_JETSON -eq 1 ]] && echo Jetson || echo x86)"
echo " Jobs: $NPROC"
echo "================================================================"

if [[ $BUILD_ONLY -eq 0 ]]; then
    # ── Step 1: system dependencies ──
    echo ""
    echo ">>> [1/5] Installing system dependencies..."
    APT_PKGS=(
        build-essential cmake git python3-venv python3-dev
        libeigen3-dev libpcl-dev libopencv-dev libyaml-cpp-dev libtbb-dev libarmadillo-dev libapr1-dev libboost-system-dev
        ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-tf ros-noetic-tf2-ros ros-noetic-eigen-conversions
        ros-noetic-nodelet ros-noetic-message-filters ros-noetic-mavros
    )
    if [[ $IS_JETSON -eq 1 ]]; then
        APT_PKGS+=(libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev)
    fi
    sudo apt-get update
    sudo apt-get install -y "${APT_PKGS[@]}"
    # MAVROS needs GeographicLib datasets at runtime
    if [[ ! -f /usr/share/GeographicLib/geoids/egm96-5.pgm ]]; then
        sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh
    fi

    # ── Step 2: Python venv ──
    echo ""
    echo ">>> [2/5] Setting up Python venv: $VENV"
    [[ -x "$PY" ]] || python3 -m venv "$VENV"
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r "$ROOT_DIR/OmniRisk/requirements.txt"

    if [[ $IS_JETSON -eq 1 ]]; then
        grep -q "R35" /etc/nv_tegra_release || echo "WARNING: the torch wheel targets JetPack 5.1.x (L4T R35)."
        if ! py_has torch; then
            "$PY" -m pip install Cython wheel
            "$PY" -m pip install --no-cache "$JETSON_TORCH_WHL"
        fi
        if ! py_has torchvision; then
            "$PY" -m pip install pybind11 "Pillow==9.5.0"
            mkdir -p "$DEPS_DIR"
            [[ -d "$DEPS_DIR/torchvision" ]] || git clone --branch v0.16.0 --depth 1 https://github.com/pytorch/vision "$DEPS_DIR/torchvision"
            (cd "$DEPS_DIR/torchvision" && BUILD_VERSION=0.16.0 "$PY" setup.py install)
        fi
        # TensorRT ships with JetPack; expose it inside the venv
        SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
        if [[ ! -e "$SITE/tensorrt" && ! -L "$SITE/tensorrt" ]]; then
            ln -s /usr/lib/python3.8/dist-packages/tensorrt "$SITE/tensorrt"
        fi
        if ! py_has torch2trt; then
            mkdir -p "$DEPS_DIR"
            [[ -d "$DEPS_DIR/torch2trt" ]] || git clone --depth 1 https://github.com/NVIDIA-AI-IOT/torch2trt "$DEPS_DIR/torch2trt"
            (cd "$DEPS_DIR/torch2trt" && "$PY" setup.py install)
        fi
    else
        py_has torch || "$PY" -m pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
            --extra-index-url https://download.pytorch.org/whl/cu118
    fi
    (cd / && "$PY" -c "import torch; print('>>> torch', torch.__version__, '| CUDA:', torch.cuda.is_available())")

    # ── Step 3: Livox-SDK2 (standalone cmake, must be installed before catkin) ──
    echo ""
    echo ">>> [3/5] Building Livox-SDK2..."
    SDK_DIR="$ROS_WS/src/Livox-SDK2"
    mkdir -p "$SDK_DIR/build"
    cd "$SDK_DIR/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$NPROC"
    sudo make install
    sudo ldconfig
else
    echo ">>> [1-3/5] Skipping apt / venv / Livox-SDK2 (--build-only)."
fi

# catkin must use system python: the venv's empy 4.x breaks message generation
if [[ -n "$VIRTUAL_ENV" ]]; then
    PATH="${PATH//"$VIRTUAL_ENV/bin:"/}"
    unset VIRTUAL_ENV
fi
source /opt/ros/noetic/setup.bash

# ── Step 4: ros_ws ──
echo ""
echo ">>> [4/5] Building ros_ws..."
cd "$ROS_WS"
catkin_make -DCMAKE_BUILD_TYPE=Release -DROS_EDITION=ROS1 -j"$NPROC"

# ── Step 5: simulation workspaces ──
if [[ $NO_SIM -eq 0 ]]; then
    echo ""
    echo ">>> [5/5] Building Controller and Simulator..."
    cd "$ROOT_DIR/Controller"
    catkin_make -DCMAKE_BUILD_TYPE=Release -j"$NPROC"
    cd "$ROOT_DIR/Simulator"
    catkin_make -DCMAKE_BUILD_TYPE=Release -j"$NPROC"
else
    echo ">>> [5/5] Skipping Controller and Simulator (--no-sim)."
fi

echo ""
echo "================================================================"
echo " Build complete!"
echo "   source $ROS_WS/devel/setup.bash"
echo "   source $VENV/bin/activate"
echo "================================================================"
