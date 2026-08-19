# syntax=docker/dockerfile:1
#
# Training image for OpenVLA on an air-gapped Grace/Blackwell (GB200 NVL72) partition.
#
# Base: an NGC PyTorch container. NGC ships a torch build with sm_100 (Blackwell) kernels
# already compiled for the image's exact CUDA version -- the alternative, a CUDA-devel base
# plus a source build of flash-attn, means compiling flash-attn on a 2-vCPU arm64 GitHub
# runner, which does not finish in a workflow's time budget. Pinned via build arg so bumping
# it is a one-line change; see docs/airgap-vla-plan.md Section 3 for the full reasoning.
#
# NGC 25.12-py3 specifics (checked 2026-08-19): torch 2.10.0a0+b4e4ee81d3, CUDA 13.1.0,
# Python 3.12, Ubuntu 24.04. Documented to publish a linux/arm64 manifest for Grace systems
# (GH200/GB200), and NGC containers have been Blackwell-optimized since the 25.01 release.
ARG NGC_TAG=25.12-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

LABEL org.opencontainers.image.title="openvla-airgap" \
      org.opencontainers.image.description="OpenVLA training image for air-gapped Grace-Blackwell (GB200 NVL72)" \
      org.opencontainers.image.source="https://github.com/mgb45/openvla"

# Offline-by-default: nothing at runtime should reach for the network. `scripts/airgap/hf_fetch.py`
# explicitly unsets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE when it needs to fetch, so baking these
# in here does not block the staging-side use of that script inside this same image.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    TF_CPP_MIN_LOG_LEVEL=3 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WANDB_MODE=offline

WORKDIR /opt/openvla

# `dlimp` installs from a git URL (see below); confirm git is present rather than assuming the
# base image has it. Idempotent and cheap if it's already there.
RUN command -v git >/dev/null || (apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*)

# Capture the exact versions the base image already shipped for the packages we must not
# silently replace. NGC's torch build carries a non-PyPI local version segment (e.g.
# `2.10.0a0+b4e4ee81d3`), so it cannot be expressed as a normal version pin in a
# repo-committed constraints file -- and pyproject.toml deliberately does not list torch/
# torchvision/torchaudio as dependencies, so a plain `pip install -e .` would not touch them
# anyway. This file exists as a second line of defense: if some *other* dependency declares a
# torch version bound the installed build doesn't satisfy, pip fails the build loudly right
# here, in CI -- instead of silently upgrading torch and breaking Blackwell support on the
# cluster three weeks from now.
RUN python -m pip freeze | grep -Ei '^(torch|torchvision|torchaudio|numpy)==' > /tmp/base-constraints.txt \
    && echo "Base image ships:" && cat /tmp/base-constraints.txt

COPY docker/constraints.txt /tmp/repo-constraints.txt

# CPU-only TensorFlow for the RLDS input pipeline (prismatic/vla/datasets/rlds/dataset.py
# already disables GPU visibility for TF at import time). Installed as its own layer, ahead of
# the repo copy, so it's cached across ordinary code changes.
RUN python -m pip install --no-cache-dir \
      -c /tmp/base-constraints.txt -c /tmp/repo-constraints.txt \
      tensorflow tensorflow_datasets "dlimp @ git+https://github.com/moojink/dlimp_openvla"

COPY . /opt/openvla

# `tensorflow_graphics` (DROID-only; see docs/airgap-vla-plan.md Section 6) is deliberately
# NOT installed here -- it is unmaintained, aarch64 wheel availability is unverified, and it
# is only needed for the Tier-3 data mixture, which is explicitly deferred.
RUN python -m pip install --no-cache-dir \
      -c /tmp/base-constraints.txt -c /tmp/repo-constraints.txt \
      -e .

# Build-time self-check: import every path the training entrypoints touch, and print what
# actually got installed. This is the gate that catches a broken dependency resolution in CI,
# rather than three weeks into a cluster job. GPUs are not expected to be visible during a CI
# build, so this only asserts things that are true without one.
RUN python - <<'PYCHECK'
import torch
import tensorflow as tf
import transformers
import timm
import draccus
import prismatic  # noqa: F401
from prismatic.vla.datasets import RLDSDataset  # noqa: F401
from prismatic.models.backbones.llm.base_llm import _resolve_attn_implementation

print(f"torch          {torch.__version__}  (cuda build: {torch.version.cuda})")
print(f"tensorflow     {tf.__version__}  (GPUs visible to TF: {len(tf.config.list_physical_devices('GPU'))}, should be 0)")
print(f"transformers   {transformers.__version__}")
print(f"timm           {timm.__version__}")

assert len(tf.config.list_physical_devices("GPU")) == 0, "TensorFlow can see a GPU -- it must stay CPU-only"

resolved = _resolve_attn_implementation(use_flash_attention_2=True)
print(f"attn backend   {resolved}  (flash-attn requested; auto-falls back to sdpa if unavailable)")
if resolved == "flash_attention_2":
    import flash_attn
    print(f"  flash_attn {flash_attn.__version__} is importable")
else:
    print("  flash_attn is NOT importable in this image -- training will run on sdpa."
          " Not a build failure: sdpa is numerically correct on Blackwell, just slower."
          " See prismatic/models/backbones/llm/base_llm.py for the fallback logic.")
PYCHECK

ENTRYPOINT []
CMD ["/bin/bash"]
