#!/usr/bin/env bash
# Install the CPU-only NeMo-Gym service runtime on an H100 host.
#
# Port of the internal Ascend runtime-install script with the CUDA image and
# neutral cache paths. NeMo-Gym is a CPU orchestration layer: it never touches
# the GPUs, so the same training image can host it.
set -Eeuo pipefail

WORK_ROOT=${WORK_ROOT:-/data/lexbrowser-rl}
NEMO_GYM_ROOT=${NEMO_GYM_ROOT:-$WORK_ROOT/cache/nemo-gym-v0.2.1}
RUNTIME_SITE=${RUNTIME_SITE:-$WORK_ROOT/runtime/nemo-gym-site}
IMAGE=${IMAGE:-lexbrowser-verl-h100:local}
NEMO_GYM_REPO=${NEMO_GYM_REPO:-https://github.com/NVIDIA-NeMo/Gym.git}
NEMO_GYM_TAG=${NEMO_GYM_TAG:-v0.2.1}
NEMO_GYM_COMMIT=${NEMO_GYM_COMMIT:-27e921137042dcdb8a39c7169128619b9108074b}

mkdir -p "$(dirname "$NEMO_GYM_ROOT")" "$RUNTIME_SITE"
if [[ ! -d "$NEMO_GYM_ROOT/.git" ]]; then
  git clone --depth 1 --branch "$NEMO_GYM_TAG" "$NEMO_GYM_REPO" "$NEMO_GYM_ROOT"
fi

actual_commit=$(git -C "$NEMO_GYM_ROOT" rev-parse HEAD)
if [[ "$actual_commit" != "$NEMO_GYM_COMMIT" ]]; then
  echo "Unexpected NeMo-Gym commit: $actual_commit (expected $NEMO_GYM_COMMIT)" >&2
  exit 1
fi

# The training image already contains FastAPI, Ray, OpenAI, Hydra, datasets,
# and the other core dependencies. Keep the compatibility layer tiny.
docker run --rm --network host \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -v "$RUNTIME_SITE:/runtime" \
  --entrypoint pip "$IMAGE" install --target /runtime --no-deps --upgrade \
  yappi==1.7.6 devtools==0.12.2 gprof2dot==2025.4.14 pydot==4.0.1

docker run --rm --network host \
  -e PYTHONPATH=/runtime:/opt/nemo-gym \
  -v "$RUNTIME_SITE:/runtime:ro" \
  -v "$NEMO_GYM_ROOT:/opt/nemo-gym:ro" \
  --entrypoint python3 "$IMAGE" -c \
  'from nemo_gym.base_resources_server import SimpleResourcesServer; print("NEMO_GYM_RUNTIME_OK")'
