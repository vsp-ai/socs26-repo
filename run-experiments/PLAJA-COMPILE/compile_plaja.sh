#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PLAJA_DOCKER_IMAGE:-victorsputrich/plaja_dependencies-chaahat:MRv0.5.1-roundingsat}"
PLAJA_DIR="${1:-}"

if [[ -z "$PLAJA_DIR" ]]; then
  read -r -p "PlaJA directory: " PLAJA_DIR
fi

if [[ -z "$PLAJA_DIR" ]]; then
  echo "[error] missing PlaJA directory" >&2
  exit 1
fi

PLAJA_DIR="${PLAJA_DIR/#\~/$HOME}"
PLAJA_DIR="$(cd "$PLAJA_DIR" && pwd)"

if [[ ! -f "$PLAJA_DIR/CMakeLists.txt" ]]; then
  echo "[error] expected CMakeLists.txt in: $PLAJA_DIR" >&2
  exit 1
fi

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 2)}"

echo "[compile] PlaJA directory: $PLAJA_DIR"
echo "[compile] Docker image: $IMAGE"
echo "[compile] jobs: $JOBS"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e JOBS="$JOBS" \
  -v "$PLAJA_DIR":/ws \
  -w /ws \
  "$IMAGE" \
  bash -lc 'set -euo pipefail
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PB_CONSTRAINTS=OFF ..
    make -j"${JOBS}" PlaJA
  '

echo "[done] $PLAJA_DIR/build/PlaJA"
