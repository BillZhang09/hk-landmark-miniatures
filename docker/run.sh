#!/bin/sh
# docker/run.sh [hk3d arguments...]
#
# Runs hk3d in the container under a hard memory cap with no swap, so a
# runaway build is killed inside the container (exit 137) instead of
# exhausting the host.
#
#   HK3D_DATA   folder holding the LandsD sheet folders (default ./data), mounted read-only
#   HK3D_CACHE  cache and output folder (default ./cache)
#   HK3D_MEM    memory cap (default 16g)
#   HK3D_LANDSD_KEY  passed through when set
#
# Build the image first:  docker build -f docker/Dockerfile -t hk3d .
# Example:  docker/run.sh build --hk80 834691 815600 --landmark --name BoC --out /cache/BoC
set -e
DATA=$(cd "${HK3D_DATA:-./data}" && pwd)
mkdir -p "${HK3D_CACHE:-./cache}"
CACHE=$(cd "${HK3D_CACHE:-./cache}" && pwd)
MEM=${HK3D_MEM:-16g}
PORT=${HK3D_PORT:-8765}
exec docker run --rm -it --memory="$MEM" --memory-swap="$MEM" \
  -v "$DATA:/data:ro" -v "$CACHE:/cache" -p "127.0.0.1:$PORT:8765" \
  ${HK3D_LANDSD_KEY:+-e HK3D_LANDSD_KEY} \
  hk3d "$@"
