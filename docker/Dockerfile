# hk3d in a Linux container: lets a build run under a hard memory cap
# (docker run --memory=...) instead of exhausting the host.
#
#   docker build -f docker/Dockerfile -t hk3d .
#   docker run --rm -v "$PWD/data:/data" -v "$PWD/cache:/cache" hk3d index
#
# The LandsD sheet folders are mounted at /data, outputs and caches go to /cache.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libgl1 libegl1 libglib2.0-0 libusb-1.0-0 libidn2-0 libgfortran5 fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy>=1.26,<2.3" "scipy>=1.11" "manifold3d>=3.0" "trimesh>=4.0" "pyproj>=3.6" \
    "fastapi>=0.110" "uvicorn>=0.29" "pillow>=10" "matplotlib>=3.8" "fast-simplification>=0.1.7" \
    "pytest>=8" \
 && (pip install --no-cache-dir "open3d>=0.18" || echo "open3d: no wheel for this platform")

WORKDIR /hk3d
COPY pyproject.toml ./
COPY hk3d ./hk3d
COPY tools ./tools
COPY tests ./tests
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg HK3D_DATA=/data HK3D_CACHE=/cache
ENTRYPOINT ["hk3d"]
CMD ["--help"]
