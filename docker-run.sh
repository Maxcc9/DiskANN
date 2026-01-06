#!/usr/bin/env bash
# DiskANN Docker 開發環境啟動腳本
# 使用當前用戶身份運行，避免 root 權限問題

set -euo pipefail

IMAGE="${1:-diskann:latest}"

# 檢查映像是否存在
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "❌ 映像 $IMAGE 不存在"
    echo "請先執行設置腳本："
    echo "  ./setup-docker.sh"
    exit 1
fi

echo "啟動 DiskANN 開發容器..."
echo "映像: $IMAGE"
echo "掛載: $(pwd) -> /workspace"
echo "用戶: $(id -un) ($(id -u):$(id -g))"

docker run \
  --hostname diskann-dev \
  --rm -it \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd)":/workspace \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /dev:/dev \
  --privileged \
  -w /workspace \
  "$IMAGE" /bin/bash
