#!/usr/bin/env bash
# DiskANN Docker 環境一鍵設置腳本
# 適用於新電腦或新用戶環境

set -euo pipefail

echo "=========================================="
echo "DiskANN Docker 環境設置"
echo "=========================================="
echo ""

# 1. 檢查 Docker 是否安裝
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ 未檢測到 Docker，請先安裝："
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install -y docker.io"
    echo "  sudo systemctl enable --now docker"
    echo "  sudo usermod -aG docker \$USER"
    echo "  newgrp docker  # 或登出後重新登入"
    echo ""
    exit 1
fi

echo "✓ Docker 已安裝: $(docker --version)"

# 2. 檢查 Docker 權限
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker 權限不足，請執行："
    echo "  sudo usermod -aG docker $(whoami)"
    echo "  newgrp docker  # 或登出後重新登入"
    exit 1
fi

echo "✓ Docker 權限正常"
echo ""

# 3. 建置映像
IMAGE_NAME="diskann:latest"

if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️  映像 $IMAGE_NAME 已存在"
    read -p "是否重新建置? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳過建置，使用現有映像"
    else
        echo "開始建置映像..."
        docker build -t "$IMAGE_NAME" .
        echo "✓ 映像建置完成"
    fi
else
    echo "開始建置映像 $IMAGE_NAME ..."
    docker build -t "$IMAGE_NAME" .
    echo "✓ 映像建置完成"
fi

echo ""
echo "=========================================="
echo "設置完成！"
echo "=========================================="
echo ""
echo "啟動開發環境："
echo "  ./docker-run.sh"
echo ""
echo "或手動運行："
echo "  docker run --rm -it -u \$(id -u):\$(id -g) \\"
echo "    -v \"\$(pwd)\":/workspace \\"
echo "    -v /etc/passwd:/etc/passwd:ro \\"
echo "    -v /etc/group:/etc/group:ro \\"
echo "    -v /dev:/dev --privileged \\"
echo "    -w /workspace $IMAGE_NAME /bin/bash"
echo ""
