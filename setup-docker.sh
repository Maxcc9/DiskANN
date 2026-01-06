#!/usr/bin/env bash
# DiskANN Docker 環境一鍵設置腳本
# 適用於新電腦或新用戶環境

set -euo pipefail

# 錯誤處理
trap 'echo "❌ 設置失敗"; exit 1' ERR

echo "=========================================="
echo "DiskANN Docker 環境設置"
echo "=========================================="
echo ""

# 1. 檢查 Dockerfile 是否存在
if [[ ! -f "Dockerfile" ]]; then
    echo "❌ 未找到 Dockerfile，請確保在專案根目錄執行此腳本"
    exit 1
fi

echo "✓ Dockerfile 找到"

# 2. 檢查 requirements.txt
if [[ ! -f "requirements.txt" ]]; then
    echo "❌ 未找到 requirements.txt，請確保在專案根目錄執行此腳本"
    exit 1
fi

echo "✓ requirements.txt 找到"

# 3. 檢查磁盤空間（至少需要 5GB）
available_space=$(df . | awk 'NR==2 {print $4}')
if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
    echo "❌ 磁盤空間不足（需要 5GB 以上，現有 $((available_space/1048576))GB）"
    exit 1
fi

echo "✓ 磁盤空間充足（$((available_space/1048576))GB）"

# 4. 檢查 Docker 是否安裝
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

# 5. 檢查 Docker daemon 是否運行
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker daemon 未運行或權限不足"
    echo "請執行以下命令："
    echo "  sudo usermod -aG docker $(whoami)"
    echo "  newgrp docker  # 或登出後重新登入"
    exit 1
fi

echo "✓ Docker 權限正常"
echo ""

# 6. 建置映像
IMAGE_NAME="diskann:latest"

if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️  映像 $IMAGE_NAME 已存在"
    read -p "是否重新建置? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "✓ 跳過建置，使用現有映像"
    else
        echo "開始建置映像（這可能需要 5-10 分鐘）..."
        if ! docker build -t "$IMAGE_NAME" . 2>&1; then
            echo "❌ 映像建置失敗，請檢查 Docker 日誌"
            exit 1
        fi
        echo "✓ 映像建置完成"
    fi
else
    echo "開始建置映像 $IMAGE_NAME（這可能需要 5-10 分鐘）..."
    if ! docker build -t "$IMAGE_NAME" . 2>&1; then
        echo "❌ 映像建置失敗"
        exit 1
    fi
    echo "✓ 映像建置完成"
fi

# 7. 驗證映像大小和可用性
echo ""
image_size=$(docker images "$IMAGE_NAME" --format "{{.Size}}")
echo "映像大小: $image_size"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "❌ 映像驗證失敗"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 設置完成！"
echo "=========================================="
echo ""
echo "啟動開發環境："
echo "  ./docker-run.sh"
echo ""
echo "啟動效能測試模式："
echo "  ./docker-run.sh diskann:latest perf"
echo ""
