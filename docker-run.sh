#!/usr/bin/env bash
# DiskANN Docker 開發環境啟動腳本

set -euo pipefail

IMAGE="${1:-diskann:latest}"
MODE="${2:-dev}"  # dev 或 perf

# 錯誤處理
error_exit() {
    echo "❌ 錯誤: $1" >&2
    exit 1
}

# 檢查映像是否存在
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    error_exit "映像 $IMAGE 不存在，請先執行設置腳本：./setup-docker.sh"
fi

echo "啟動 DiskANN Docker 環境..."
echo "映像: $IMAGE"
echo "模式: $MODE"
echo "掛載: $(pwd) -> /workspace"

# 檢查 /dev/nvme0 是否存在
nvme_device="/dev/nvme0"
if [[ ! -e "$nvme_device" ]]; then
    echo "⚠️  警告: $nvme_device 不存在，溫度監控功能不可用"
    nvme_device=""
fi

# 自動偵測 CPU 核心數
num_cpus=$(nproc 2>/dev/null || echo 8)
if [[ $num_cpus -lt 4 ]]; then
    num_cpus=4
fi

# 效能測試模式：檢查並提示暫停其他用戶進程
if [[ "$MODE" == "perf" ]]; then
    echo ""
    echo "⚠️  效能測試模式"
    echo ""
    
    # 只檢查真實登入用戶（who 命令的輸出）
    current_user=$(whoami)
    logged_in_users=$(who 2>/dev/null | awk '{print $1}' | sort -u || echo "")
    other_users=$(echo "$logged_in_users" | grep -v "^${current_user}$" | grep -v '^$' | tr '\n' ' ' | sed 's/ $//' || echo "")
    
    if [[ -n "$other_users" ]]; then
        echo "⚠️  偵測到其他登入用戶: $other_users"
        echo ""
        echo "選項："
        echo "  1) 暫停他們的進程（需要 root 權限）"
        echo "  2) 繼續（可能被 IO 干擾）"
        echo "  3) 取消"
        
        choice=""
        read -p "請選擇 [1/2/3] (預設 2，30秒後自動選擇): " -t 30 choice || choice="2"
        
        if [[ -z "$choice" ]]; then
            choice="2"
        fi
        
        # 輸入驗證
        if ! [[ "$choice" =~ ^[123]$ ]]; then
            echo "❌ 無效選擇，使用預設值 2"
            choice="2"
        fi
        
        if [[ "$choice" == "1" ]]; then
            echo ""
            echo "嘗試暫停其他用戶的進程..."
            for user in $other_users; do
                if killall -STOP -u "$user" 2>/dev/null; then
                    echo "  ✓ 暫停 $user 的進程"
                else
                    echo "  ⚠️  無法暫停 $user（可能需要 root 權限或無法訪問該進程）"
                fi
            done
            
            # 容器退出時恢復進程（使用單引號避免展開）
            trap 'echo ""; echo "恢復進程..."; for u in '"$other_users"'; do killall -CONT -u "$u" 2>/dev/null || true; done; echo "完成"' EXIT INT TERM
            echo "進程暫停已設置（容器退出時自動恢復）"
        elif [[ "$choice" == "3" ]]; then
            echo "已取消"
            exit 0
        fi
    else
        echo "沒有其他登入用戶"
    fi
    
    # 檢查磁盤空間
    available_space=$(df /tmp 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
    if [[ $available_space -lt 1048576 ]]; then  # 1GB in KB
        echo "⚠️  警告: 磁盤空間即將不足"
    fi
    
    # 計算記憶體限制（系統內存的 75%，最多 32GB）
    total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    mem_limit_kb=$((total_mem_kb * 3 / 4))
    mem_limit_gb=$((mem_limit_kb / 1048576))
    if [[ $mem_limit_gb -gt 32 ]]; then
        mem_limit_gb=32
    fi
    if [[ $mem_limit_gb -lt 4 ]]; then
        mem_limit_gb=4
    fi
    
    echo ""
    echo "資源限制："
    echo "  CPU: $num_cpus 核（自動偵測）"
    echo "  記憶體: ${mem_limit_gb}GB"
    echo ""
    
    docker run \
      --hostname diskann-perf \
      --rm -it \
      --cpus "$num_cpus" \
      --memory "${mem_limit_gb}g" \
      --cpuset-cpus "0-$((num_cpus-1))" \
      -u "$(id -u):$(id -g)" \
      -v "$(pwd)":/workspace \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      ${nvme_device:+-v $nvme_device:$nvme_device} \
      -v /dev:/dev \
      --privileged \
      -w /workspace \
      "$IMAGE" /bin/bash || error_exit "容器啟動失敗"
else
    # 開發模式
    docker run \
      --hostname diskann-dev \
      --rm -it \
      -u "$(id -u):$(id -g)" \
      -v "$(pwd)":/workspace \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      ${nvme_device:+-v $nvme_device:$nvme_device} \
      -v /dev:/dev \
      --privileged \
      -w /workspace \
      "$IMAGE" /bin/bash || error_exit "容器啟動失敗"
fi

echo ""
echo "✓ 容器已退出"
