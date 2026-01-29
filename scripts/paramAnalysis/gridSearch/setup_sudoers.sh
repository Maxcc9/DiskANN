#!/bin/bash
# DiskANN: 設定免密碼執行 drop_caches 的 sudoers 規則
# 用途：讓 search_batch.sh 可以自動化執行，不需要每次輸入密碼

set -euo pipefail

SUDOERS_FILE="/tmp/diskann-drop-caches"
INSTALL_PATH="/etc/sudoers.d/diskann-drop-caches"

echo "=== DiskANN: 設定 sudoers 免密碼清除快取 ==="
echo ""
echo "這將允許用戶 '$USER' 免密碼執行以下命令："
echo "  - sync"
echo "  - echo 3 | sudo tee /proc/sys/vm/drop_caches"
echo ""

# 產生 sudoers 規則
cat > "$SUDOERS_FILE" << EOF
# DiskANN benchmark: 允許免密碼執行 drop_caches 與 nvme smart-log
# 用途：讓 search_batch.sh 可以自動清除系統快取、讀取 NVMe 溫度，確保每次 benchmark 從冷啟動開始
# 安全性：只允許特定命令，不會開放完整 sudo 權限
# 創建時間：$(date)
# 用戶：$USER

# 允許免密碼執行 sync
$USER ALL=(root) NOPASSWD: /usr/bin/sync

# 允許免密碼執行 tee 寫入 drop_caches
$USER ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches

# 允許免密碼執行 sh -c（備用方案）
$USER ALL=(root) NOPASSWD: /bin/sh -c echo [0-9] > /proc/sys/vm/drop_caches

# 允許免密碼執行 nvme smart-log（讀取 SSD 溫度）
$USER ALL=(root) NOPASSWD: /usr/sbin/nvme smart-log *
$USER ALL=(root) NOPASSWD: /usr/bin/nvme smart-log *
EOF

echo "✓ 已產生 sudoers 規則檔案: $SUDOERS_FILE"
echo ""
cat "$SUDOERS_FILE"
echo ""

# 驗證語法
if ! visudo -cf "$SUDOERS_FILE" 2>/dev/null; then
    echo "✗ 錯誤：sudoers 語法驗證失敗" >&2
    exit 1
fi

echo "✓ sudoers 語法驗證通過"
echo ""

# 詢問是否安裝
read -p "是否安裝此 sudoers 規則？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消安裝"
    echo "如需手動安裝，請執行："
    echo "  sudo cp $SUDOERS_FILE $INSTALL_PATH"
    echo "  sudo chmod 440 $INSTALL_PATH"
    exit 0
fi

# 安裝
echo "正在安裝..."
sudo cp "$SUDOERS_FILE" "$INSTALL_PATH"
sudo chmod 440 "$INSTALL_PATH"
sudo chown root:root "$INSTALL_PATH"

echo ""
echo "✓ 安裝完成！"
echo ""

# 測試
echo "=== 測試免密碼清除快取 ==="
if echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
    echo "✓ 測試成功！可以免密碼清除快取"
else
    echo "✗ 測試失敗！請檢查配置" >&2
    exit 1
fi

echo ""
echo "=== 設定完成 ==="
echo "現在可以執行 search_batch.sh 而不需要輸入密碼"
echo ""
echo "如需移除此規則："
echo "  sudo rm $INSTALL_PATH"
