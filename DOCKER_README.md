# DiskANN Docker 開發環境

快速啟動 DiskANN 開發與性能測試環境的 Docker 容器。

## 首次設置（新電腦/新用戶）

```bash
cd /path/to/DiskANN
./setup-docker.sh
```

這會自動：
- ✅ 檢查 Docker 是否安裝
- ✅ 檢查 Docker 權限
- ✅ 建置 `diskann:latest` 映像

## 日常使用

```bash
./docker-run.sh
```

容器內自動：
- 掛載當前目錄到 `/workspace`
- 使用你的用戶身份（避免 root 權限問題）
- 可存取 NVMe 設備（溫度監控）

## 編譯與運行

容器內執行：

```bash
# 編譯
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -- -j$(nproc)

# 運行參數分析流程
cd scripts/paramAnalysis/gridSearch
bash build_batch.sh --build-csv ./inputFiles/build_configs.csv
ENABLE_IOSTAT=1 COOLDOWN_TEMP_C=60 bash search_batch.sh
```

## 手動安裝 Docker（若未安裝）

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker  # 或登出後重新登入
```

## 檔案說明

- `setup-docker.sh`: 首次設置腳本
- `docker-run.sh`: 啟動容器腳本
- `Dockerfile`: 映像定義

## 環境包含

- DiskANN 編譯依賴（MKL, Boost, libaio 等）
- 性能監測工具（nvme-cli, iostat, htop）
- Python 資料科學套件（pandas, matplotlib, scikit-learn, xgboost）
- Jupyter 環境
- 調試工具（gdb, valgrind）
