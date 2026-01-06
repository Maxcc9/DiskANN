#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.

FROM ubuntu:jammy

LABEL maintainer="DiskANN Development" \
      version="1.0" \
      description="DiskANN development environment with monitoring and analysis tools"

# 設置時區與語言環境
ENV TZ=UTC LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive

# 合併 apt 指令以減少層數，添加開發與監測工具
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
        # 編譯工具
        git make cmake g++ clang-format \
        # DiskANN 依賴
        libaio-dev libgoogle-perftools-dev libunwind-dev \
        libboost-dev libboost-program-options-dev \
        libmkl-full-dev libcpprest-dev \
        # OpenMP 運行時（修正 libomp.so.5 缺失問題）
        libomp-dev \
        # Python 開發工具
        python3.10 python3-pip python3-dev \
        # 性能監測工具
        nvme-cli sysstat linux-tools-generic htop iotop \
        # 溫度監控
        lm-sensors \
        # 調試工具
        gdb valgrind strace \
        # 便利工具
        vim less tree curl wget ca-certificates && \
    apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 驗證關鍵依賴安裝成功
RUN ldconfig && \
    pkg-config --exists libaio libboost_program_options || \
    (echo "ERROR: 關鍵依賴未正確安裝" && exit 1)

# 升級 pip 並安裝 Python 依賴（使用 requirements.txt 確保版本固定）
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt && \
    pip3 cache purge

# 為開發使用：掛載本地代碼而非 clone
WORKDIR /workspace

# 如果需要在容器內編譯，取消註解以下行：
# RUN mkdir -p build && \
#     cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
#     cmake --build build -- -j$(nproc)

# ==========================================
# 推薦運行方式（避免 root 權限問題）：
# ==========================================
# docker run \
#   --hostname diskann-dev \
#   --rm -it \
#   -u $(id -u):$(id -g) \
#   -v "$(pwd)":/workspace \
#   -v /etc/passwd:/etc/passwd:ro \
#   -v /etc/group:/etc/group:ro \
#   -v /dev:/dev \
#   --privileged \
#   -w /workspace \
#   diskann:latest /bin/bash
#
# 說明：
# -u $(id -u):$(id -g)   使用當前用戶身份運行（檔案不會變成 root 擁有）
# -v /etc/passwd:ro      讓容器識別用戶名
# -v /dev:/dev           訪問 NVMe 設備（溫度監控）
# --privileged           允許硬體監控
#
# 編譯範例：
# cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
# cmake --build build -- -j$(nproc)
