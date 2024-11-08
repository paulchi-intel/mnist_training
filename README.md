# MNIST 手寫數字辨識系統

## 專案描述

這是一個結合深度學習與網頁應用的 MNIST 手寫數字辨識系統。專案特色在於整合了 Intel oneAPI 技術，實現了在 Intel GPU 上的深度學習加速，同時也支援 NVIDIA GPU 和 CPU 運算。系統提供了友善的網頁介面，讓使用者可以即時觀察模型訓練過程，並進行手寫數字的辨識測試。

### 核心功能：

1. **多重運算裝置支援**：
   - 支援 Intel GPU (XPU)、NVIDIA GPU (CUDA) 和 CPU
   - 訓練過程中可動態切換運算裝置
   - 自動檢測並列出可用的運算資源

2. **視覺化訓練監控**：
   - 即時顯示訓練進度條和損失值
   - 動態更新的損失值和準確率曲線圖
   - 即時預覽當前訓練批次的圖像和預測結果
   - 訓練時間追蹤

3. **互動式手寫辨識**：
   - 網頁畫布即時手寫輸入
   - 實時數字辨識結果顯示
   - 一鍵清除重寫功能

4. **整合式環境設定**：
   - 自動化的 Intel oneAPI 環境配置
   - 一鍵式啟動腳本（app.bat）
   - 自動開啟瀏覽器並導向應用介面

### 應用場景：
- 深度學習教學示範
- Intel GPU 加速開發範例
- 手寫辨識系統原型
- PyTorch 與 Flask 整合參考

## 系統需求

- Python 3.8+
- PyTorch
- Flask
- Intel oneAPI（如需使用 Intel GPU）
- CUDA（如需使用 NVIDIA GPU）

## 安裝步驟

1. 克隆專案：
    ```bash
    git clone [repository_url]
    cd [repository_name]
    ```

2. 建立虛擬環境：
    ```bash
    python -m venv .venv
    ```

3. 啟動虛擬環境：
    - Windows:
        ```bash
        .venv\Scripts\activate
        ```
    - Linux/Mac:
        ```bash
        source .venv/bin/activate
        ```

4. 安裝依賴套件：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

### Windows 使用者（建議方式）

直接執行 `app.bat`，此批次檔會：
- 自動啟動虛擬環境
- 設定必要的 Intel oneAPI 環境變數
- 設定 PyTorch GPU 環境
- 啟動應用程式並開啟瀏覽器

## 系統架構

- `app.py`: Flask 應用程式主程式
- `mnist_model.py`: MNIST 神經網路模型定義
- `check_dll.py`: Intel oneAPI 環境檢查工具
- `app.bat`: Windows 啟動腳本
- `templates/`: 網頁模板目錄
  - `index.html`: 主要網頁介面

## 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 貢獻

歡迎提交 Issue 或 Pull Request 來協助改善專案。

## 作者

[Paul Chi]

## 致謝

- PyTorch 團隊
- Intel oneAPI 團隊
- MNIST 資料集提供者