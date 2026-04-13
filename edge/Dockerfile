FROM python:3.11-slim

# システム依存パッケージ
# libgl1 / libglib2.0-0: OpenCV のヘッドレス描画に必要
# portaudio19-dev: PyAudio（マイク入力）に必要
# libgomp1: MediaPipe の並列処理に必要
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    portaudio19-dev \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードは docker-compose の volume mount で提供するため
# コンテナビルド時のコピーは不要（開発時のホットリロードを優先）

CMD ["python", "main.py"]
