# ファースト肯定ペンギン

最初に拍手をした観客（ファーストペンギン）を検知してARでペンギン化し、その肯定アクションをAWSでデータ化・分析するシステム。

---

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/Tanakee/progate-20264-penguin.git
cd progate-20264-penguin
```

### 2. 環境変数を設定

```bash
cp .env.example .env
```

`.env` を開き、Slackで共有された実際の値を入力してください。
**`.env` は絶対にコミットしないでください。**

### 3. Docker環境を起動（LocalStack / Lambda開発用）

```bash
docker compose up --build
```

### 4. エッジ処理をネイティブ実行（カメラ・マイクを使う場合）

> **注意**: カメラ・マイクはDockerコンテナからアクセスできません。
> 本番実行は必ずネイティブで行ってください。

```bash
cd edge
pip install -r requirements.txt
python main.py
```

---

## OS別の注意事項

### macOS

- カメラ・マイクはネイティブ実行のみ対応（Docker不可）
- OpenCVのウィンドウ表示をDockerコンテナ内で行う場合は [XQuartz](https://www.xquartz.org/) が必要

```bash
# XQuartz インストール後
xhost +localhost
# docker-compose.yml の DISPLAY コメントを解除して起動
```

### Windows 11

- Docker Desktop の Backend を **WSL2** に設定すること
  - Docker Desktop → Settings → General → **Use WSL 2 based engine** をON
- WSL2 + WSLg により GUI（OpenCVウィンドウ）は追加設定なしで動作
- カメラ・マイクはネイティブ実行のみ対応（Docker不可）

### Windows 10

- Docker Desktop の Backend を **WSL2** に設定すること
- OpenCVのウィンドウ表示をDockerコンテナ内で行う場合は [VcXsrv](https://sourceforge.net/projects/vcxsrv/) が必要
  1. VcXsrv をインストール・起動
  2. 起動オプションで **「Disable access control」** にチェック
  3. `docker-compose.yml` の `DISPLAY=host.docker.internal:0.0` のコメントを解除
- カメラ・マイクはネイティブ実行のみ対応（Docker不可）

---

## プロジェクト構成

```
.
├── edge/                        # エッジ処理（カメラ・マイク・AR合成）
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py                  # エントリポイント（要実装）
├── lambda/                      # AWS Lambda関数
│   ├── Dockerfile
│   ├── requirements.txt
│   └── handler.py               # Lambda ハンドラ（要実装）
├── assets/                      # ペンギンARアセット（画像等）
├── infra/
│   └── localstack-init/
│       └── 01_setup.sh          # LocalStack起動時にS3バケットを自動作成
├── docker-compose.yml
├── .env.example                 # 環境変数テンプレート
└── 要件定義書.md
```

---

## Dockerサービス一覧

| サービス | 役割 | URL |
|---|---|---|
| `edge` | エッジ処理コンテナ（ロジックテスト用） | - |
| `lambda-dev` | Lambda関数の開発・テスト用 | - |
| `localstack` | ローカルAWSモック（S3, Lambda等） | http://localhost:4566 |

---

## よくあるトラブル

**`docker compose up` でLocalStackが起動しない**
```bash
# Dockerが起動しているか確認
docker info

# ポート4566が使用中でないか確認
lsof -i :4566
```

**`01_setup.sh: bad interpreter` エラー（Windows）**

シェルスクリプトの改行コードがCRLFになっています。`.gitattributes` で防止していますが、なった場合は以下で修正してください。

```bash
sed -i 's/\r//' infra/localstack-init/01_setup.sh
```

**`mediapipe` のインストールが失敗する（ネイティブ実行時）**

Python 3.11 を使っているか確認してください。

```bash
python --version  # 3.11.x であること
```

**カメラが認識されない**

`.env` の `CAMERA_INDEX` を変更して試してください（0, 1, 2 ...）。

```bash
# 接続中のカメラ一覧を確認（macOS）
system_profiler SPCameraDataType
```
