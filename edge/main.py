"""
ファースト肯定ペンギン - エッジ処理メインループ

実行方法:
    cd edge
    python main.py

カメラ・マイクが必要なため、Docker コンテナ内ではなくネイティブで実行すること。
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import cv2
from dotenv import load_dotenv

from ar_overlay import AROverlay
from clap_detector import ClapDetector
from face_tracker import FaceTracker

load_dotenv()

# ── 設定 ─────────────────────────────────────────────────────────────────
CAMERA_INDEX        = int(os.getenv("CAMERA_INDEX", 0))
MIC_INDEX           = int(v) if (v := os.getenv("MIC_INDEX")) else None
CLAP_THRESHOLD_RMS  = int(os.getenv("CLAP_THRESHOLD_RMS", 3000))
AR_DISPLAY_SEC      = int(os.getenv("AR_DISPLAY_DURATION_SEC", 10))
ASSET_PATH          = os.getenv("ASSET_PATH", str(Path(__file__).parent.parent / "assets" / "penguin.png"))
S3_BUCKET           = os.getenv("S3_BUCKET_NAME", "")
S3_RAW_PREFIX       = os.getenv("S3_RAW_PREFIX", "raw/")
S3_COMPOSED_PREFIX  = os.getenv("S3_COMPOSED_PREFIX", "composed/")
AWS_ENDPOINT_URL    = os.getenv("AWS_ENDPOINT_URL")  # LocalStack 用


def _upload_to_s3(img, prefix: str, label: str):
    """S3 に JPEG としてアップロードする（非同期で呼ばれる）。"""
    if not S3_BUCKET:
        return
    try:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        key = f"{prefix}{int(time.time() * 1000)}.jpg"
        kwargs = {}
        if AWS_ENDPOINT_URL:
            kwargs["endpoint_url"] = AWS_ENDPOINT_URL
        s3 = boto3.client("s3", **kwargs)
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.tobytes(), ContentType="image/jpeg")
        print(f"[S3] {label} → s3://{S3_BUCKET}/{key}")
    except Exception as e:
        print(f"[S3] アップロード失敗 ({label}): {e}", file=sys.stderr)


def main():
    # ── 初期化 ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"[ERROR] カメラ {CAMERA_INDEX} を開けませんでした。CAMERA_INDEX を確認してください。")

    try:
        overlay = AROverlay(ASSET_PATH)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")

    tracker = FaceTracker()
    detector = ClapDetector(threshold_rms=CLAP_THRESHOLD_RMS)
    executor = ThreadPoolExecutor(max_workers=2)

    # トラッキング状態
    penguin_track_id: int | None = None
    penguin_expire: float = 0.0

    detector.start(device_index=MIC_INDEX)
    print("='q' キーで終了します。拍手を検知するとペンギンが出現します。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] フレーム取得失敗。カメラ接続を確認してください。")
                break

            raw_frame = frame.copy()  # Rekognition 送信用の原画像
            tracks, hand_result = tracker.process(frame)
            now = time.monotonic()

            # ── 拍手検知 ──────────────────────────────────────────────────
            if detector.consume():
                target = tracker.find_clapping_face(hand_result, frame.shape, tracks)
                if target:
                    penguin_track_id = target.track_id
                    penguin_expire = now + AR_DISPLAY_SEC
                    print(f"[検知] ファーストペンギン! track_id={target.track_id} center={target.center}")
                    # 原画像を S3 に非同期アップロード
                    executor.submit(_upload_to_s3, raw_frame, S3_RAW_PREFIX, "raw")
                else:
                    print("[検知] 拍手音を検知しましたが顔を特定できませんでした（誤検知として無視）")

            # ── AR合成 ────────────────────────────────────────────────────
            composed = frame
            if penguin_track_id is not None and now < penguin_expire:
                face = next((t for t in tracks if t.track_id == penguin_track_id), None)
                if face:
                    x, y, w, h = face.bbox
                    # 顔BBoxより1.5倍大きく、上方向にオフセットして被せる
                    ow = int(w * 1.5)
                    oh = int(h * 1.5)
                    ox = x - (ow - w) // 2
                    oy = y - oh // 2
                    composed = overlay.apply(frame, ox, oy, ow, oh)
                else:
                    # 顔ロスト：タイムアウトまで待ち、次フレームで再検出を試みる
                    pass
            elif now >= penguin_expire:
                penguin_track_id = None

            # 合成画像を S3 に非同期アップロード（ペンギン表示中のみ）
            if penguin_track_id is not None:
                executor.submit(_upload_to_s3, composed, S3_COMPOSED_PREFIX, "composed")

            cv2.imshow("First Penguin AR", composed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=False)
        print("終了しました。")


if __name__ == "__main__":
    main()
