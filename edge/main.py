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
from face_tracker import FaceTracker, draw_debug

load_dotenv(Path(__file__).parent.parent / ".env")

# ── 設定 ─────────────────────────────────────────────────────────────────
CAMERA_INDEX        = int(os.getenv("CAMERA_INDEX", 0))
MIC_INDEX           = int(v) if (v := os.getenv("MIC_INDEX")) else None
CLAP_THRESHOLD_RMS  = int(os.getenv("CLAP_THRESHOLD_RMS", 2000))
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

    # トラッキング状態: {track_id: expire_time}
    penguin_expires: dict[int, float] = {}

    detector.start(device_index=MIC_INDEX)
    print("'q' キーで終了します。拍手を検知するとペンギンが出現します。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] フレーム取得失敗。カメラ接続を確認してください。")
                break

            raw_frame = frame.copy()  # S3 送信用の原画像
            tracks, yolo_result = tracker.process(frame)
            now = time.monotonic()

            # ── 拍手検知 ──────────────────────────────────────────────────
            if detector.consume():
                targets = tracker.find_clapping_faces(frame.shape, tracks)
                if targets:
                    for target in targets:
                        penguin_expires[target.track_id] = now + AR_DISPLAY_SEC
                        print(f"[検知] ファーストペンギン! track_id={target.track_id} center={target.center}")
                    # 原画像をS3にアップロード（拍手検知時の1枚のみ）
                    executor.submit(_upload_to_s3, raw_frame, S3_RAW_PREFIX, "raw")
                    # AR合成済み画像もこのフレームの1枚のみアップロード
                    upload_composed = True
                else:
                    print("[検知] 拍手音を検知しましたが顔を特定できませんでした（誤検知として無視）")
                    upload_composed = False
            else:
                upload_composed = False

            # 期限切れエントリを削除
            penguin_expires = {tid: exp for tid, exp in penguin_expires.items() if now < exp}

            # ── AR合成 ────────────────────────────────────────────────────
            composed = frame
            for tid in penguin_expires:
                face = next((t for t in tracks if t.track_id == tid), None)
                if face:
                    x, y, w, h = face.bbox
                    ow = int(w * 1.5)
                    oh = int(h * 1.5)
                    ox = x - (ow - w) // 2
                    oy = y - oh // 2
                    composed = overlay.apply(composed, ox, oy, ow, oh)

            # 合成画像を S3 にアップロード（拍手検知時の1枚のみ）
            if upload_composed:
                executor.submit(_upload_to_s3, composed, S3_COMPOSED_PREFIX, "composed")

            cv2.imshow("First Penguin AR", draw_debug(composed, tracks, yolo_result))
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
