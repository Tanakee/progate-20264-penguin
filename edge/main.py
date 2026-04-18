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
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

from ar_overlay import AROverlay
from clap_detector import ClapDetector
from face_tracker import FaceTracker

# Pose で描画対象にする腕・手首ランドマーク
_POSE_ARM_LANDMARKS = [
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
]
_POSE_ARM_CONNECTIONS = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,  mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW,     mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,    mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
]


def _draw_debug(frame: np.ndarray, tracks, hand_result, pose_result) -> np.ndarray:
    """デバッグ用：顔BBox・手・腕ランドマークを描画する。"""
    out = frame.copy()
    h, w = out.shape[:2]

    # 顔BBox
    for track in tracks:
        x, y, bw, bh = track.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(out, f"id={track.track_id}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # Hands：各手のランドマーク全点 + 手首を強調
    mp_drawing = mp.solutions.drawing_utils
    if hand_result.multi_hand_landmarks:
        for hand_lms in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                out, hand_lms, mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=1),
            )
            # 手首(landmark 0)を大きく強調
            lm0 = hand_lms.landmark[0]
            cx, cy = int(lm0.x * w), int(lm0.y * h)
            cv2.circle(out, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(out, "wrist(H)", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Pose：腕・手首ランドマークのみ描画
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks.landmark
        # 接続線
        for a, b in _POSE_ARM_CONNECTIONS:
            pa = (int(lm[a].x * w), int(lm[a].y * h))
            pb = (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(out, pa, pb, (0, 200, 255), 2)
        # 各ランドマーク点
        for pl in _POSE_ARM_LANDMARKS:
            px, py = int(lm[pl].x * w), int(lm[pl].y * h)
            is_wrist = pl in (
                mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            )
            color = (0, 80, 255) if is_wrist else (0, 200, 255)
            radius = 10 if is_wrist else 6
            cv2.circle(out, (px, py), radius, color, -1)
            cv2.putText(out, pl.name.replace("_", " ").lower(), (px + 6, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return out

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

    # トラッキング状態: {track_id: expire_time}
    penguin_expires: dict[int, float] = {}

    detector.start(device_index=MIC_INDEX)
    print("='q' キーで終了します。拍手を検知するとペンギンが出現します。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] フレーム取得失敗。カメラ接続を確認してください。")
                break

            raw_frame = frame.copy()  # Rekognition 送信用の原画像
            tracks, hand_result, pose_result = tracker.process(frame)
            now = time.monotonic()

            # ── 拍手検知 ──────────────────────────────────────────────────
            if detector.consume():
                target = tracker.find_clapping_face(hand_result, pose_result, frame.shape, tracks)
                if target:
                    penguin_expires[target.track_id] = now + AR_DISPLAY_SEC
                    print(f"[検知] ファーストペンギン! track_id={target.track_id} center={target.center}")
                    # 原画像を S3 に非同期アップロード
                    executor.submit(_upload_to_s3, raw_frame, S3_RAW_PREFIX, "raw")
                else:
                    print("[検知] 拍手音を検知しましたが顔を特定できませんでした（誤検知として無視）")

            # 期限切れエントリを削除
            penguin_expires = {tid: exp for tid, exp in penguin_expires.items() if now < exp}

            # ── AR合成 ────────────────────────────────────────────────────
            composed = frame
            for tid, exp in penguin_expires.items():
                face = next((t for t in tracks if t.track_id == tid), None)
                if face:
                    x, y, w, h = face.bbox
                    # 顔BBoxより1.5倍大きく、上方向にオフセットして被せる
                    ow = int(w * 1.5)
                    oh = int(h * 1.5)
                    ox = x - (ow - w) // 2
                    oy = y - oh // 2
                    composed = overlay.apply(composed, ox, oy, ow, oh)

            # 合成画像を S3 に非同期アップロード（ペンギン表示中のみ）
            if penguin_expires:
                executor.submit(_upload_to_s3, composed, S3_COMPOSED_PREFIX, "composed")

            debug = _draw_debug(composed, tracks, hand_result, pose_result)
            cv2.imshow("First Penguin AR", debug)
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
