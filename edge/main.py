"""
ファースト肯定ペンギン - エッジ処理メインループ

実行方法:
    cd edge
    python main.py

カメラ・マイクが必要なため、Docker コンテナ内ではなくネイティブで実行すること。
"""

import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
import cv2
from dotenv import load_dotenv

from ar_overlay import AROverlay
from clap_detector import ClapDetector
from face_tracker import FaceTracker, TrackedFace, draw_debug

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- config --

AR_OVERLAY_SCALE = 1.5


def _load_config() -> dict:
    try:
        camera_index = int(os.getenv("CAMERA_INDEX", 0))
    except ValueError:
        sys.exit("[ERROR] CAMERA_INDEX は整数で指定してください")

    mic_raw = os.getenv("MIC_INDEX")
    try:
        mic_index = int(mic_raw) if mic_raw else None
    except ValueError:
        sys.exit("[ERROR] MIC_INDEX は整数で指定してください")

    try:
        clap_threshold = int(os.getenv("CLAP_THRESHOLD_RMS", 2000))
    except ValueError:
        sys.exit("[ERROR] CLAP_THRESHOLD_RMS は整数で指定してください")

    try:
        ar_display_sec = int(os.getenv("AR_DISPLAY_DURATION_SEC", 10))
    except ValueError:
        sys.exit("[ERROR] AR_DISPLAY_DURATION_SEC は整数で指定してください")

    return {
        "camera_index": camera_index,
        "mic_index": mic_index,
        "clap_threshold": clap_threshold,
        "ar_display_sec": ar_display_sec,
        "asset_path": os.getenv(
            "ASSET_PATH", str(Path(__file__).parent.parent / "assets" / "penguin.png")
        ),
        "s3_bucket": os.getenv("S3_BUCKET_NAME", ""),
        "s3_raw_prefix": os.getenv("S3_RAW_PREFIX", "raw/"),
        "s3_composed_prefix": os.getenv("S3_COMPOSED_PREFIX", "composed/"),
        "aws_endpoint_url": os.getenv("AWS_ENDPOINT_URL"),
    }


# -- S3 upload --


def _create_s3_client(endpoint_url: str | None):
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)


def _upload_to_s3(s3_client, bucket: str, img, prefix: str, label: str):
    if not bucket:
        return
    try:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        key = f"{prefix}{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=buf.tobytes(), ContentType="image/jpeg"
        )
        log.info("[S3] %s -> s3://%s/%s", label, bucket, key)
    except Exception as e:
        log.error("[S3] アップロード失敗 (%s): %s", label, e)


# -- AR state --

CAMERA_RETRY_MAX = 30


@dataclass
class _PenguinState:
    """AR 表示中の状態。顔トラックが消えても最終位置を保持する。"""
    expire_time: float
    last_bbox: tuple[int, int, int, int]
    last_body_bbox: tuple[int, int, int, int] | None = None


def main():
    cfg = _load_config()

    cap = cv2.VideoCapture(cfg["camera_index"])
    if not cap.isOpened():
        sys.exit(f"[ERROR] カメラ {cfg['camera_index']} を開けません")

    try:
        overlay = AROverlay(cfg["asset_path"])
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")

    tracker = FaceTracker()
    detector = ClapDetector(threshold_rms=cfg["clap_threshold"])
    executor = ThreadPoolExecutor(max_workers=2)

    s3_client = _create_s3_client(cfg["aws_endpoint_url"]) if cfg["s3_bucket"] else None

    penguin_states: dict[int, _PenguinState] = {}
    camera_fail_count = 0

    mic_ok = detector.start(device_index=cfg["mic_index"])
    if mic_ok:
        log.info("'q' キーで終了。拍手を検知するとペンギンが出現します。")
    else:
        log.info("'q' キーで終了。マイクなしで起動しました。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                camera_fail_count += 1
                if camera_fail_count >= CAMERA_RETRY_MAX:
                    log.error("カメラフレーム取得が %d 回連続失敗。終了します。", CAMERA_RETRY_MAX)
                    break
                log.warning("フレーム取得失敗 (%d/%d)", camera_fail_count, CAMERA_RETRY_MAX)
                continue
            camera_fail_count = 0

            raw_frame = frame.copy()
            tracks, yolo_result = tracker.process(frame)
            now = time.monotonic()

            # -- clap detection --
            upload_composed = False
            if detector.consume():
                log.debug(
                    "音声検知時バッファ内ジェスチャー数: %d", tracker.gesture_buffer_size
                )
                targets = tracker.find_clapping_faces(frame.shape, tracks)
                if targets:
                    detector.acknowledge()
                    for target in targets:
                        penguin_states[target.track_id] = _PenguinState(
                            expire_time=now + cfg["ar_display_sec"],
                            last_bbox=target.bbox,
                            last_body_bbox=target.body_bbox,
                        )
                        log.info(
                            "ファーストペンギン! track_id=%d center=%s",
                            target.track_id,
                            target.center,
                        )
                    if s3_client:
                        executor.submit(
                            _upload_to_s3,
                            s3_client,
                            cfg["s3_bucket"],
                            raw_frame,
                            cfg["s3_raw_prefix"],
                            "raw",
                        )
                    upload_composed = True
                else:
                    log.info("拍手音を検知しましたが顔を特定できませんでした")

            # -- expire penguin states --
            penguin_states = {
                tid: st for tid, st in penguin_states.items() if now < st.expire_time
            }

            # -- update last_bbox for still-visible faces --
            for tid, state in penguin_states.items():
                face = next((t for t in tracks if t.track_id == tid), None)
                if face:
                    state.last_bbox = face.bbox
                    state.last_body_bbox = face.body_bbox

            # -- AR overlay (uses body_bbox for full-body penguin) --
            composed = frame
            for state in penguin_states.values():
                if state.last_body_bbox:
                    x, y, w, h = state.last_body_bbox
                else:
                    x, y, w, h = state.last_bbox
                ow = int(w * AR_OVERLAY_SCALE)
                oh = int(h * AR_OVERLAY_SCALE)
                ox = x - (ow - w) // 2
                oy = y - (oh - h) // 2
                composed = overlay.apply(composed, ox, oy, ow, oh)

            if upload_composed and s3_client:
                executor.submit(
                    _upload_to_s3,
                    s3_client,
                    cfg["s3_bucket"],
                    composed,
                    cfg["s3_composed_prefix"],
                    "composed",
                )

            cv2.imshow("First Penguin AR", draw_debug(composed, tracks, yolo_result))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True, cancel_futures=True)
        log.info("終了しました。")


if __name__ == "__main__":
    main()
