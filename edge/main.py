"""
ファースト肯定ペンギン - エッジ処理メインループ

実行方法:
    cd edge
    python main.py

カメラが必要なため、Docker コンテナ内ではなくネイティブで実行すること。
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
import numpy as np
try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None
from PIL import ImageFont, ImageDraw, Image
from dotenv import load_dotenv

from appsync_notifier import AppSyncNotifier
from ar_overlay import AROverlay
from comment_receiver import CommentReceiver
from face_tracker import FaceTracker, TrackedPerson, draw_debug

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.INFO)
log = logging.getLogger(__name__)

# -- config --

AR_OVERLAY_SCALE_X = 4.0
AR_OVERLAY_SCALE_Y = 1.0

_JP_FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
_JP_FONT_SIZE = 28
_JP_FONT = ImageFont.truetype(_JP_FONT_PATH, _JP_FONT_SIZE)


def _load_config() -> dict:
    try:
        camera_index = int(os.getenv("CAMERA_INDEX", 0))
    except ValueError:
        sys.exit("[ERROR] CAMERA_INDEX は整数で指定してください")

    try:
        ar_display_sec = int(os.getenv("AR_DISPLAY_DURATION_SEC", 10))
    except ValueError:
        sys.exit("[ERROR] AR_DISPLAY_DURATION_SEC は整数で指定してください")

    return {
        "camera_index": camera_index,
        "ar_display_sec": ar_display_sec,
        "asset_path": os.getenv(
            "ASSET_PATH", str(Path(__file__).parent.parent / "assets" / "penguin.png")
        ),
        "s3_bucket": os.getenv("S3_BUCKET_NAME", ""),
        "s3_raw_prefix": os.getenv("S3_RAW_PREFIX", "raw/"),
        "s3_composed_prefix": os.getenv("S3_COMPOSED_PREFIX", "composed/"),
        "aws_endpoint_url": os.getenv("AWS_ENDPOINT_URL"),
        "appsync_endpoint": os.getenv("APPSYNC_ENDPOINT", ""),
        "appsync_api_key": os.getenv("APPSYNC_API_KEY", ""),
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


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _get_text_size(text: str) -> tuple[int, int]:
    bbox = _JP_FONT.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _put_jp_text(frame, text: str, x: int, y: int, color_rgb: tuple[int, int, int]):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    # 縁取り
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=_JP_FONT, fill=(0, 0, 0))
    draw.text((x, y), text, font=_JP_FONT, fill=color_rgb)
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_comment(frame, comment):
    text = comment.text
    color = _hex_to_bgr(comment.color)
    x, y = int(comment.x), comment.y
    alpha = 0.5

    tw, th = _get_text_size(text)

    pad_x, pad_y = 20, 14
    body_w = tw + pad_x * 2
    body_h = th + pad_y * 2
    tail_w = int(body_h * 0.7)
    fin_h = int(body_h * 0.3)

    cx = x + tw // 2
    cy = y - th // 2

    total_w = body_w + tail_w + 10
    total_h = body_h + fin_h + 10
    rx1 = max(0, cx - body_w // 2 - 5)
    ry1 = max(0, cy - body_h // 2 - 5)
    rx2 = min(frame.shape[1], rx1 + total_w)
    ry2 = min(frame.shape[0], ry1 + total_h)
    if rx2 <= rx1 or ry2 <= ry1:
        return

    overlay = frame[ry1:ry2, rx1:rx2].copy()
    fish = overlay.copy()

    lcx = cx - rx1
    lcy = cy - ry1

    # 胴体（楕円）
    body_rx = body_w // 2
    body_ry = body_h // 2
    cv2.ellipse(fish, (lcx, lcy), (body_rx, body_ry), 0, 0, 360, color, -1)

    # 尾びれ（二股の三角形）
    tail_base_x = lcx + body_rx - 5
    tail_tip_x = tail_base_x + tail_w
    tail_spread = int(body_ry * 0.9)
    tail_pts_upper = np.array([
        [tail_base_x, lcy - 2],
        [tail_tip_x, lcy - tail_spread],
        [tail_tip_x - tail_w // 3, lcy],
    ], dtype=np.int32)
    tail_pts_lower = np.array([
        [tail_base_x, lcy + 2],
        [tail_tip_x, lcy + tail_spread],
        [tail_tip_x - tail_w // 3, lcy],
    ], dtype=np.int32)
    cv2.fillPoly(fish, [tail_pts_upper], color)
    cv2.fillPoly(fish, [tail_pts_lower], color)

    # 背びれ（上の三角）
    dorsal_pts = np.array([
        [lcx - body_rx // 3, lcy - body_ry + 2],
        [lcx, lcy - body_ry - fin_h],
        [lcx + body_rx // 3, lcy - body_ry + 2],
    ], dtype=np.int32)
    cv2.fillPoly(fish, [dorsal_pts], color)

    # 腹びれ（下の小さい三角）
    pelvic_pts = np.array([
        [lcx + body_rx // 4, lcy + body_ry - 2],
        [lcx + body_rx // 2, lcy + body_ry + fin_h // 2],
        [lcx - body_rx // 6, lcy + body_ry - 2],
    ], dtype=np.int32)
    cv2.fillPoly(fish, [pelvic_pts], color)

    # 目（白丸 + 黒瞳）
    eye_x = lcx - body_rx // 2
    eye_y = lcy - body_ry // 4
    eye_r = max(4, body_ry // 5)
    cv2.circle(fish, (eye_x, eye_y), eye_r, (255, 255, 255), -1)
    cv2.circle(fish, (eye_x - 1, eye_y), max(2, eye_r // 2), (0, 0, 0), -1)

    # 口（小さい弧）
    mouth_x = lcx - body_rx + 8
    mouth_y = lcy + 2
    cv2.ellipse(fish, (mouth_x, mouth_y), (5, 3), 0, 30, 150, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.addWeighted(fish, alpha, overlay, 1 - alpha, 0, overlay)
    frame[ry1:ry2, rx1:rx2] = overlay

    # テキスト（魚の胴体中央）- Pillow で日本語描画
    text_x = cx - tw // 2
    text_y = cy - th // 2
    _put_jp_text(frame, text, text_x, text_y, (255, 255, 255))


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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("カメラ解像度: %dx%d", actual_w, actual_h)

    try:
        overlay = AROverlay(cfg["asset_path"])
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")

    tracker = FaceTracker()
    executor = ThreadPoolExecutor(max_workers=2)

    s3_client = _create_s3_client(cfg["aws_endpoint_url"]) if cfg["s3_bucket"] else None
    notifier = AppSyncNotifier(cfg["appsync_endpoint"], cfg["appsync_api_key"])
    comment_receiver = CommentReceiver(cfg["appsync_endpoint"], cfg["appsync_api_key"], frame_width=actual_w)
    comment_receiver.start()

    vcam = None
    if pyvirtualcam is not None:
        try:
            vcam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=30)
            log.info("仮想カメラ起動: %s (%dx%d)", vcam.device, actual_w, actual_h)
        except RuntimeError as e:
            log.warning("仮想カメラ起動失敗（OBS未設定）: %s — 仮想カメラなしで続行", e)

    penguin_states: dict[int, _PenguinState] = {}
    camera_fail_count = 0

    log.info("'q' キーで終了。拍手を検知するとペンギンが出現します。")

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

            # -- visual clap detection --
            upload_composed = False
            targets = tracker.find_clapping_persons(frame.shape, tracks)

            if targets:
                for target in targets:
                    penguin_states[target.track_id] = _PenguinState(
                        expire_time=now + cfg["ar_display_sec"],
                        last_bbox=target.face_bbox,
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
                for target in targets:
                    executor.submit(
                        notifier.notify,
                        target.track_id,
                        f"s3://{cfg['s3_bucket']}/{cfg['s3_raw_prefix']}",
                    )
                upload_composed = True

            # -- expire penguin states --
            penguin_states = {
                tid: st for tid, st in penguin_states.items() if now < st.expire_time
            }

            # -- update last_bbox + extend if still clapping --
            current_clappers = {t.track_id for t in targets}
            for tid, state in penguin_states.items():
                person = next((t for t in tracks if t.track_id == tid), None)
                if person:
                    state.last_bbox = person.face_bbox
                    state.last_body_bbox = person.body_bbox
                if tid in current_clappers:
                    state.expire_time = now + cfg["ar_display_sec"]

            # -- AR overlay --
            composed = frame
            for state in penguin_states.values():
                if state.last_body_bbox:
                    x, y, w, h = state.last_body_bbox
                else:
                    x, y, w, h = state.last_bbox
                ow = int(w * AR_OVERLAY_SCALE_X)
                oh = int(h * AR_OVERLAY_SCALE_Y)
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

            # -- ニコニコ風コメント描画 --
            comments = comment_receiver.get_comments()
            alive = []
            for c in comments:
                c.x -= c.speed
                if c.x > -500:
                    _draw_comment(composed, c)
                    alive.append(c)
            comment_receiver.update_comments(alive)

            debug_frame = draw_debug(composed, tracks, yolo_result, tracker.clap_trackers)
            if vcam is not None:
                vcam.send(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB))
            cv2.imshow("First Penguin AR", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if vcam is not None:
            vcam.close()
        notifier.send_summary()
        comment_receiver.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True, cancel_futures=True)
        log.info("終了しました。")


if __name__ == "__main__":
    main()
