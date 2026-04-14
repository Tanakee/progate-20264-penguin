import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class TrackedFace:
    track_id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h) pixel座標
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


class FaceTracker:
    """
    MediaPipe Face Detection + Hands を同一フレームで処理し、
    IoUベースのシンプルなトラッキングで顔IDを維持する。
    """

    TRACK_TIMEOUT_SEC = 3.0   # この秒数検出されなければトラックを削除
    IOU_THRESHOLD = 0.3
    CLAP_DIST_RATIO = 0.15    # 両手首間距離がフレーム幅のこの割合以下なら拍手と判定

    def __init__(self, model_selection: int = 0, min_confidence: float = 0.6):
        """
        Args:
            model_selection: 0=近距離(~2m)、1=遠距離(~5m)
                             会場が広い場合は 1 に変更すること
        """
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=4,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self._tracks: list[TrackedFace] = []
        self._next_id = 0

    def process(self, frame_bgr: np.ndarray) -> tuple[list[TrackedFace], object]:
        """
        フレームを受け取り、トラッキング済み顔リストと手検出結果を返す。
        RGB変換は1回だけ行い両モデルで共有する。
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # MediaPipe 内部コピーを抑制

        face_result = self._face_det.process(frame_rgb)
        hand_result = self._hands.process(frame_rgb)

        h, w = frame_bgr.shape[:2]
        detected = self._extract_bboxes(face_result, w, h)
        self._update_tracks(detected)
        self._expire_tracks()

        return self._tracks, hand_result

    def find_clapping_face(
        self,
        hand_result,
        frame_shape: tuple,
        tracks: list[TrackedFace],
    ) -> TrackedFace | None:
        """
        拍手ポーズ（両手首が近接）を検出し、その手に最も近い顔を返す。
        手が検出できない場合はフレーム中央に最も近い顔を返す（フォールバック）。
        """
        h, w = frame_shape[:2]
        clap_center = self._detect_clap_center(hand_result, w, h)

        if not tracks:
            return None

        anchor = clap_center if clap_center else (w // 2, h // 2)
        return min(tracks, key=lambda t: _dist(t.center, anchor))

    # ── private ──────────────────────────────────────────────────────────

    def _extract_bboxes(self, face_result, w: int, h: int) -> list[tuple]:
        boxes = []
        if not face_result.detections:
            return boxes
        for det in face_result.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            boxes.append((x, y, bw, bh))
        return boxes

    def _update_tracks(self, detected: list[tuple]):
        matched_ids = set()
        for bbox in detected:
            best_iou, best_track = 0.0, None
            for track in self._tracks:
                iou = _calc_iou(bbox, track.bbox)
                if iou > best_iou:
                    best_iou, best_track = iou, track

            if best_track and best_iou >= self.IOU_THRESHOLD:
                best_track.bbox = bbox
                best_track.last_seen = time.monotonic()
                matched_ids.add(best_track.track_id)
            else:
                self._tracks.append(TrackedFace(track_id=self._next_id, bbox=bbox))
                self._next_id += 1

    def _expire_tracks(self):
        now = time.monotonic()
        self._tracks = [
            t for t in self._tracks
            if now - t.last_seen < self.TRACK_TIMEOUT_SEC
        ]

    def _detect_clap_center(self, hand_result, w: int, h: int) -> tuple[int, int] | None:
        """両手首が閾値距離以内なら拍手と判定し、その中点を返す。"""
        if not hand_result.multi_hand_landmarks or len(hand_result.multi_hand_landmarks) < 2:
            return None

        wrists = []
        for landmarks in hand_result.multi_hand_landmarks:
            lm = landmarks.landmark[0]  # 手首
            wrists.append(np.array([lm.x * w, lm.y * h]))

        # 最初の2手（複数ペアある場合は最も近い組み合わせにしてもよい）
        dist = np.linalg.norm(wrists[0] - wrists[1])
        if dist < w * self.CLAP_DIST_RATIO:
            center = ((wrists[0] + wrists[1]) / 2).astype(int)
            return (int(center[0]), int(center[1]))
        return None

    def close(self):
        self._face_det.close()
        self._hands.close()


# ── ユーティリティ ────────────────────────────────────────────────────────

def _calc_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))
