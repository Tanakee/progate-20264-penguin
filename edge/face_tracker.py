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
    MediaPipe Face Detection + Hands + Pose を同一フレームで処理し、
    IoUベースのシンプルなトラッキングで顔IDを維持する。
    拍手検出は Hands を優先し、手が隠れている場合は Pose でフォールバックする。
    """

    TRACK_TIMEOUT_SEC = 3.0   # この秒数検出されなければトラックを削除
    IOU_THRESHOLD = 0.3
    CLAP_DIST_RATIO = 0.15    # 両手首間距離がフレーム幅のこの割合以下なら拍手と判定
    CLAP_FACE_MAX_DIST_RATIO = 0.4  # 拍手中心から顔中心までの最大距離（フレーム幅の割合）

    def __init__(self, model_selection: int = 0, min_confidence: float = 0.8):
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
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self._tracks: list[TrackedFace] = []
        self._next_id = 0

    def process(self, frame_bgr: np.ndarray) -> tuple[list[TrackedFace], object, object]:
        """
        フレームを受け取り、トラッキング済み顔リストと手・姿勢検出結果を返す。
        RGB変換は1回だけ行い全モデルで共有する。
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # MediaPipe 内部コピーを抑制

        face_result = self._face_det.process(frame_rgb)
        hand_result = self._hands.process(frame_rgb)
        pose_result = self._pose.process(frame_rgb)

        h, w = frame_bgr.shape[:2]
        detected = self._extract_bboxes(face_result, w, h)
        self._update_tracks(detected)
        self._expire_tracks()

        return self._tracks, hand_result, pose_result

    def find_clapping_face(
        self,
        hand_result,
        pose_result,
        frame_shape: tuple,
        tracks: list[TrackedFace],
    ) -> TrackedFace | None:
        """
        拍手ポーズを検出し、その手に最も近い顔を返す。
        Hands で検出できない場合は Pose の手首座標でフォールバックする。
        手のジェスチャーが確認できない場合・顔が手から離れすぎている場合は None を返す。
        """
        h, w = frame_shape[:2]
        clap_center = self._detect_clap_from_hands(hand_result, w, h)

        if clap_center is None:
            clap_center = self._detect_clap_from_pose(pose_result, w, h)

        if clap_center is None or not tracks:
            return None

        max_dist = w * self.CLAP_FACE_MAX_DIST_RATIO
        closest = min(tracks, key=lambda t: _dist(t.center, clap_center))

        if _dist(closest.center, clap_center) > max_dist:
            print(f"[Motion] 最近傍の顔が遠すぎるため除外 (dist={_dist(closest.center, clap_center):.0f}px > {max_dist:.0f}px)", flush=True)
            return None

        return closest

    # ── private ──────────────────────────────────────────────────────────

    # 顔として認める最小サイズ（フレーム幅に対する割合）
    MIN_FACE_SIZE_RATIO = 0.04

    def _extract_bboxes(self, face_result, w: int, h: int) -> list[tuple]:
        boxes = []
        if not face_result.detections:
            return boxes
        min_size = w * self.MIN_FACE_SIZE_RATIO
        for det in face_result.detections:
            bb = det.location_data.relative_bounding_box
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            # 小さすぎる検出は誤検知として除外
            if bw < min_size or bh < min_size:
                continue
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
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

    def _detect_clap_from_hands(self, hand_result, w: int, h: int) -> tuple[int, int] | None:
        """Hands モデルで両手首が閾値距離以内なら拍手と判定し、その中点を返す。"""
        num_hands = len(hand_result.multi_hand_landmarks) if hand_result.multi_hand_landmarks else 0
        print(f"[Hands] 検出された手の数={num_hands}", flush=True)

        if not hand_result.multi_hand_landmarks or len(hand_result.multi_hand_landmarks) < 2:
            return None

        wrists = []
        for landmarks in hand_result.multi_hand_landmarks:
            lm = landmarks.landmark[0]  # 手首
            wrists.append(np.array([lm.x * w, lm.y * h]))

        dist = np.linalg.norm(wrists[0] - wrists[1])
        print(f"[Hands] 両手首間距離={dist:.0f}px (閾値={w * self.CLAP_DIST_RATIO:.0f}px)", flush=True)
        if dist < w * self.CLAP_DIST_RATIO:
            center = ((wrists[0] + wrists[1]) / 2).astype(int)
            print(f"[Hands] 拍手ジェスチャー検出! 中心={tuple(center)}", flush=True)
            return (int(center[0]), int(center[1]))
        return None

    def _detect_clap_from_pose(self, pose_result, w: int, h: int) -> tuple[int, int] | None:
        """Pose モデルの手首ランドマークで拍手を判定し、その中点を返す（Hands のフォールバック）。"""
        if not pose_result.pose_landmarks:
            return None

        lm = pose_result.pose_landmarks.landmark
        PL = mp.solutions.pose.PoseLandmark
        lw = np.array([lm[PL.LEFT_WRIST].x * w, lm[PL.LEFT_WRIST].y * h])
        rw = np.array([lm[PL.RIGHT_WRIST].x * w, lm[PL.RIGHT_WRIST].y * h])

        dist = np.linalg.norm(lw - rw)
        print(f"[Pose] 両手首間距離={dist:.0f}px (閾値={w * self.CLAP_DIST_RATIO:.0f}px)", flush=True)
        if dist < w * self.CLAP_DIST_RATIO:
            center = ((lw + rw) / 2).astype(int)
            print(f"[Pose] 拍手ジェスチャー検出! 中心={tuple(center)}", flush=True)
            return (int(center[0]), int(center[1]))
        return None

    def close(self):
        self._face_det.close()
        self._hands.close()
        self._pose.close()


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
