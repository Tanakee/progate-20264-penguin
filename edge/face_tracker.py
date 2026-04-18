import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# YOLOv8-pose COCO キーポイントインデックス
_KP_LEFT_SHOULDER  = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_ELBOW     = 7
_KP_RIGHT_ELBOW    = 8
_KP_LEFT_WRIST     = 9
_KP_RIGHT_WRIST    = 10

_ARM_CONNECTIONS = [
    (_KP_LEFT_SHOULDER,  _KP_LEFT_ELBOW),
    (_KP_LEFT_ELBOW,     _KP_LEFT_WRIST),
    (_KP_RIGHT_SHOULDER, _KP_RIGHT_ELBOW),
    (_KP_RIGHT_ELBOW,    _KP_RIGHT_WRIST),
]


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
    MediaPipe Face Detection で顔トラッキング、
    YOLOv8-pose で複数人の姿勢推定（拍手検出）を行う。
    """

    TRACK_TIMEOUT_SEC       = 1.5
    IOU_THRESHOLD           = 0.3
    CLAP_DIST_RATIO         = 0.15   # 両手首間距離がフレーム幅のこの割合以下なら拍手
    CLAP_FACE_MAX_DIST_RATIO = 0.4   # 拍手中心から顔中心までの最大距離（フレーム幅の割合）
    MIN_FACE_SIZE_RATIO     = 0.04
    FACE_ASPECT_RATIO_RANGE = (0.5, 2.0)
    KP_CONF_THRESHOLD       = 0.3    # キーポイントの信頼度下限

    def __init__(self, model_selection: int = 1, min_confidence: float = 0.9,
                 yolo_model: str = "yolov8n-pose.pt"):
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )
        self._yolo = YOLO(yolo_model)
        self._tracks: list[TrackedFace] = []
        self._next_id = 0

    def process(self, frame_bgr: np.ndarray) -> tuple[list[TrackedFace], object]:
        """
        フレームを受け取り、トラッキング済み顔リストと YOLO 推論結果を返す。
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        face_result = self._face_det.process(frame_rgb)
        yolo_result = self._yolo(frame_bgr, verbose=False)[0]

        h, w = frame_bgr.shape[:2]
        detected = self._extract_bboxes(face_result, w, h)
        self._update_tracks(detected)
        self._expire_tracks()

        return self._tracks, yolo_result

    def find_clapping_faces(
        self,
        yolo_result,
        frame_shape: tuple,
        tracks: list[TrackedFace],
    ) -> list[TrackedFace]:
        """
        拍手ジェスチャーをしている全員の顔を返す。
        各人物の両手首座標を YOLO から取得し、近接していれば拍手と判定する。
        """
        if not tracks:
            return []

        h, w = frame_shape[:2]
        clap_centers = self._detect_clap_centers(yolo_result, w, h)

        if not clap_centers:
            return []

        max_dist = w * self.CLAP_FACE_MAX_DIST_RATIO
        result: list[TrackedFace] = []
        used_track_ids: set[int] = set()

        for center in clap_centers:
            candidates = [t for t in tracks if t.track_id not in used_track_ids]
            if not candidates:
                break
            closest = min(candidates, key=lambda t: _dist(t.center, center))
            if _dist(closest.center, center) <= max_dist:
                result.append(closest)
                used_track_ids.add(closest.track_id)
            else:
                print(f"[YOLO] 最近傍の顔が遠すぎるため除外 "
                      f"(dist={_dist(closest.center, center):.0f}px > {max_dist:.0f}px)", flush=True)

        return result

    # ── private ──────────────────────────────────────────────────────────

    def _extract_bboxes(self, face_result, w: int, h: int) -> list[tuple]:
        boxes = []
        if not face_result.detections:
            return boxes
        min_size = w * self.MIN_FACE_SIZE_RATIO
        ar_min, ar_max = self.FACE_ASPECT_RATIO_RANGE
        for det in face_result.detections:
            bb = det.location_data.relative_bounding_box
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            if bw < min_size or bh < min_size:
                continue
            aspect = bh / bw if bw > 0 else 0
            if not (ar_min <= aspect <= ar_max):
                continue
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            boxes.append((x, y, bw, bh))
        return boxes

    def _update_tracks(self, detected: list[tuple]):
        for bbox in detected:
            best_iou, best_track = 0.0, None
            for track in self._tracks:
                iou = _calc_iou(bbox, track.bbox)
                if iou > best_iou:
                    best_iou, best_track = iou, track
            if best_track and best_iou >= self.IOU_THRESHOLD:
                best_track.bbox = bbox
                best_track.last_seen = time.monotonic()
            else:
                self._tracks.append(TrackedFace(track_id=self._next_id, bbox=bbox))
                self._next_id += 1

    def _expire_tracks(self):
        now = time.monotonic()
        self._tracks = [t for t in self._tracks if now - t.last_seen < self.TRACK_TIMEOUT_SEC]

    CLAP_CENTER_DEDUP_RATIO = 0.1  # この距離以内の拍手中心は同一人物として重複除去

    def _detect_clap_centers(self, yolo_result, w: int, h: int) -> list[tuple[int, int]]:
        """YOLO の推論結果から拍手している人全員の手首中点を返す。"""
        centers = []
        kps = yolo_result.keypoints
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            print("[YOLO] 人物未検出", flush=True)
            return centers

        conf = kps.conf  # shape: (num_persons, 17) or None
        print(f"[YOLO] 検出人数={len(kps.xy)}", flush=True)

        for i, person_kps in enumerate(kps.xy):
            lw = person_kps[_KP_LEFT_WRIST].cpu().numpy()
            rw = person_kps[_KP_RIGHT_WRIST].cpu().numpy()

            # キーポイントの信頼度チェック
            if conf is not None:
                person_conf = conf[i].cpu().numpy()
                lw_conf = person_conf[_KP_LEFT_WRIST]
                rw_conf = person_conf[_KP_RIGHT_WRIST]
                print(f"[YOLO] person={i} 手首信頼度 L={lw_conf:.2f} R={rw_conf:.2f} (閾値={self.KP_CONF_THRESHOLD})", flush=True)
                if lw_conf < self.KP_CONF_THRESHOLD or rw_conf < self.KP_CONF_THRESHOLD:
                    print(f"[YOLO] person={i} 手首信頼度不足でスキップ", flush=True)
                    continue

            # 座標が (0, 0) の場合は未検出
            if lw[0] == 0 and lw[1] == 0:
                print(f"[YOLO] person={i} 左手首未検出", flush=True)
                continue
            if rw[0] == 0 and rw[1] == 0:
                print(f"[YOLO] person={i} 右手首未検出", flush=True)
                continue

            dist = np.linalg.norm(lw - rw)
            print(f"[YOLO] person={i} 両手首間距離={dist:.0f}px (閾値={w * self.CLAP_DIST_RATIO:.0f}px)", flush=True)
            if dist < w * self.CLAP_DIST_RATIO:
                center = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))
                # 既存の拍手中心と近すぎる場合は同一人物の二重検出として除外
                dedup_dist = w * self.CLAP_CENTER_DEDUP_RATIO
                if any(np.linalg.norm(np.array(center) - np.array(c)) < dedup_dist for c in centers):
                    print(f"[YOLO] person={i} 二重検出のためスキップ", flush=True)
                    continue
                print(f"[YOLO] person={i} 拍手検出! 中心={center}", flush=True)
                centers.append(center)

        return centers

    def close(self):
        self._face_det.close()


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


# ── デバッグ描画 ──────────────────────────────────────────────────────────

def draw_debug(frame: np.ndarray, tracks: list[TrackedFace], yolo_result) -> np.ndarray:
    """顔BBox・YOLOの腕スケルトン・手首を描画する。"""
    out = frame.copy()
    h, w = out.shape[:2]

    # 顔BBox
    for track in tracks:
        x, y, bw, bh = track.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(out, f"id={track.track_id}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # YOLO 腕スケルトン
    kps = yolo_result.keypoints
    if kps is not None and kps.xy is not None:
        conf = kps.conf
        for i, person_kps in enumerate(kps.xy):
            pts = person_kps.cpu().numpy().astype(int)  # (17, 2)
            person_conf = conf[i].cpu().numpy() if conf is not None else None

            def visible(idx):
                if person_conf is not None and person_conf[idx] < 0.3:
                    return False
                return not (pts[idx][0] == 0 and pts[idx][1] == 0)

            # 腕の接続線
            for a, b in _ARM_CONNECTIONS:
                if visible(a) and visible(b):
                    cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)

            # 手首を強調
            for idx, label in [(_KP_LEFT_WRIST, "L"), (_KP_RIGHT_WRIST, "R")]:
                if visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 10, (0, 60, 255), -1)
                    cv2.putText(out, f"wrist {label}", (pts[idx][0] + 8, pts[idx][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 255), 1)

            # 肩・肘
            for idx in [_KP_LEFT_SHOULDER, _KP_RIGHT_SHOULDER,
                        _KP_LEFT_ELBOW, _KP_RIGHT_ELBOW]:
                if visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 6, (0, 200, 255), -1)

    return out
