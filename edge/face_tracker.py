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


@dataclass
class _GestureEntry:
    """拍手ジェスチャーの1フレーム分の記録。"""
    timestamp: float
    clap_center: tuple[int, int]
    person_bbox: tuple[float, float, float, float]  # YOLO の x1,y1,x2,y2


class FaceTracker:
    """
    MediaPipe Face Detection で顔トラッキング、
    YOLOv8s-pose で複数人の姿勢推定・拍手検出を行う。

    精度向上のための施策:
    - ジェスチャーバッファ: 直近1秒の検出履歴から音声と映像のタイミングズレを吸収
    - 複数フレーム確認: CLAP_CONFIRM_MIN フレーム以上検出されたジェスチャーのみ採用
    - 人物BBoxで顔マッチング: YOLO の人物領域を使って正確に顔を特定
    - スペクトル解析: ClapDetector 側で高周波エネルギー比率を追加判定
    """

    TRACK_TIMEOUT_SEC       = 1.5
    IOU_THRESHOLD           = 0.3
    CLAP_DIST_RATIO         = 0.15    # 両手首間距離がフレーム幅のこの割合以下なら拍手
    MIN_FACE_SIZE_RATIO     = 0.04
    FACE_ASPECT_RATIO_RANGE = (0.5, 2.0)
    KP_CONF_THRESHOLD       = 0.3     # キーポイントの信頼度下限
    CLAP_CENTER_DEDUP_RATIO = 0.1     # 同一フレーム内での二重検出除去距離
    GESTURE_BUFFER_SEC      = 1.0     # ジェスチャー履歴の保持秒数
    CLAP_CONFIRM_MIN        = 3       # 拍手確定に必要な最小検出フレーム数
    CLAP_GROUP_DIST_RATIO   = 0.15    # 同一人物のジェスチャーとみなす距離（フレーム幅比）

    def __init__(self, model_selection: int = 1, min_confidence: float = 0.9,
                 yolo_model: str = "yolov8s-pose.pt"):
        self._face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )
        self._yolo = YOLO(yolo_model)
        self._tracks: list[TrackedFace] = []
        self._next_id = 0
        self._gesture_buffer: list[_GestureEntry] = []

    def process(self, frame_bgr: np.ndarray) -> tuple[list[TrackedFace], object]:
        """
        フレームを受け取り、顔トラッキングと YOLO 推論を実行する。
        ジェスチャーバッファは毎フレーム自動更新される。
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        face_result = self._face_det.process(frame_rgb)
        yolo_result = self._yolo(frame_bgr, verbose=False)[0]

        h, w = frame_bgr.shape[:2]
        now = time.monotonic()

        detected = self._extract_bboxes(face_result, w, h)
        self._update_tracks(detected)
        self._expire_tracks()

        # 毎フレームジェスチャーを検出してバッファを更新
        new_gestures = self._detect_clap_gestures(yolo_result, w, h, now)
        self._update_gesture_buffer(new_gestures, now)

        return self._tracks, yolo_result

    def find_clapping_faces(
        self,
        frame_shape: tuple,
        tracks: list[TrackedFace],
    ) -> list[TrackedFace]:
        """
        バッファ内でCLAP_CONFIRM_MIN回以上検出された拍手ジェスチャーに対応する
        顔トラックのリストを返す。人物BBoxを使って顔を特定する。
        """
        if not tracks:
            return []

        h, w = frame_shape[:2]
        confirmed = self._get_confirmed_clap_gestures(w)
        if not confirmed:
            return []

        result: list[TrackedFace] = []
        used_ids: set[int] = set()
        for gesture in confirmed:
            face = self._find_face_in_person_bbox(gesture.person_bbox, tracks, used_ids)
            if face:
                result.append(face)
                used_ids.add(face.track_id)
            else:
                print(f"[Tracker] 人物BBox内に顔が見つかりませんでした", flush=True)

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

    def _detect_clap_gestures(
        self, yolo_result, w: int, h: int, now: float
    ) -> list[_GestureEntry]:
        """YOLO結果から拍手ジェスチャーを検出し GestureEntry のリストを返す。"""
        entries = []
        kps = yolo_result.keypoints
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            return entries

        conf = kps.conf
        boxes = yolo_result.boxes
        seen_centers: list[tuple] = []
        dedup_dist = w * self.CLAP_CENTER_DEDUP_RATIO

        for i, person_kps in enumerate(kps.xy):
            lw = person_kps[_KP_LEFT_WRIST].cpu().numpy()
            rw = person_kps[_KP_RIGHT_WRIST].cpu().numpy()

            if conf is not None:
                person_conf = conf[i].cpu().numpy()
                if (person_conf[_KP_LEFT_WRIST] < self.KP_CONF_THRESHOLD or
                        person_conf[_KP_RIGHT_WRIST] < self.KP_CONF_THRESHOLD):
                    continue

            if (lw[0] == 0 and lw[1] == 0) or (rw[0] == 0 and rw[1] == 0):
                continue

            dist = np.linalg.norm(lw - rw)
            if dist >= w * self.CLAP_DIST_RATIO:
                continue

            center = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))

            # 同一フレーム内の二重検出を除去
            if any(np.linalg.norm(np.array(center) - np.array(c)) < dedup_dist
                   for c in seen_centers):
                continue
            seen_centers.append(center)

            # 人物BBoxを取得（顔マッチングに使用）
            if boxes is not None and i < len(boxes.xyxy):
                person_bbox = tuple(boxes.xyxy[i].cpu().numpy().tolist())
            else:
                person_bbox = (center[0] - w * 0.1, 0, center[0] + w * 0.1, float(h))

            entries.append(_GestureEntry(
                timestamp=now,
                clap_center=center,
                person_bbox=person_bbox,
            ))

        return entries

    def _update_gesture_buffer(self, new_entries: list[_GestureEntry], now: float):
        """バッファに新エントリを追加し、期限切れエントリを削除する。"""
        cutoff = now - self.GESTURE_BUFFER_SEC
        self._gesture_buffer = [g for g in self._gesture_buffer if g.timestamp > cutoff]
        self._gesture_buffer.extend(new_entries)

    def _get_confirmed_clap_gestures(self, w: int) -> list[_GestureEntry]:
        """
        バッファ内のジェスチャーを近接でグループ化し、
        CLAP_CONFIRM_MIN フレーム以上検出されたものを返す。
        """
        if not self._gesture_buffer:
            return []

        group_dist = w * self.CLAP_GROUP_DIST_RATIO
        groups: list[list[_GestureEntry]] = []

        for entry in self._gesture_buffer:
            merged = False
            for group in groups:
                ref = np.array(group[-1].clap_center)
                if np.linalg.norm(np.array(entry.clap_center) - ref) < group_dist:
                    group.append(entry)
                    merged = True
                    break
            if not merged:
                groups.append([entry])

        confirmed = []
        for group in groups:
            if len(group) >= self.CLAP_CONFIRM_MIN:
                print(f"[Buffer] 拍手確定 ({len(group)}フレーム検出)", flush=True)
                confirmed.append(group[-1])  # 最新エントリを使用

        return confirmed

    def _find_face_in_person_bbox(
        self,
        person_bbox: tuple,
        tracks: list[TrackedFace],
        exclude_ids: set[int],
    ) -> TrackedFace | None:
        """YOLO の人物BBox内にある顔トラックを返す。"""
        x1, y1, x2, y2 = person_bbox
        margin = 30  # ピクセル余白

        candidates = [
            t for t in tracks
            if t.track_id not in exclude_ids
            and x1 - margin <= t.center[0] <= x2 + margin
            and y1 - margin <= t.center[1] <= y2 + margin
        ]

        if not candidates:
            return None

        # 人物BBoxの頭部付近（上部15%）に最も近い顔を返す
        head_pos = ((x1 + x2) / 2, y1 + (y2 - y1) * 0.15)
        return min(candidates, key=lambda t: _dist(t.center, head_pos))

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


def _dist(a, b) -> float:
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

    # YOLO 人物BBox
    if yolo_result.boxes is not None:
        for box in yolo_result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 0), 1)

    # YOLO 腕スケルトン
    kps = yolo_result.keypoints
    if kps is not None and kps.xy is not None:
        conf = kps.conf
        for i, person_kps in enumerate(kps.xy):
            pts = person_kps.cpu().numpy().astype(int)
            person_conf = conf[i].cpu().numpy() if conf is not None else None

            def visible(idx):
                if person_conf is not None and person_conf[idx] < 0.3:
                    return False
                return not (pts[idx][0] == 0 and pts[idx][1] == 0)

            for a, b in _ARM_CONNECTIONS:
                if visible(a) and visible(b):
                    cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)

            for idx, label in [(_KP_LEFT_WRIST, "L"), (_KP_RIGHT_WRIST, "R")]:
                if visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 10, (0, 60, 255), -1)
                    cv2.putText(out, f"wrist {label}", (pts[idx][0] + 8, pts[idx][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 255), 1)

            for idx in [_KP_LEFT_SHOULDER, _KP_RIGHT_SHOULDER,
                        _KP_LEFT_ELBOW, _KP_RIGHT_ELBOW]:
                if visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 6, (0, 200, 255), -1)

    return out
