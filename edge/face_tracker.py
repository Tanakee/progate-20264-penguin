import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

log = logging.getLogger(__name__)

# YOLOv8-pose COCO keypoint indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

_ARM_CONNECTIONS = [
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
]


@dataclass
class TrackedFace:
    track_id: int
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    body_bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass
class _GestureEntry:
    timestamp: float
    clap_center: tuple[int, int]
    person_bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    wrist_dist: float = 0.0
    left_wrist: tuple[float, float] = (0.0, 0.0)
    right_wrist: tuple[float, float] = (0.0, 0.0)


class FaceTracker:
    """
    YOLOv8-pose で顔検出・姿勢推定・拍手ジェスチャー検出を一本化。
    MediaPipe を廃止し、YOLO の nose keypoint + person bbox から顔領域を推定する。
    """

    TRACK_TIMEOUT_SEC = 1.5
    IOU_THRESHOLD = 0.3
    CLAP_DIST_RATIO = 0.45
    KP_CONF_THRESHOLD = 0.3
    CLAP_CENTER_DEDUP_RATIO = 0.1
    GESTURE_BUFFER_SEC = 0.5
    CLAP_CONFIRM_MIN = 3
    CLAP_GROUP_DIST_RATIO = 0.15
    WRIST_SPEED_THRESHOLD = 150.0
    HEAD_REGION_RATIO = 0.15
    FACE_BBOX_MARGIN_RATIO = 0.03

    def __init__(self, yolo_model: str = "yolov8s-pose.pt"):
        self._yolo = YOLO(yolo_model)
        self._tracks: list[TrackedFace] = []
        self._next_id = 0
        self._gesture_buffer: list[_GestureEntry] = []

    def process(self, frame_bgr: np.ndarray) -> tuple[list[TrackedFace], Results]:
        yolo_result: Results = self._yolo(frame_bgr, verbose=False)[0]
        h, w = frame_bgr.shape[:2]
        now = time.monotonic()

        detected = self._extract_face_bboxes(yolo_result, w, h)
        self._update_tracks(detected)
        self._expire_tracks()

        new_gestures = self._detect_clap_gestures(yolo_result, w, h, now)
        self._update_gesture_buffer(new_gestures, now)

        return self._tracks, yolo_result

    def find_clapping_faces(
        self,
        frame_shape: tuple,
        tracks: list[TrackedFace],
    ) -> list[TrackedFace]:
        """
        確定した拍手ジェスチャーに対応する顔を返す。
        フレーム中央に近い順にソートして返す（要件: 中央優先）。
        """
        if not tracks:
            return []

        h, w = frame_shape[:2]
        confirmed = self._get_confirmed_clap_gestures(w)
        if not confirmed:
            return []

        frame_center = (w / 2, h / 2)
        result: list[TrackedFace] = []
        used_ids: set[int] = set()

        for gesture in confirmed:
            face = self._find_face_in_person_bbox(gesture.person_bbox, tracks, used_ids)
            if face:
                result.append(face)
                used_ids.add(face.track_id)
            else:
                log.debug("人物BBox内に顔が見つかりませんでした")

        result.sort(key=lambda t: _dist(t.center, frame_center))
        return result

    def find_visual_clappers(
        self,
        frame_shape: tuple,
        tracks: list[TrackedFace],
        exclude_ids: set[int] | None = None,
    ) -> list[TrackedFace]:
        """音声トリガーなしで、映像のジェスチャーバッファだけで拍手者を返す。"""
        if not tracks:
            return []

        h, w = frame_shape[:2]
        confirmed = self._get_confirmed_clap_gestures(w)
        if not confirmed:
            return []

        frame_center = (w / 2, h / 2)
        result: list[TrackedFace] = []
        used_ids: set[int] = set(exclude_ids or set())

        for gesture in confirmed:
            face = self._find_face_in_person_bbox(gesture.person_bbox, tracks, used_ids)
            if face:
                result.append(face)
                used_ids.add(face.track_id)

        result.sort(key=lambda t: _dist(t.center, frame_center))
        return result

    @property
    def gesture_buffer_size(self) -> int:
        return len(self._gesture_buffer)

    # -- private --

    def _extract_face_bboxes(
        self, yolo_result: Results, w: int, h: int
    ) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]]:
        """YOLO の nose keypoint + person bbox から顔領域を推定する。
        Returns list of (face_bbox, body_bbox) tuples.
        """
        boxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]] = []
        kps = yolo_result.keypoints
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            return boxes

        conf = kps.conf
        person_boxes = yolo_result.boxes

        for i, person_kps in enumerate(kps.xy):
            nose = person_kps[KP_NOSE].cpu().numpy()
            if nose[0] == 0 and nose[1] == 0:
                continue
            if conf is not None and float(conf[i][KP_NOSE].cpu()) < self.KP_CONF_THRESHOLD:
                continue

            body_bbox = None
            if person_boxes is not None and i < len(person_boxes.xyxy):
                px1, py1, px2, py2 = person_boxes.xyxy[i].cpu().numpy()
                person_h = py2 - py1
                person_w = px2 - px1
                bx = max(0, int(px1))
                by = max(0, int(py1))
                bw = min(int(person_w), w - bx)
                bh = min(int(person_h), h - by)
                if bw > 0 and bh > 0:
                    body_bbox = (bx, by, bw, bh)
            else:
                person_h = h * 0.3
                person_w = w * 0.15

            face_h = int(person_h * 0.18)
            face_w = int(person_w * 0.35)
            margin = int(w * self.FACE_BBOX_MARGIN_RATIO)

            fx = max(0, int(nose[0]) - face_w // 2 - margin)
            fy = max(0, int(nose[1]) - face_h // 2 - margin)
            fw = min(face_w + 2 * margin, w - fx)
            fh = min(face_h + 2 * margin, h - fy)

            if fw > 0 and fh > 0:
                boxes.append(((fx, fy, fw, fh), body_bbox))

        return boxes

    def _update_tracks(self, detected: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]]):
        for face_bbox, body_bbox in detected:
            best_iou, best_track = 0.0, None
            for track in self._tracks:
                iou = _calc_iou(face_bbox, track.bbox)
                if iou > best_iou:
                    best_iou, best_track = iou, track
            if best_track and best_iou >= self.IOU_THRESHOLD:
                best_track.bbox = face_bbox
                best_track.body_bbox = body_bbox
                best_track.last_seen = time.monotonic()
            else:
                self._tracks.append(TrackedFace(track_id=self._next_id, bbox=face_bbox, body_bbox=body_bbox))
                self._next_id += 1

    def _expire_tracks(self):
        now = time.monotonic()
        self._tracks = [
            t for t in self._tracks if now - t.last_seen < self.TRACK_TIMEOUT_SEC
        ]

    def _detect_clap_gestures(
        self, yolo_result: Results, w: int, h: int, now: float
    ) -> list[_GestureEntry]:
        entries: list[_GestureEntry] = []
        kps = yolo_result.keypoints
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            return entries

        conf = kps.conf
        boxes = yolo_result.boxes
        seen_centers: list[tuple[int, int]] = []
        dedup_dist = w * self.CLAP_CENTER_DEDUP_RATIO

        for i, person_kps in enumerate(kps.xy):
            lw = person_kps[KP_LEFT_WRIST].cpu().numpy()
            rw = person_kps[KP_RIGHT_WRIST].cpu().numpy()

            if conf is not None:
                person_conf = conf[i].cpu().numpy()
                if (
                    person_conf[KP_LEFT_WRIST] < self.KP_CONF_THRESHOLD
                    or person_conf[KP_RIGHT_WRIST] < self.KP_CONF_THRESHOLD
                ):
                    continue

            if (lw[0] == 0 and lw[1] == 0) or (rw[0] == 0 and rw[1] == 0):
                continue

            if boxes is not None and i < len(boxes.xyxy):
                person_bbox = tuple(boxes.xyxy[i].cpu().numpy().tolist())
                person_w = person_bbox[2] - person_bbox[0]
            else:
                person_bbox = (
                    float(min(lw[0], rw[0]) - w * 0.05),
                    0,
                    float(max(lw[0], rw[0]) + w * 0.05),
                    float(h),
                )
                person_w = person_bbox[2] - person_bbox[0]

            dist = float(np.linalg.norm(lw - rw))
            clap_threshold = person_w * self.CLAP_DIST_RATIO
            if dist >= clap_threshold:
                continue

            center = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))

            if any(
                np.linalg.norm(np.array(center) - np.array(c)) < dedup_dist
                for c in seen_centers
            ):
                continue
            seen_centers.append(center)

            entries.append(
                _GestureEntry(
                    timestamp=now, clap_center=center, person_bbox=person_bbox,
                    wrist_dist=dist,
                    left_wrist=(float(lw[0]), float(lw[1])),
                    right_wrist=(float(rw[0]), float(rw[1])),
                )
            )

        return entries

    def _update_gesture_buffer(self, new_entries: list[_GestureEntry], now: float):
        cutoff = now - self.GESTURE_BUFFER_SEC
        self._gesture_buffer = [g for g in self._gesture_buffer if g.timestamp > cutoff]
        self._gesture_buffer.extend(new_entries)

    def _get_confirmed_clap_gestures(self, w: int) -> list[_GestureEntry]:
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

        confirmed: list[_GestureEntry] = []
        for group in groups:
            if len(group) < self.CLAP_CONFIRM_MIN:
                continue

            max_speed = 0.0
            for i in range(1, len(group)):
                dt = group[i].timestamp - group[i - 1].timestamp
                if dt <= 0:
                    continue
                for wrist in ("left_wrist", "right_wrist"):
                    prev = np.array(getattr(group[i - 1], wrist))
                    curr = np.array(getattr(group[i], wrist))
                    speed = float(np.linalg.norm(curr - prev)) / dt
                    max_speed = max(max_speed, speed)

            if max_speed < self.WRIST_SPEED_THRESHOLD:
                log.debug("低速ポーズとして棄却 (速度=%.0f px/s < 閾値=%.0f)", max_speed, self.WRIST_SPEED_THRESHOLD)
                continue

            log.info("拍手確定 (%dフレーム, 手首速度=%.0f px/s)", len(group), max_speed)
            confirmed.append(group[-1])

        return confirmed

    def _find_face_in_person_bbox(
        self,
        person_bbox: tuple,
        tracks: list[TrackedFace],
        exclude_ids: set[int],
    ) -> TrackedFace | None:
        x1, y1, x2, y2 = person_bbox
        margin_px = int((x2 - x1) * self.FACE_BBOX_MARGIN_RATIO * 10)

        candidates = [
            t
            for t in tracks
            if t.track_id not in exclude_ids
            and x1 - margin_px <= t.center[0] <= x2 + margin_px
            and y1 - margin_px <= t.center[1] <= y2 + margin_px
        ]

        if not candidates:
            return None

        head_pos = ((x1 + x2) / 2, y1 + (y2 - y1) * self.HEAD_REGION_RATIO)
        return min(candidates, key=lambda t: _dist(t.center, head_pos))

    def close(self):
        pass  # YOLO には close 不要


# -- utilities --


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


# -- debug drawing --


def draw_debug(
    frame: np.ndarray, tracks: list[TrackedFace], yolo_result: Results
) -> np.ndarray:
    out = frame.copy()

    for track in tracks:
        x, y, bw, bh = track.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"id={track.track_id}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )

    if yolo_result.boxes is not None:
        for box in yolo_result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 0), 1)

    kps = yolo_result.keypoints
    if kps is not None and kps.xy is not None:
        conf = kps.conf
        for i, person_kps in enumerate(kps.xy):
            pts = person_kps.cpu().numpy().astype(int)
            person_conf = conf[i].cpu().numpy() if conf is not None else None

            def _visible(idx: int) -> bool:
                if person_conf is not None and person_conf[idx] < FaceTracker.KP_CONF_THRESHOLD:
                    return False
                return not (pts[idx][0] == 0 and pts[idx][1] == 0)

            for a, b in _ARM_CONNECTIONS:
                if _visible(a) and _visible(b):
                    cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)

            for idx, label in [(KP_LEFT_WRIST, "L"), (KP_RIGHT_WRIST, "R")]:
                if _visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 10, (0, 60, 255), -1)
                    cv2.putText(
                        out,
                        f"wrist {label}",
                        (pts[idx][0] + 8, pts[idx][1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 60, 255),
                        1,
                    )

            for idx in [
                KP_LEFT_SHOULDER,
                KP_RIGHT_SHOULDER,
                KP_LEFT_ELBOW,
                KP_RIGHT_ELBOW,
            ]:
                if _visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 6, (0, 200, 255), -1)

    return out
