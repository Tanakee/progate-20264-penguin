import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

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
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

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


class ClapState(Enum):
    FAR = auto()
    CLOSE = auto()


@dataclass
class PersonClapTracker:
    track_id: int
    state: ClapState = ClapState.FAR
    prev_dist: float | None = None
    prev_threshold: float | None = None
    last_fire_time: float = 0.0
    current_dist: float | None = None


@dataclass
class ClapEvent:
    track_id: int
    clap_center: tuple[int, int]
    person_bbox: tuple[float, float, float, float]
    is_sustained: bool = False
    timestamp: float = field(default_factory=time.monotonic)


class FaceTracker:
    """
    YOLOv8-pose で顔検出・姿勢推定・拍手ジェスチャー検出を一本化。
    v3: 手首距離の閾値クロスのみで発火するシンプルなアルゴリズム。
    """

    # -- tracking --
    TRACK_TIMEOUT_SEC = 1.5
    IOU_THRESHOLD = 0.3
    FACE_BBOX_MARGIN_RATIO = 0.03
    NOSE_CONF_THRESHOLD = 0.3
    WRIST_CONF_MIN = 0.3

    # -- clap detection v3 --
    CLOSE_RATIO = 0.35
    APPROACH_SPEED_RATIO = 0.10
    DEBOUNCE_SEC = 1.0

    def __init__(self, yolo_model: str = "yolov8s-pose.pt"):
        self._yolo = YOLO(yolo_model)
        self._tracks: list[TrackedFace] = []
        self._next_id = 0
        self._person_clap_trackers: dict[int, PersonClapTracker] = {}
        self._current_clap_events: list[ClapEvent] = []

    def process(self, frame_bgr: np.ndarray) -> tuple[list[TrackedFace], Results]:
        yolo_result: Results = self._yolo(frame_bgr, verbose=False)[0]
        h, w = frame_bgr.shape[:2]
        now = time.monotonic()

        detected = self._extract_face_bboxes(yolo_result, w, h)
        self._update_tracks(detected)
        self._expire_tracks()

        self._current_clap_events = self._run_clap_detection(yolo_result, now)
        self._expire_person_clap_trackers()

        return self._tracks, yolo_result

    def find_clapping_faces(
        self,
        frame_shape: tuple,
        tracks: list[TrackedFace],
    ) -> list[TrackedFace]:
        if not tracks or not self._current_clap_events:
            return []

        h, w = frame_shape[:2]
        frame_center = (w / 2, h / 2)
        result: list[TrackedFace] = []
        used_ids: set[int] = set()

        for event in self._current_clap_events:
            if event.track_id in used_ids:
                continue
            face = next((t for t in tracks if t.track_id == event.track_id), None)
            if face:
                result.append(face)
                used_ids.add(face.track_id)

        result.sort(key=lambda t: _dist(t.center, frame_center))
        return result

    def find_visual_clappers(
        self,
        frame_shape: tuple,
        tracks: list[TrackedFace],
        exclude_ids: set[int] | None = None,
    ) -> list[TrackedFace]:
        if not tracks or not self._current_clap_events:
            return []

        h, w = frame_shape[:2]
        frame_center = (w / 2, h / 2)
        result: list[TrackedFace] = []
        used_ids: set[int] = set(exclude_ids or set())

        for event in self._current_clap_events:
            if event.track_id in used_ids:
                continue
            face = next((t for t in tracks if t.track_id == event.track_id), None)
            if face:
                result.append(face)
                used_ids.add(face.track_id)

        result.sort(key=lambda t: _dist(t.center, frame_center))
        return result

    def find_best_audio_candidate(
        self,
        tracks: list[TrackedFace],
    ) -> TrackedFace | None:
        if not tracks:
            return None
        best: TrackedFace | None = None
        best_dist = float("inf")
        for t in tracks:
            pct = self._person_clap_trackers.get(t.track_id)
            if pct is not None and pct.current_dist is not None:
                if pct.current_dist < best_dist:
                    best_dist = pct.current_dist
                    best = t
        if best is not None:
            return best
        if len(tracks) == 1:
            return tracks[0]
        return None

    @property
    def gesture_buffer_size(self) -> int:
        return len(self._person_clap_trackers)

    def is_arm_crossing(self, track_id: int) -> bool:
        return False

    # -- private --

    def _extract_face_bboxes(
        self, yolo_result: Results, w: int, h: int
    ) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]]:
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
            if conf is not None and float(conf[i][KP_NOSE].cpu()) < self.NOSE_CONF_THRESHOLD:
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
                new_id = self._next_id
                self._tracks.append(TrackedFace(track_id=new_id, bbox=face_bbox, body_bbox=body_bbox))
                self._next_id += 1

    def _expire_tracks(self):
        now = time.monotonic()
        self._tracks = [
            t for t in self._tracks if now - t.last_seen < self.TRACK_TIMEOUT_SEC
        ]

    def _run_clap_detection(
        self, yolo_result: Results, now: float,
    ) -> list[ClapEvent]:
        events: list[ClapEvent] = []
        kps = yolo_result.keypoints
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            return events

        conf = kps.conf
        boxes = yolo_result.boxes
        processed_track_ids: set[int] = set()

        for i, person_kps_t in enumerate(kps.xy):
            person_kps = person_kps_t.cpu().numpy()
            person_conf = conf[i].cpu().numpy() if conf is not None else None

            if boxes is not None and i < len(boxes.xyxy):
                person_bbox = tuple(boxes.xyxy[i].cpu().numpy().tolist())
            else:
                continue

            track_id = self._map_person_to_track(person_bbox)
            if track_id is None:
                continue

            if track_id in processed_track_ids:
                continue
            processed_track_ids.add(track_id)

            lw_conf = float(person_conf[KP_LEFT_WRIST]) if person_conf is not None else 1.0
            rw_conf = float(person_conf[KP_RIGHT_WRIST]) if person_conf is not None else 1.0
            if lw_conf < self.WRIST_CONF_MIN or rw_conf < self.WRIST_CONF_MIN:
                continue

            lw = person_kps[KP_LEFT_WRIST]
            rw = person_kps[KP_RIGHT_WRIST]
            if (lw[0] == 0 and lw[1] == 0) or (rw[0] == 0 and rw[1] == 0):
                continue

            dist = float(np.linalg.norm(lw - rw))

            x1, y1, x2, y2 = person_bbox
            bbox_width = x2 - x1
            if bbox_width < 1:
                continue
            threshold = bbox_width * self.CLOSE_RATIO

            pct = self._person_clap_trackers.get(track_id)
            if pct is None:
                pct = PersonClapTracker(track_id=track_id)
                self._person_clap_trackers[track_id] = pct

            pct.current_dist = dist

            speed = (pct.prev_dist - dist) if pct.prev_dist is not None else 0.0
            min_speed = bbox_width * self.APPROACH_SPEED_RATIO

            crossed = (
                pct.prev_dist is not None
                and pct.prev_threshold is not None
                and pct.prev_dist >= pct.prev_threshold
                and dist < threshold
                and speed >= min_speed
            )

            pct.state = ClapState.CLOSE if dist < threshold else ClapState.FAR
            pct.prev_dist = dist
            pct.prev_threshold = threshold

            if crossed and (now - pct.last_fire_time) > self.DEBOUNCE_SEC:
                pct.last_fire_time = now
                center = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))
                events.append(ClapEvent(
                    track_id=track_id,
                    clap_center=center,
                    person_bbox=person_bbox,
                    timestamp=now,
                ))
                log.info("拍手検出 track_id=%d dist=%.0f thresh=%.0f speed=%.0f min_speed=%.0f", track_id, dist, threshold, speed, min_speed)

        return events

    def _map_person_to_track(
        self, person_bbox_xyxy: tuple,
    ) -> int | None:
        x1, y1, x2, y2 = person_bbox_xyxy
        person_area = max(1.0, (x2 - x1) * (y2 - y1))
        best_iou = 0.0
        best_tid = None
        for t in self._tracks:
            bb = t.body_bbox if t.body_bbox else t.bbox
            tx, ty, tw, th = bb
            tx2, ty2 = tx + tw, ty + th
            ix1 = max(x1, tx)
            iy1 = max(y1, ty)
            ix2 = min(x2, tx2)
            iy2 = min(y2, ty2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            track_area = max(1.0, tw * th)
            iou = inter / (person_area + track_area - inter)
            if iou > best_iou:
                best_iou = iou
                best_tid = t.track_id
        return best_tid if best_iou > 0.1 else None

    def _expire_person_clap_trackers(self):
        active_ids = {t.track_id for t in self._tracks}
        stale = [tid for tid in self._person_clap_trackers if tid not in active_ids]
        for tid in stale:
            del self._person_clap_trackers[tid]

    def close(self):
        pass


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
    frame: np.ndarray,
    tracks: list[TrackedFace],
    yolo_result: Results,
    clap_trackers: dict[int, PersonClapTracker] | None = None,
) -> np.ndarray:
    out = frame.copy()

    _STATE_COLORS = {
        ClapState.FAR: (180, 180, 180),
        ClapState.CLOSE: (0, 0, 255),
    }

    for track in tracks:
        x, y, bw, bh = track.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"id={track.track_id}"

        pct = clap_trackers.get(track.track_id) if clap_trackers else None
        if pct is not None:
            dist_str = f"{pct.current_dist:.0f}" if pct.current_dist is not None else "?"
            thresh_str = f"{pct.prev_threshold:.0f}" if pct.prev_threshold is not None else "?"
            label += f" {pct.state.name} d={dist_str}/{thresh_str}"
            color = _STATE_COLORS.get(pct.state, (0, 255, 0))
        else:
            color = (0, 255, 0)

        cv2.putText(out, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

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
                if person_conf is not None and person_conf[idx] < FaceTracker.NOSE_CONF_THRESHOLD:
                    return False
                return not (pts[idx][0] == 0 and pts[idx][1] == 0)

            for a, b in _ARM_CONNECTIONS:
                if _visible(a) and _visible(b):
                    cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)

            for idx, wrist_label in [(KP_LEFT_WRIST, "L"), (KP_RIGHT_WRIST, "R")]:
                if _visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 10, (0, 60, 255), -1)
                    cv2.putText(
                        out,
                        f"wrist {wrist_label}",
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
