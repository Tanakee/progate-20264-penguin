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
    IDLE = auto()
    APPROACHING = auto()
    CONTACT = auto()
    SEPARATING = auto()


@dataclass
class PersonClapTracker:
    track_id: int
    state: ClapState = ClapState.IDLE
    smoothed_dist: float | None = None
    prev_smoothed_dist: float | None = None
    prev_velocity: float | None = None
    contact_frame_count: int = 0
    frames_since_transition: int = 0
    last_valid_kps: dict | None = None
    occluded_frames: int = 0
    distance_history: list = field(default_factory=list)
    debounce_remaining: int = 0
    sustained_mode: bool = False


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
    MediaPipe を廃止し、YOLO の nose keypoint + person bbox から顔領域を推定する。
    """

    # -- tracking --
    TRACK_TIMEOUT_SEC = 1.5
    IOU_THRESHOLD = 0.3
    HEAD_REGION_RATIO = 0.15
    FACE_BBOX_MARGIN_RATIO = 0.03
    NOSE_CONF_THRESHOLD = 0.3
    KP_CONF_THRESHOLD = 0.3

    # -- Layer 1: normalized distance + EMA --
    EMA_ALPHA = 0.35

    # -- Layer 2: state machine --
    FAR_THRESHOLD = 1.3
    CONTACT_THRESHOLD = 0.4
    APPROACH_VELOCITY_MIN = -0.05
    MIN_CONTACT_FRAMES = 2
    MAX_CONTACT_FRAMES = 8
    STATE_TIMEOUT_FRAMES = 15
    WRIST_Y_DIFF_RATIO = 0.3
    WRIST_X_MARGIN_RATIO = 0.2
    OCCLUSION_GRACE_FRAMES = 5

    # -- debounce --
    DEBOUNCE_SINGLE_FRAMES = 18
    DEBOUNCE_SUSTAINED_FRAMES = 9

    # -- Layer 3: periodicity --
    DISTANCE_HISTORY_SIZE = 90
    PERIODICITY_MIN_PEAKS = 3
    PERIODICITY_CV_MAX = 0.4
    PERIODICITY_FREQ_MIN = 2.0
    PERIODICITY_FREQ_MAX = 7.5
    PERIODICITY_PROMINENCE = 0.15
    PERIODICITY_MIN_PEAK_DISTANCE = 5

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

        if self._tracks or self._person_clap_trackers:
            track_ids = [t.track_id for t in self._tracks]
            pct_summary = {tid: pct.state.name for tid, pct in self._person_clap_trackers.items()}
            log.debug("[frame] tracks=%s clap_states=%s events=%d", track_ids, pct_summary, len(self._current_clap_events))

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
            else:
                log.debug("ClapEvent track_id=%d に対応する顔が見つかりません", event.track_id)

        result.sort(key=lambda t: _dist(t.center, frame_center))
        return result

    def find_visual_clappers(
        self,
        frame_shape: tuple,
        tracks: list[TrackedFace],
        exclude_ids: set[int] | None = None,
    ) -> list[TrackedFace]:
        """音声トリガーなしで、映像のジェスチャーバッファだけで拍手者を返す。"""
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

    @property
    def gesture_buffer_size(self) -> int:
        return len(self._person_clap_trackers)

    def is_arm_crossing(self, track_id: int) -> bool:
        pct = self._person_clap_trackers.get(track_id)
        if pct is None:
            return False
        return (
            pct.state == ClapState.CONTACT
            and pct.contact_frame_count > self.MAX_CONTACT_FRAMES
        )

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
                log.debug("[track] updated track_id=%d iou=%.2f bbox=%s", best_track.track_id, best_iou, face_bbox)
            else:
                new_id = self._next_id
                self._tracks.append(TrackedFace(track_id=new_id, bbox=face_bbox, body_bbox=body_bbox))
                self._next_id += 1
                log.info("[track] NEW track_id=%d bbox=%s body=%s", new_id, face_bbox, body_bbox)

    def _expire_tracks(self):
        now = time.monotonic()
        before = len(self._tracks)
        expired = [t for t in self._tracks if now - t.last_seen >= self.TRACK_TIMEOUT_SEC]
        for t in expired:
            log.info("[track] EXPIRED track_id=%d (unseen %.1fs)", t.track_id, now - t.last_seen)
        self._tracks = [
            t for t in self._tracks if now - t.last_seen < self.TRACK_TIMEOUT_SEC
        ]

    # -- 3-layer clap detection --

    def _run_clap_detection(
        self, yolo_result: Results, now: float,
    ) -> list[ClapEvent]:
        events: list[ClapEvent] = []
        kps = yolo_result.keypoints
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            return events

        conf = kps.conf
        boxes = yolo_result.boxes
        required_kps = [
            KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
            KP_LEFT_WRIST, KP_RIGHT_WRIST,
            KP_LEFT_HIP, KP_RIGHT_HIP,
        ]

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
                log.debug("[clap-diag] person %d: no track_id mapped", i)
                continue

            if track_id in processed_track_ids:
                log.debug("[clap-diag] person %d: track_id=%d already processed, skip", i, track_id)
                continue
            processed_track_ids.add(track_id)

            pct = self._person_clap_trackers.get(track_id)
            if pct is None:
                pct = PersonClapTracker(track_id=track_id)
                self._person_clap_trackers[track_id] = pct

            resolved = self._resolve_keypoints(pct, person_kps, person_conf, required_kps)
            if resolved is None:
                log.debug("[clap-diag] track=%d: keypoints rejected (occluded=%d)", track_id, pct.occluded_frames)
                self._reset_state(pct)
                continue

            ls = resolved[KP_LEFT_SHOULDER]
            rs = resolved[KP_RIGHT_SHOULDER]
            shoulder_width = float(np.linalg.norm(np.array(ls) - np.array(rs)))
            if shoulder_width < 1e-6:
                continue

            lh = resolved[KP_LEFT_HIP]
            rh = resolved[KP_RIGHT_HIP]
            mid_shoulder_y = (ls[1] + rs[1]) / 2
            mid_hip_y = (lh[1] + rh[1]) / 2
            torso_height = abs(mid_hip_y - mid_shoulder_y)

            lw_pos = resolved[KP_LEFT_WRIST]
            rw_pos = resolved[KP_RIGHT_WRIST]
            wrist_dist = float(np.linalg.norm(np.array(lw_pos) - np.array(rw_pos)))
            norm_dist = wrist_dist / shoulder_width

            if norm_dist > 5.0:
                log.debug("[clap-diag] track=%d: norm_dist=%.2f too large, skip", track_id, norm_dist)
                continue

            spatial_ok = self._is_valid_clap_position(
                resolved, shoulder_width, torso_height,
            )
            is_occluded = pct.occluded_frames > 0

            log.debug(
                "[clap-diag] track=%d state=%s d=%.2f sd=%.2f v=%s spatial=%s occ=%d",
                track_id, pct.state.name, norm_dist,
                pct.smoothed_dist if pct.smoothed_dist is not None else -1,
                f"{pct.prev_velocity:.4f}" if pct.prev_velocity is not None else "None",
                spatial_ok, pct.occluded_frames,
            )

            event = self._update_person_clap_state(
                pct, norm_dist, spatial_ok, is_occluded, person_bbox, now,
            )
            if event is not None:
                events.append(event)

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
        if best_tid is not None:
            log.debug("[map] person_bbox=(%.0f,%.0f,%.0f,%.0f) -> track_id=%d iou=%.3f",
                      x1, y1, x2, y2, best_tid, best_iou)
        return best_tid if best_iou > 0.1 else None

    def _resolve_keypoints(
        self,
        pct: PersonClapTracker,
        person_kps: np.ndarray,
        person_conf: np.ndarray | None,
        required: list[int],
    ) -> dict[int, tuple[float, float]] | None:
        if pct.last_valid_kps is None:
            pct.last_valid_kps = {}

        resolved: dict[int, tuple[float, float]] = {}
        any_occluded = False

        for idx in required:
            pt = person_kps[idx]
            c = float(person_conf[idx]) if person_conf is not None else 1.0
            is_zero = pt[0] == 0 and pt[1] == 0

            if c >= self.KP_CONF_THRESHOLD and not is_zero:
                resolved[idx] = (float(pt[0]), float(pt[1]))
                pct.last_valid_kps[idx] = resolved[idx]
            elif idx in pct.last_valid_kps:
                any_occluded = True
                resolved[idx] = pct.last_valid_kps[idx]
            else:
                return None

        if any_occluded:
            pct.occluded_frames += 1
            if pct.occluded_frames > self.OCCLUSION_GRACE_FRAMES:
                return None
            if pct.state == ClapState.CONTACT:
                return None
        else:
            pct.occluded_frames = 0

        return resolved

    @staticmethod
    def _is_valid_clap_position(
        kps: dict[int, tuple[float, float]],
        shoulder_width: float,
        torso_height: float,
    ) -> bool:
        lw = kps[KP_LEFT_WRIST]
        rw = kps[KP_RIGHT_WRIST]
        ls = kps[KP_LEFT_SHOULDER]
        rs = kps[KP_RIGHT_SHOULDER]
        lh = kps[KP_LEFT_HIP]
        rh = kps[KP_RIGHT_HIP]

        mid_hip_y = (lh[1] + rh[1]) / 2
        if lw[1] > mid_hip_y or rw[1] > mid_hip_y:
            return False

        if torso_height > 1e-6 and abs(lw[1] - rw[1]) > 0.3 * torso_height:
            return False

        min_x = min(ls[0], rs[0]) - 0.2 * abs(ls[0] - rs[0])
        max_x = max(ls[0], rs[0]) + 0.2 * abs(ls[0] - rs[0])
        mid_wrist_x = (lw[0] + rw[0]) / 2
        if not (min_x <= mid_wrist_x <= max_x):
            return False

        return True

    def _update_person_clap_state(
        self,
        pct: PersonClapTracker,
        norm_dist: float,
        spatial_ok: bool,
        is_occluded: bool,
        person_bbox: tuple,
        now: float,
    ) -> ClapEvent | None:
        alpha = self.EMA_ALPHA
        if pct.smoothed_dist is None:
            pct.smoothed_dist = norm_dist
        else:
            pct.smoothed_dist = alpha * norm_dist + (1 - alpha) * pct.smoothed_dist

        velocity = None
        if pct.prev_smoothed_dist is not None:
            velocity = pct.smoothed_dist - pct.prev_smoothed_dist

        acceleration = None
        if velocity is not None and pct.prev_velocity is not None:
            acceleration = velocity - pct.prev_velocity

        pct.distance_history.append(pct.smoothed_dist)
        if len(pct.distance_history) > self.DISTANCE_HISTORY_SIZE:
            pct.distance_history = pct.distance_history[-self.DISTANCE_HISTORY_SIZE:]

        pct.prev_velocity = velocity
        pct.prev_smoothed_dist = pct.smoothed_dist

        if pct.debounce_remaining > 0:
            pct.debounce_remaining -= 1

        pct.frames_since_transition += 1
        if pct.frames_since_transition > self.STATE_TIMEOUT_FRAMES:
            self._reset_state(pct)
            return None

        if is_occluded:
            return None

        if not spatial_ok and pct.state in (ClapState.IDLE, ClapState.APPROACHING):
            return None

        event = None
        sd = pct.smoothed_dist

        if pct.state == ClapState.IDLE:
            if sd < self.FAR_THRESHOLD and velocity is not None and velocity < 0:
                pct.state = ClapState.APPROACHING
                pct.frames_since_transition = 0

        elif pct.state == ClapState.APPROACHING:
            if sd > self.FAR_THRESHOLD:
                self._reset_state(pct)
            elif (
                sd < self.CONTACT_THRESHOLD
                and velocity is not None
                and velocity < self.APPROACH_VELOCITY_MIN
            ):
                pct.state = ClapState.CONTACT
                pct.contact_frame_count = 0
                pct.frames_since_transition = 0

        elif pct.state == ClapState.CONTACT:
            pct.contact_frame_count += 1
            if pct.contact_frame_count > self.MAX_CONTACT_FRAMES:
                self._reset_state(pct)
                return None
            if velocity is not None and velocity > 0:
                if pct.contact_frame_count >= self.MIN_CONTACT_FRAMES:
                    pct.state = ClapState.SEPARATING
                    pct.frames_since_transition = 0
                else:
                    self._reset_state(pct)

        elif pct.state == ClapState.SEPARATING:
            if sd > self.FAR_THRESHOLD:
                self._reset_state(pct)
                if pct.debounce_remaining <= 0:
                    is_sustained = self._check_periodicity(pct)
                    cooldown = (
                        self.DEBOUNCE_SUSTAINED_FRAMES
                        if is_sustained
                        else self.DEBOUNCE_SINGLE_FRAMES
                    )
                    pct.debounce_remaining = cooldown

                    lw = pct.last_valid_kps.get(KP_LEFT_WRIST, (0, 0))
                    rw = pct.last_valid_kps.get(KP_RIGHT_WRIST, (0, 0))
                    center = (int((lw[0] + rw[0]) / 2), int((lw[1] + rw[1]) / 2))

                    event = ClapEvent(
                        track_id=pct.track_id,
                        clap_center=center,
                        person_bbox=person_bbox,
                        is_sustained=is_sustained,
                        timestamp=now,
                    )
                    log.info(
                        "拍手検出 track_id=%d state_machine sustained=%s",
                        pct.track_id, is_sustained,
                    )
            elif velocity is not None and velocity < 0:
                pct.state = ClapState.APPROACHING
                pct.frames_since_transition = 0

        return event

    def _check_periodicity(self, pct: PersonClapTracker, fps: float = 30.0) -> bool:
        hist = pct.distance_history
        if len(hist) < 30:
            return False

        try:
            from scipy.signal import find_peaks
        except ImportError:
            return False

        signal = -np.array(hist)
        peaks, _ = find_peaks(
            signal,
            prominence=self.PERIODICITY_PROMINENCE,
            distance=self.PERIODICITY_MIN_PEAK_DISTANCE,
        )

        if len(peaks) < self.PERIODICITY_MIN_PEAKS:
            return False

        intervals = np.diff(peaks)
        mean_interval = float(np.mean(intervals))
        if mean_interval < 1e-6:
            return False

        cv = float(np.std(intervals) / mean_interval)
        freq = fps / mean_interval

        is_sustained = (
            cv < self.PERIODICITY_CV_MAX
            and self.PERIODICITY_FREQ_MIN < freq < self.PERIODICITY_FREQ_MAX
        )
        if is_sustained:
            pct.sustained_mode = True
            log.info("持続的拍手検出 freq=%.1fHz CV=%.2f peaks=%d", freq, cv, len(peaks))
        return is_sustained

    @staticmethod
    def _reset_state(pct: PersonClapTracker):
        pct.state = ClapState.IDLE
        pct.contact_frame_count = 0
        pct.frames_since_transition = 0

    def _expire_person_clap_trackers(self):
        active_ids = {t.track_id for t in self._tracks}
        stale = [tid for tid in self._person_clap_trackers if tid not in active_ids]
        for tid in stale:
            log.info("[track] GC PersonClapTracker track_id=%d (no longer tracked)", tid)
            del self._person_clap_trackers[tid]

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
    frame: np.ndarray,
    tracks: list[TrackedFace],
    yolo_result: Results,
    clap_trackers: dict[int, PersonClapTracker] | None = None,
) -> np.ndarray:
    out = frame.copy()

    _STATE_COLORS = {
        ClapState.IDLE: (180, 180, 180),
        ClapState.APPROACHING: (0, 200, 255),
        ClapState.CONTACT: (0, 0, 255),
        ClapState.SEPARATING: (0, 255, 0),
    }

    for track in tracks:
        x, y, bw, bh = track.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"id={track.track_id}"

        pct = clap_trackers.get(track.track_id) if clap_trackers else None
        if pct is not None:
            state_name = pct.state.name
            dist_str = f"{pct.smoothed_dist:.2f}" if pct.smoothed_dist is not None else "?"
            label += f" {state_name} d={dist_str}"
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
