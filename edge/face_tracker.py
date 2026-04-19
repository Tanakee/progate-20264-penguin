import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

log = logging.getLogger(__name__)

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
class TrackedPerson:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    face_bbox: tuple[int, int, int, int]  # (x, y, w, h)
    keypoints: dict[int, tuple[float, float]] | None = None
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.face_bbox
        return (x + w // 2, y + h // 2)

    @property
    def body_bbox(self) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = self.bbox_xyxy
        w = int(x2 - x1)
        h = int(y2 - y1)
        if w > 0 and h > 0:
            return (int(x1), int(y1), w, h)
        return None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.face_bbox


@dataclass
class PersonClapTracker:
    track_id: int
    is_close: bool = False
    close_entered_frame: int = 0
    close_wrist_y_samples: list[float] = field(default_factory=list)
    cycle_count: int = 0
    first_cycle_time: float = 0.0
    debounce_until: float = 0.0
    frame_counter: int = 0


@dataclass
class ClapEvent:
    track_id: int
    person_bbox_xyxy: tuple[float, float, float, float]
    timestamp: float = field(default_factory=time.monotonic)


class FaceTracker:
    NOSE_CONF_THRESHOLD = 0.3
    WRIST_CONF_MIN = 0.3
    MIN_BBOX_WIDTH = 50

    CLOSE_ENTER_RATIO = 0.25
    CLOSE_EXIT_RATIO = 0.40
    CYCLES_TO_FIRE = 2
    CYCLE_WINDOW_SEC = 2.5
    MAX_CLOSE_FRAMES = 15
    DEBOUNCE_SEC = 0.8
    WRIST_Y_VARIANCE_MIN = 4.0

    FACE_HEAD_RATIO = 0.18
    FACE_WIDTH_RATIO = 0.35
    FACE_BBOX_MARGIN_RATIO = 0.03

    TRACK_TIMEOUT_SEC = 1.5

    def __init__(self, yolo_model: str = "yolov8s-pose.pt"):
        self._yolo = YOLO(yolo_model)
        self._clap_trackers: dict[int, PersonClapTracker] = {}
        self._current_clap_events: list[ClapEvent] = []
        self._tracks: dict[int, TrackedPerson] = {}

    def process(self, frame_bgr: np.ndarray) -> tuple[list[TrackedPerson], Results]:
        h, w = frame_bgr.shape[:2]
        now = time.monotonic()

        yolo_result: Results = self._yolo.track(
            frame_bgr, persist=True, verbose=False,
        )[0]

        persons = self._extract_persons(yolo_result, w, h)
        self._update_tracks(persons, now)

        kp_counts = sum(1 for p in persons if p.keypoints is not None)
        if persons:
            log.debug(
                "[pipeline] persons=%d kp_ok=%d kp_none=%d tracks=%s",
                len(persons), kp_counts, len(persons) - kp_counts,
                [p.track_id for p in persons],
            )

        self._current_clap_events = self._run_clap_detection(persons, now)
        self._expire_clap_trackers()

        tracks_list = list(self._tracks.values())
        return tracks_list, yolo_result

    def get_clap_events(self) -> list[ClapEvent]:
        return self._current_clap_events

    def find_clapping_persons(
        self,
        frame_shape: tuple,
        tracks: list[TrackedPerson],
    ) -> list[TrackedPerson]:
        if not tracks or not self._current_clap_events:
            return []

        h, w = frame_shape[:2]
        frame_center = (w / 2, h / 2)
        result: list[TrackedPerson] = []
        used_ids: set[int] = set()

        for event in self._current_clap_events:
            if event.track_id in used_ids:
                continue
            person = next((t for t in tracks if t.track_id == event.track_id), None)
            if person:
                result.append(person)
                used_ids.add(person.track_id)

        result.sort(key=lambda t: _dist(t.center, frame_center))
        return result

    # -- person extraction --

    def _extract_persons(
        self, yolo_result: Results, frame_w: int, frame_h: int,
    ) -> list[TrackedPerson]:
        persons: list[TrackedPerson] = []
        kps = yolo_result.keypoints
        boxes = yolo_result.boxes
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            return persons
        if boxes is None:
            return persons

        conf = kps.conf
        track_ids = boxes.id

        for i, person_kps_t in enumerate(kps.xy):
            if track_ids is None or i >= len(track_ids):
                continue
            tid_val = track_ids[i]
            if tid_val is None:
                continue
            track_id = int(tid_val.item())

            if i >= len(boxes.xyxy):
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < self.MIN_BBOX_WIDTH:
                continue

            person_kps = person_kps_t.cpu().numpy()
            person_conf = conf[i].cpu().numpy() if conf is not None else None

            nose = person_kps[KP_NOSE]
            nose_conf = float(person_conf[KP_NOSE]) if person_conf is not None else 1.0
            if (nose[0] == 0 and nose[1] == 0) or nose_conf < self.NOSE_CONF_THRESHOLD:
                face_bbox = self._fallback_face_bbox(x1, y1, bbox_w, bbox_h, frame_w, frame_h)
            else:
                face_bbox = self._nose_to_face_bbox(
                    nose, bbox_w, bbox_h, frame_w, frame_h,
                )

            kp_dict = self._extract_keypoints(person_kps, person_conf)

            persons.append(TrackedPerson(
                track_id=track_id,
                bbox_xyxy=(x1, y1, x2, y2),
                face_bbox=face_bbox,
                keypoints=kp_dict,
            ))

        return persons

    def _nose_to_face_bbox(
        self, nose: np.ndarray, person_w: float, person_h: float,
        frame_w: int, frame_h: int,
    ) -> tuple[int, int, int, int]:
        face_h = int(person_h * self.FACE_HEAD_RATIO)
        face_w = int(person_w * self.FACE_WIDTH_RATIO)
        margin = int(frame_w * self.FACE_BBOX_MARGIN_RATIO)
        fx = max(0, int(nose[0]) - face_w // 2 - margin)
        fy = max(0, int(nose[1]) - face_h // 2 - margin)
        fw = min(face_w + 2 * margin, frame_w - fx)
        fh = min(face_h + 2 * margin, frame_h - fy)
        return (fx, fy, fw, fh)

    @staticmethod
    def _fallback_face_bbox(
        x1: float, y1: float, bbox_w: float, bbox_h: float,
        frame_w: int, frame_h: int,
    ) -> tuple[int, int, int, int]:
        face_w = int(bbox_w * 0.35)
        face_h = int(bbox_h * 0.15)
        cx = int(x1 + bbox_w / 2)
        fy = max(0, int(y1))
        fx = max(0, cx - face_w // 2)
        fw = min(face_w, frame_w - fx)
        fh = min(face_h, frame_h - fy)
        return (fx, fy, fw, fh)

    def _extract_keypoints(
        self, person_kps: np.ndarray, person_conf: np.ndarray | None,
    ) -> dict[int, tuple[float, float]] | None:
        required = [
            KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
            KP_LEFT_WRIST, KP_RIGHT_WRIST,
            KP_LEFT_HIP, KP_RIGHT_HIP,
        ]
        result: dict[int, tuple[float, float]] = {}
        for idx in required:
            pt = person_kps[idx]
            c = float(person_conf[idx]) if person_conf is not None else 1.0
            if pt[0] == 0 and pt[1] == 0:
                log.debug("[keypoints] kp_idx=%d is (0,0), returning None", idx)
                return None
            if idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST) and c < self.WRIST_CONF_MIN:
                log.debug("[keypoints] wrist kp_idx=%d conf=%.2f < %.2f, returning None", idx, c, self.WRIST_CONF_MIN)
                return None
            result[idx] = (float(pt[0]), float(pt[1]))
        return result

    # -- tracking --

    def _update_tracks(self, persons: list[TrackedPerson], now: float):
        seen_ids: set[int] = set()
        for p in persons:
            p.last_seen = now
            self._tracks[p.track_id] = p
            seen_ids.add(p.track_id)

        stale = [
            tid for tid, t in self._tracks.items()
            if now - t.last_seen > self.TRACK_TIMEOUT_SEC
        ]
        for tid in stale:
            del self._tracks[tid]

    # -- clap detection (2-state + hysteresis + cycle count) --

    def _run_clap_detection(
        self, persons: list[TrackedPerson], now: float,
    ) -> list[ClapEvent]:
        events: list[ClapEvent] = []

        for person in persons:
            kps = person.keypoints
            if kps is None:
                continue

            x1, y1, x2, y2 = person.bbox_xyxy
            bbox_width = x2 - x1

            pct = self._clap_trackers.get(person.track_id)
            if pct is None:
                pct = PersonClapTracker(track_id=person.track_id)
                self._clap_trackers[person.track_id] = pct
            pct.frame_counter += 1

            lw = kps[KP_LEFT_WRIST]
            rw = kps[KP_RIGHT_WRIST]
            d = float(np.sqrt((lw[0] - rw[0]) ** 2 + (lw[1] - rw[1]) ** 2))

            threshold_in = bbox_width * self.CLOSE_ENTER_RATIO
            threshold_out = bbox_width * self.CLOSE_EXIT_RATIO

            log.debug(
                "[clap] track=%d state=%s d=%.0f thresh_in=%.0f thresh_out=%.0f cycles=%d",
                person.track_id,
                "CLOSE" if pct.is_close else "FAR",
                d, threshold_in, threshold_out, pct.cycle_count,
            )

            if not pct.is_close:
                # FAR → check if entering CLOSE
                if d < threshold_in:
                    if self._is_arm_crossing(kps):
                        log.debug(
                            "[clap] track=%d arm_crossing detected, skip FAR→CLOSE",
                            person.track_id,
                        )
                        continue
                    pct.is_close = True
                    pct.close_entered_frame = pct.frame_counter
                    pct.close_wrist_y_samples = [lw[1], rw[1]]
                    log.debug(
                        "[clap] track=%d FAR→CLOSE d=%.0f thresh=%.0f",
                        person.track_id, d, threshold_in,
                    )
            else:
                # CLOSE state
                pct.close_wrist_y_samples.append(lw[1])
                pct.close_wrist_y_samples.append(rw[1])

                close_duration = pct.frame_counter - pct.close_entered_frame

                if close_duration > self.MAX_CLOSE_FRAMES:
                    log.debug(
                        "[clap] track=%d CLOSE too long (%d frames), reset close state",
                        person.track_id, close_duration,
                    )
                    pct.is_close = False
                    pct.close_wrist_y_samples = []
                    continue

                if d > threshold_out:
                    # CLOSE → FAR: check if valid cycle
                    valid = True

                    if len(pct.close_wrist_y_samples) >= 2:
                        variance = float(np.var(pct.close_wrist_y_samples))
                        if variance < self.WRIST_Y_VARIANCE_MIN:
                            valid = False
                            log.debug(
                                "[clap] track=%d cycle invalid: wrist static (var=%.1f)",
                                person.track_id, variance,
                            )

                    pct.is_close = False
                    pct.close_wrist_y_samples = []

                    if valid:
                        if pct.cycle_count == 0:
                            pct.first_cycle_time = now
                        pct.cycle_count += 1
                        log.debug(
                            "[clap] track=%d valid cycle #%d",
                            person.track_id, pct.cycle_count,
                        )

                        if pct.cycle_count >= self.CYCLES_TO_FIRE:
                            elapsed = now - pct.first_cycle_time
                            if elapsed <= self.CYCLE_WINDOW_SEC and now >= pct.debounce_until:
                                events.append(ClapEvent(
                                    track_id=person.track_id,
                                    person_bbox_xyxy=person.bbox_xyxy,
                                    timestamp=now,
                                ))
                                pct.debounce_until = now + self.DEBOUNCE_SEC
                                log.info(
                                    "拍手検出 track_id=%d (%.1fms, %d cycles)",
                                    person.track_id,
                                    elapsed * 1000,
                                    pct.cycle_count,
                                )
                            self._reset_cycles(pct)

                    if pct.cycle_count > 0:
                        elapsed = now - pct.first_cycle_time
                        if elapsed > self.CYCLE_WINDOW_SEC:
                            self._reset_cycles(pct)

        return events

    @staticmethod
    def _is_arm_crossing(kps: dict[int, tuple[float, float]]) -> bool:
        lw = kps[KP_LEFT_WRIST]
        rw = kps[KP_RIGHT_WRIST]
        ls = kps[KP_LEFT_SHOULDER]
        rs = kps[KP_RIGHT_SHOULDER]
        lh = kps[KP_LEFT_HIP]
        rh = kps[KP_RIGHT_HIP]

        # check 1: wrists crossed
        if lw[0] > rs[0] or rw[0] < ls[0]:
            log.debug("[arm_cross] check1: wrists crossed lw_x=%.0f rs_x=%.0f rw_x=%.0f ls_x=%.0f", lw[0], rs[0], rw[0], ls[0])
            return True

        # check 2: wrists well below hips (margin to allow natural clap height)
        mid_hip_y = (lh[1] + rh[1]) / 2
        hip_margin = abs(mid_hip_y - (ls[1] + rs[1]) / 2) * 0.3
        if lw[1] > mid_hip_y + hip_margin or rw[1] > mid_hip_y + hip_margin:
            log.debug("[arm_cross] check2: wrists below hips lw_y=%.0f rw_y=%.0f hip_y=%.0f+margin=%.0f", lw[1], rw[1], mid_hip_y, hip_margin)
            return True

        # check 3: wrist midpoint outside shoulder range
        mid_wrist_x = (lw[0] + rw[0]) / 2
        shoulder_left = min(ls[0], rs[0])
        shoulder_right = max(ls[0], rs[0])
        shoulder_margin = (shoulder_right - shoulder_left) * 0.5
        if mid_wrist_x < shoulder_left - shoulder_margin:
            log.debug("[arm_cross] check3: wrists too far left mid_x=%.0f range=%.0f-%.0f", mid_wrist_x, shoulder_left - shoulder_margin, shoulder_right + shoulder_margin)
            return True
        if mid_wrist_x > shoulder_right + shoulder_margin:
            log.debug("[arm_cross] check3: wrists too far right mid_x=%.0f range=%.0f-%.0f", mid_wrist_x, shoulder_left - shoulder_margin, shoulder_right + shoulder_margin)
            return True

        return False

    @staticmethod
    def _reset_clap(pct: PersonClapTracker):
        pct.is_close = False
        pct.close_wrist_y_samples = []
        pct.cycle_count = 0
        pct.first_cycle_time = 0.0

    @staticmethod
    def _reset_cycles(pct: PersonClapTracker):
        pct.cycle_count = 0
        pct.first_cycle_time = 0.0

    def _expire_clap_trackers(self):
        active_ids = set(self._tracks.keys())
        stale = [tid for tid in self._clap_trackers if tid not in active_ids]
        for tid in stale:
            del self._clap_trackers[tid]

    @property
    def clap_trackers(self) -> dict[int, PersonClapTracker]:
        return self._clap_trackers

    def close(self):
        pass


# -- utilities --


def _dist(a, b) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


# -- debug drawing --


def draw_debug(
    frame: np.ndarray,
    tracks: list[TrackedPerson],
    yolo_result: Results,
    clap_trackers: dict[int, PersonClapTracker] | None = None,
) -> np.ndarray:
    out = frame.copy()

    for track in tracks:
        x, y, bw, bh = track.face_bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"id={track.track_id}"

        pct = clap_trackers.get(track.track_id) if clap_trackers else None
        if pct is not None:
            state = "CLOSE" if pct.is_close else "FAR"
            label += f" {state} c={pct.cycle_count}"
            color = (0, 0, 255) if pct.is_close else (180, 180, 180)
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
                if person_conf is not None and person_conf[idx] < 0.3:
                    return False
                return not (pts[idx][0] == 0 and pts[idx][1] == 0)

            for a, b in _ARM_CONNECTIONS:
                if _visible(a) and _visible(b):
                    cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 200, 255), 2)

            for idx, wlabel in [(KP_LEFT_WRIST, "L"), (KP_RIGHT_WRIST, "R")]:
                if _visible(idx):
                    cv2.circle(out, tuple(pts[idx]), 10, (0, 60, 255), -1)
                    cv2.putText(
                        out,
                        f"wrist {wlabel}",
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
