import logging
import queue
import threading
import time

import numpy as np
import pyaudio

log = logging.getLogger(__name__)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000


class ClapDetector:
    """
    PyAudio コールバックモードで拍手を検知する。

    判定ロジック (2段階):
      1. RMS が閾値を超えたチャンクのみ次へ進む (軽量)
      2. FFT でスペクトル比率を計算し、高周波成分が十分なら拍手と判定

    連続拍手対応:
      - 検知結果をキューに溜めるので、消費側が追いつくまで失われない
      - デバウンス 50ms で同一拍手の重複チャンクを除外
      - クールダウンは消費側が acknowledge() で開始（デフォルト 0.4秒）
    """

    CALIBRATION_MULTIPLIER = 3.0
    CALIBRATION_MIN_THRESHOLD = 800

    def __init__(
        self,
        threshold_rms: int = 2000,
        cooldown_sec: float = 0.4,
        spectral_ratio_threshold: float = 0.3,
    ):
        self._threshold = threshold_rms
        self._cooldown_sec = cooldown_sec
        self._spectral_ratio_threshold = spectral_ratio_threshold
        self._queue: queue.Queue[float] = queue.Queue(maxsize=32)
        self._lock = threading.Lock()
        self._last_trigger_time = 0.0
        self._cooldown_until = 0.0
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

        self._calibrating = False
        self._calibration_samples: list[float] = []

    def _audio_callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)))

        if self._calibrating:
            self._calibration_samples.append(rms)
            return (None, pyaudio.paContinue)

        if rms <= self._threshold:
            return (None, pyaudio.paContinue)

        fft_mag = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / RATE)
        total_energy = np.sum(fft_mag**2)
        if total_energy > 0:
            spectral_ratio = float(np.sum(fft_mag[freqs >= 1000] ** 2) / total_energy)
        else:
            spectral_ratio = 0.0

        if spectral_ratio <= self._spectral_ratio_threshold:
            return (None, pyaudio.paContinue)

        now = time.monotonic()
        with self._lock:
            if now < self._cooldown_until:
                return (None, pyaudio.paContinue)
            if now - self._last_trigger_time < 0.05:
                return (None, pyaudio.paContinue)
            self._last_trigger_time = now

        try:
            self._queue.put_nowait(now)
        except queue.Full:
            pass
        log.info("拍手検知 RMS=%.0f spectral=%.2f (閾値=%d)", rms, spectral_ratio, self._threshold)

        return (None, pyaudio.paContinue)

    def start(self, device_index: int | None = None) -> bool:
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK,
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
            log.info(
                "ClapDetector 開始 (threshold_rms=%d, spectral=%.2f, cooldown=%.2fs)",
                self._threshold,
                self._spectral_ratio_threshold,
                self._cooldown_sec,
            )
            return True
        except OSError as e:
            log.warning("マイクを開けませんでした: %s — 音声なしで続行します", e)
            self._pa = None
            self._stream = None
            return False

    def calibrate(self, duration_sec: float = 3.0) -> int:
        if not self._stream:
            log.warning("ストリームが開始されていないためキャリブレーションをスキップ")
            return self._threshold

        log.info("キャリブレーション開始 (%.1f秒間、静かにしてください...)", duration_sec)
        self._calibration_samples = []
        self._calibrating = True

        time.sleep(duration_sec)

        self._calibrating = False
        samples = self._calibration_samples
        self._calibration_samples = []

        if not samples:
            log.warning("キャリブレーションデータなし — デフォルト閾値を維持")
            return self._threshold

        mean_rms = float(np.mean(samples))
        max_rms = float(np.max(samples))
        std_rms = float(np.std(samples))

        new_threshold = int(max(
            mean_rms * self.CALIBRATION_MULTIPLIER,
            max_rms * 1.5,
            mean_rms + 3 * std_rms,
            self.CALIBRATION_MIN_THRESHOLD,
        ))

        self._threshold = new_threshold
        log.info(
            "キャリブレーション完了: 環境音 平均RMS=%.0f 最大RMS=%.0f 標準偏差=%.0f → 閾値=%d",
            mean_rms, max_rms, std_rms, new_threshold,
        )
        return new_threshold

    @property
    def threshold(self) -> int:
        return self._threshold

    def consume(self) -> bool:
        try:
            self._queue.get_nowait()
            return True
        except queue.Empty:
            return False

    def acknowledge(self):
        with self._lock:
            self._cooldown_until = time.monotonic() + self._cooldown_sec
            log.info("クールダウン開始 (%.1f秒)", self._cooldown_sec)

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        log.info("ClapDetector 停止")
