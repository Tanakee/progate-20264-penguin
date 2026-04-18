import logging
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
    """

    def __init__(
        self,
        threshold_rms: int = 2000,
        cooldown_sec: float = 1.5,
        spectral_ratio_threshold: float = 0.3,
    ):
        self._threshold = threshold_rms
        self._cooldown_sec = cooldown_sec
        self._spectral_ratio_threshold = spectral_ratio_threshold
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._last_trigger_time = 0.0
        self._cooldown_until = 0.0
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2)))

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
            # 短いデバウンス（同一拍手の連続チャンクを抑制）
            if now - self._last_trigger_time < 0.1:
                return (None, pyaudio.paContinue)
            self._last_trigger_time = now
            self._event.set()
            log.info("拍手検知 RMS=%.0f spectral=%.2f", rms, spectral_ratio)

        return (None, pyaudio.paContinue)

    def start(self, device_index: int | None = None) -> bool:
        """マイクストリームを開始する。失敗時は False を返す。"""
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
                "ClapDetector 開始 (threshold_rms=%d, spectral=%.2f)",
                self._threshold,
                self._spectral_ratio_threshold,
            )
            return True
        except OSError as e:
            log.warning("マイクを開けませんでした: %s — 音声なしで続行します", e)
            self._pa = None
            self._stream = None
            return False

    def consume(self) -> bool:
        if self._event.is_set():
            self._event.clear()
            return True
        return False

    def acknowledge(self):
        """映像側で拍手確定した時に呼ぶ。ここからクールダウンを開始する。"""
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
