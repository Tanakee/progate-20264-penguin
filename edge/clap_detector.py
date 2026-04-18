import threading
import time

import numpy as np
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000


class ClapDetector:
    """
    PyAudio のコールバックモードでマイク音圧を監視し、
    拍手を検知したらフラグを立てる。
    RMS（音量）とスペクトル比率（高周波成分の割合）の両方を判定することで、
    拍手以外の音（声・BGM）による誤検知を低減する。
    メインループとは別スレッドで動作する。
    """

    def __init__(
        self,
        threshold_rms: int = 2000,
        cooldown_sec: float = 1.5,
        spectral_ratio_threshold: float = 0.3,
    ):
        """
        Args:
            threshold_rms: 拍手と判定する RMS 閾値（int16スケール: 0〜32767）
                           静かな室内での拍手は概ね 1500〜3000 程度。
                           .env の CLAP_THRESHOLD_RMS で調整すること。
            cooldown_sec:  連続検知を防ぐクールダウン秒数
            spectral_ratio_threshold: 1kHz 以上の高周波エネルギー比率の下限。
                           拍手は広帯域のため 0.3 前後。声・BGMは低め。
        """
        self._threshold = threshold_rms
        self._cooldown_sec = cooldown_sec
        self._spectral_ratio_threshold = spectral_ratio_threshold
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._last_clap_time = 0.0
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 内部スレッドから呼ばれるコールバック。重い処理は禁止。"""
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))

        # スペクトル解析: 1kHz以上の高周波エネルギー比率を計算
        fft_mag = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / RATE)
        total_energy = np.sum(fft_mag ** 2)
        if total_energy > 0:
            high_freq_energy = np.sum(fft_mag[freqs >= 1000] ** 2)
            spectral_ratio = high_freq_energy / total_energy
        else:
            spectral_ratio = 0.0

        print(f"[Audio] RMS={rms:.0f} spectral={spectral_ratio:.2f} "
              f"(threshold RMS={self._threshold} spectral={self._spectral_ratio_threshold})", flush=True)

        now = time.monotonic()
        with self._lock:
            if (rms > self._threshold
                    and spectral_ratio > self._spectral_ratio_threshold
                    and (now - self._last_clap_time) > self._cooldown_sec):
                self._last_clap_time = now
                self._event.set()
                print(f"[Audio] 拍手検知! RMS={rms:.0f} spectral={spectral_ratio:.2f}", flush=True)
        return (None, pyaudio.paContinue)

    def start(self, device_index: int | None = None):
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
        print(f"[ClapDetector] 監視開始 (threshold_rms={self._threshold} "
              f"spectral_ratio={self._spectral_ratio_threshold})")

    def consume(self) -> bool:
        """拍手フラグを読み取ってリセットする。メインループから毎フレーム呼ぶ。"""
        if self._event.is_set():
            self._event.clear()
            return True
        return False

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        print("[ClapDetector] 停止")
