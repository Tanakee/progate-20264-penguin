import cv2
import numpy as np


class AROverlay:
    """アルファチャンネル付きPNGを顔のBBoxに合成する。"""

    def __init__(self, asset_path: str):
        img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(
                f"アセットが見つかりません: {asset_path}\n"
                "assets/penguin.png にアルファチャンネル付きPNGを配置してください。"
            )
        if img.ndim < 3 or img.shape[2] != 4:
            raise ValueError(f"アルファチャンネル付きPNG(BGRA)が必要です: {asset_path}")
        self._asset: np.ndarray = img  # (H, W, 4) BGRA uint8

    def apply(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        frame の (x, y, w, h) 領域にアセットをアルファブレンドして返す。
        frame は変更せず新しい ndarray を返す。
        """
        if w <= 0 or h <= 0:
            return frame

        resized = cv2.resize(self._asset, (w, h), interpolation=cv2.INTER_AREA)

        # フレーム境界クリッピング
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            return frame

        # オーバーレイ側のクリップ範囲
        ox1, oy1 = x1 - x, y1 - y
        ox2, oy2 = ox1 + rw, oy1 + rh

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        src = resized[oy1:oy2, ox1:ox2, :3].astype(np.float32)
        alpha = resized[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) / 255.0

        blended = (alpha * src + (1.0 - alpha) * roi).astype(np.uint8)

        result = frame.copy()
        result[y1:y2, x1:x2] = blended
        return result
