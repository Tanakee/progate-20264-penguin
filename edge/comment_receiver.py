"""
AppSync WebSocket でコメントをリアルタイム受信する。
"""

import base64
import json
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlparse

import websocket

log = logging.getLogger(__name__)

SUBSCRIPTION_QUERY = json.dumps({
    "query": """subscription OnComment {
        onComment {
            id
            text
            color
            timestamp
        }
    }"""
})


@dataclass
class Comment:
    text: str
    color: str
    x: float
    y: int
    speed: float
    font_scale: float = 1.2


class CommentReceiver:
    def __init__(self, endpoint: str, api_key: str, max_comments: int = 30, frame_width: int = 1920):
        self._endpoint = endpoint
        self._api_key = api_key
        self._enabled = bool(endpoint and api_key)
        self._comments: deque[Comment] = deque(maxlen=max_comments)
        self._lock = threading.Lock()
        self._ws = None
        self._thread = None
        self._frame_width = frame_width

    def start(self):
        if not self._enabled:
            log.warning("AppSync 未設定 — コメント受信は無効です")
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get_comments(self) -> list[Comment]:
        with self._lock:
            return list(self._comments)

    def update_comments(self, updated: list[Comment]):
        with self._lock:
            self._comments.clear()
            for c in updated:
                self._comments.append(c)

    def stop(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _run(self):
        host = urlparse(self._endpoint).hostname
        header_data = base64.b64encode(json.dumps({
            "host": host,
            "x-api-key": self._api_key,
        }).encode()).decode()
        payload = base64.b64encode(b"{}").decode()

        realtime_url = self._endpoint.replace("https://", "wss://").replace(
            "appsync-api", "appsync-realtime-api"
        )
        url = f"{realtime_url}?header={urlencode({'': header_data})[1:]}&payload={payload}"

        self._ws = websocket.WebSocketApp(
            url,
            subprotocols=["graphql-ws"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever()

    def _on_open(self, ws):
        ws.send(json.dumps({"type": "connection_init"}))

    def _on_message(self, ws, message):
        data = json.loads(message)
        msg_type = data.get("type")

        if msg_type == "connection_ack":
            sub_id = str(uuid.uuid4())
            host = urlparse(self._endpoint).hostname
            ws.send(json.dumps({
                "id": sub_id,
                "type": "start",
                "payload": {
                    "data": SUBSCRIPTION_QUERY,
                    "extensions": {
                        "authorization": {
                            "host": host,
                            "x-api-key": self._api_key,
                        }
                    }
                }
            }))
        elif msg_type == "start_ack":
            log.info("コメント受信サブスクリプション接続完了")
        elif msg_type == "data":
            log.info("コメント data 受信: %s", json.dumps(data.get("payload", {}), ensure_ascii=False))
            comment_data = data.get("payload", {}).get("data", {}).get("onComment")
            if comment_data:
                self._add_comment(comment_data)
        elif msg_type == "error":
            log.error("コメント受信エラー: %s", json.dumps(data.get("payload", {}), ensure_ascii=False))
        elif msg_type == "ka":
            pass
        else:
            log.info("コメント WS 不明メッセージ: %s %s", msg_type, json.dumps(data, ensure_ascii=False))

    def _on_error(self, ws, error):
        log.error("コメント WebSocket エラー: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        log.info("コメント WebSocket 切断")

    def _add_comment(self, data: dict):
        import random
        y_slot = random.randint(30, 680)
        speed = random.uniform(7.0, 12.0)
        comment = Comment(
            text=data.get("text", ""),
            color=data.get("color", "#FFFFFF"),
            x=float(self._frame_width + 50),
            y=y_slot,
            speed=speed,
        )
        with self._lock:
            self._comments.append(comment)
        log.info("コメント受信: %s", comment.text)
