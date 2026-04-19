"""
AppSync にリアルタイム通知を送信する。

Lambda 経由の通知は Rekognition 処理分の遅延があるため、
エッジから直接 Mutation を送ることでダッシュボードの即時更新を実現する。
Lambda 側で感情データが追加された場合は上書きされる。
"""

import json
import logging
import urllib.request
import uuid
from datetime import datetime, timezone

log = logging.getLogger(__name__)

PUBLISH_MUTATION = """
mutation PublishClapEvent($input: ClapEventInput!) {
    publishClapEvent(input: $input) {
        id
        timestamp
        trackId
        imageUrl
        composedImageUrl
        emotions { type confidence }
        dominantEmotion
        confidence
    }
}
"""


class AppSyncNotifier:
    def __init__(self, endpoint: str, api_key: str):
        self._endpoint = endpoint
        self._api_key = api_key
        self._enabled = bool(endpoint and api_key)
        if not self._enabled:
            log.warning("AppSync 未設定 — リアルタイム通知は無効です")

    def notify(self, track_id: int, image_url: str, composed_url: str | None = None):
        if not self._enabled:
            return

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        variables = {
            "input": {
                "id": event_id,
                "timestamp": now,
                "trackId": track_id,
                "imageUrl": image_url,
                "composedImageUrl": composed_url,
            }
        }

        self._send_mutation(variables)
        log.info("AppSync 通知送信: %s", event_id)

    def send_summary(self):
        if not self._enabled:
            return

        now = datetime.now(timezone.utc).isoformat()
        variables = {
            "input": {
                "id": "__summary__",
                "timestamp": now,
                "trackId": 0,
                "imageUrl": "",
            }
        }

        self._send_mutation(variables)
        log.info("AppSync まとめ画面シグナル送信")

    def _send_mutation(self, variables: dict):
        payload = json.dumps({"query": PUBLISH_MUTATION, "variables": variables}).encode()

        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                if "errors" in body:
                    log.error("AppSync errors: %s", body["errors"])
        except Exception:
            log.exception("AppSync 送信失敗")
