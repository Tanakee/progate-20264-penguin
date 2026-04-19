"""
Lambda handler: S3 raw/ アップロード → Rekognition → DynamoDB → AppSync

S3イベント通知で起動される。
"""

import json
import logging
import os
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "first-penguin-clap-events")
APPSYNC_ENDPOINT = os.environ.get("APPSYNC_ENDPOINT", "")
APPSYNC_API_KEY = os.environ.get("APPSYNC_API_KEY", "")
AWS_REGION = os.environ.get("APP_REGION", os.environ.get("AWS_REGION", "ap-northeast-1"))
COMPOSED_PREFIX = os.environ.get("S3_COMPOSED_PREFIX", "composed/")

rekognition = boto3.client("rekognition", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)


def lambda_handler(event, context):
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

        if not key.startswith("raw/"):
            log.info("Skipping non-raw key: %s", key)
            continue

        log.info("Processing: s3://%s/%s", bucket, key)

        try:
            emotions = _boost_happy(_analyze_face(bucket, key))
            dominant = _get_dominant_emotion(emotions)
            event_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            composed_key = _find_composed_key(key)

            _write_dynamodb(event_id, now, key, composed_key, emotions, dominant)
            _publish_appsync(event_id, now, key, composed_key, emotions, dominant)

            log.info("Done: %s dominant=%s", event_id, dominant)
        except Exception:
            log.exception("Failed to process %s", key)
            raise


def _analyze_face(bucket: str, key: str) -> list[dict]:
    resp = rekognition.detect_faces(
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        Attributes=["ALL"],
    )
    faces = resp.get("FaceDetails", [])
    if not faces:
        log.info("No faces detected in %s", key)
        return []

    face = max(faces, key=lambda f: f["BoundingBox"]["Width"] * f["BoundingBox"]["Height"])
    return [
        {"type": e["Type"], "confidence": round(e["Confidence"], 2)}
        for e in face.get("Emotions", [])
    ]


def _boost_happy(emotions: list[dict]) -> list[dict]:
    if not emotions:
        return emotions
    boosted = []
    for e in emotions:
        conf = e["confidence"]
        if e["type"] == "HAPPY":
            conf = min(conf * 2.0 + 15, 99.99)
        elif e["type"] == "SAD":
            conf = max(conf * 0.3 - 10, 0.0)
        elif e["type"] in ("CALM", "CONFUSED"):
            conf = conf * 0.5
        boosted.append({"type": e["type"], "confidence": round(conf, 2)})
    total = sum(b["confidence"] for b in boosted)
    if total > 0:
        boosted = [
            {"type": b["type"], "confidence": round(b["confidence"] / total * 100, 2)}
            for b in boosted
        ]
    return boosted


def _get_dominant_emotion(emotions: list[dict]) -> str | None:
    if not emotions:
        return None
    return max(emotions, key=lambda e: e["confidence"])["type"]


def _write_dynamodb(
    event_id: str,
    now: datetime,
    raw_key: str,
    composed_key: str | None,
    emotions: list[dict],
    dominant: str | None,
):
    table = dynamodb.Table(DYNAMODB_TABLE)
    item = {
        "id": event_id,
        "timestamp": now.isoformat(),
        "imageUrl": f"s3://{S3_BUCKET}/{raw_key}",
        "dominantEmotion": dominant or "UNKNOWN",
    }

    if composed_key:
        item["composedImageUrl"] = f"s3://{S3_BUCKET}/{composed_key}"

    if emotions:
        item["emotions"] = [
            {"type": e["type"], "confidence": Decimal(str(e["confidence"]))}
            for e in emotions
        ]
        item["confidence"] = Decimal(
            str(max(e["confidence"] for e in emotions))
        )

    table.put_item(Item=item)
    log.info("DynamoDB: wrote event %s", event_id)


def _find_composed_key(raw_key: str) -> str | None:
    filename = raw_key.split("/")[-1]
    timestamp_part = filename.split("_")[0]
    prefix = f"{COMPOSED_PREFIX}{timestamp_part}"

    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1)
        contents = resp.get("Contents", [])
        if contents:
            return contents[0]["Key"]
    except Exception:
        log.warning("Could not find composed image for %s", raw_key)

    return None


def _publish_appsync(
    event_id: str,
    now: datetime,
    raw_key: str,
    composed_key: str | None,
    emotions: list[dict],
    dominant: str | None,
):
    if not APPSYNC_ENDPOINT:
        log.warning("APPSYNC_ENDPOINT not set, skipping publish")
        return

    image_url = f"s3://{S3_BUCKET}/{raw_key}"
    composed_url = f"s3://{S3_BUCKET}/{composed_key}" if composed_key else None

    mutation = """
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

    variables = {
        "input": {
            "id": event_id,
            "timestamp": now.isoformat(),
            "trackId": 0,
            "imageUrl": image_url,
            "composedImageUrl": composed_url,
            "emotions": emotions if emotions else None,
            "dominantEmotion": dominant,
            "confidence": max((e["confidence"] for e in emotions), default=None) if emotions else None,
        }
    }

    payload = json.dumps({"query": mutation, "variables": variables})

    req = urllib.request.Request(
        APPSYNC_ENDPOINT,
        data=payload.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": APPSYNC_API_KEY,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if "errors" in body:
                log.error("AppSync errors: %s", body["errors"])
            else:
                log.info("AppSync: published event %s", event_id)
    except Exception:
        log.exception("AppSync publish failed")
