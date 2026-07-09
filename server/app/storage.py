import hashlib
import hmac
import json
from pathlib import Path

import boto3
from botocore.client import Config

from server.app.config import settings


class StorageService:
    def save_bytes(self, relative_path: str, content: bytes) -> str:
        raise NotImplementedError

    def get_path(self, relative_path: str) -> str:
        raise NotImplementedError


class LocalStorage(StorageService):
    def __init__(self, base_path: str) -> None:
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, relative_path: str, content: bytes) -> str:
        target = self.base / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    def get_path(self, relative_path: str) -> str:
        return str(self.base / relative_path)


class S3Storage(StorageService):
    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint or None,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
        )
        self.bucket = settings.s3_bucket

    def save_bytes(self, relative_path: str, content: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=relative_path, Body=content)
        return f"s3://{self.bucket}/{relative_path}"

    def get_path(self, relative_path: str) -> str:
        return f"s3://{self.bucket}/{relative_path}"


def get_storage() -> StorageService:
    if settings.storage_backend == "s3":
        return S3Storage()
    return LocalStorage(settings.storage_path)


def sign_payload(secret: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, default=str).encode()
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
