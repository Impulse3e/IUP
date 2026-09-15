import threading
import time

import httpx

from agent.http_util import http_timeout, normalize_server_url
from shared.types import ProctorEvent


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    except Exception:
        pass
    return response.text or response.reason_phrase or f"HTTP {response.status_code}"


class SessionClient:
    def __init__(self, server_url: str, token: str) -> None:
        self.base = normalize_server_url(server_url)
        self.token = token
        self._lock = threading.Lock()
        self.client = httpx.Client(timeout=http_timeout("upload"))
        self._heartbeat = httpx.Client(timeout=http_timeout("heartbeat"))

    def _request(self, method: str, url: str, **kwargs):
        with self._lock:
            response = getattr(self.client, method)(url, **kwargs)
        return response

    def _request_retry(self, method: str, url: str, retries: int = 5, **kwargs):
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self._request(method, url, **kwargs)
                if response.status_code >= 500 and attempt < retries - 1:
                    time.sleep(min(8.0, 0.4 * (attempt + 1)))
                    continue
                return response
            except (httpx.TransportError, httpx.TimeoutException) as error:
                last_error = error
                time.sleep(min(8.0, 0.4 * (attempt + 1)))
        if last_error:
            raise last_error
        raise RuntimeError("Нет связи с сервером")

    def _checked(self, response: httpx.Response) -> httpx.Response:
        if response.is_success:
            return response
        raise RuntimeError(_detail(response))

    def _url(self, path: str) -> str:
        return f"{self.base}/api/sessions/token/{self.token}{path}"

    def fetch_session(self) -> dict:
        response = self._checked(self._request_retry("get", f"{self.base}/api/sessions/token/{self.token}"))
        return response.json()

    def accept_consent(self, accepted: bool) -> dict:
        response = self._checked(
            self._request(
                "post",
                self._url("/consent"),
                json={"accepted": accepted, "policy_version": "1.0"},
            )
        )
        return response.json()

    def submit_identity(self, embedding: list[float]) -> dict:
        response = self._checked(self._request("post", self._url("/identity"), json={"embedding": embedding}))
        return response.json()

    def start(self) -> dict:
        response = self._checked(self._request("post", self._url("/start")))
        return response.json()

    def heartbeat(self, agent_version: str, status: str = "active", payload: dict | None = None) -> None:
        try:
            self._heartbeat.post(
                self._url("/heartbeat"),
                json={"agent_version": agent_version, "status": status, "payload": payload or {}},
            )
        except Exception:
            pass

    def post_event(self, event: ProctorEvent) -> dict:
        response = self._checked(self._request_retry("post", self._url("/events"), json=event.to_dict()))
        return response.json()

    def upload_evidence(self, evidence_type: str, path: str, violation_id: str | None = None) -> dict:
        with open(path, "rb") as handle:
            content = handle.read()
        files = {"file": (path.split("/")[-1], content, "application/octet-stream")}
        data = {"evidence_type": evidence_type}
        if violation_id:
            data["violation_id"] = violation_id
        response = self._checked(self._request_retry("post", self._url("/evidence"), data=data, files=files))
        return response.json()

    def upload_evidence_bytes(
        self,
        evidence_type: str,
        content: bytes,
        filename: str,
        violation_id: str | None = None,
        mime: str = "image/jpeg",
    ) -> dict:
        files = {"file": (filename, content, mime)}
        data = {"evidence_type": evidence_type}
        if violation_id:
            data["violation_id"] = violation_id
        response = self._checked(self._request_retry("post", self._url("/evidence"), data=data, files=files))
        return response.json()

    def upload_chunk(self, source: str, chunk_index: int, content: bytes) -> dict:
        files = {"file": (f"{source}_{chunk_index}.bin", content, "application/octet-stream")}
        data = {"source": source, "chunk_index": str(chunk_index)}
        response = self._checked(self._request_retry("post", self._url("/chunks"), data=data, files=files))
        return response.json()

    def end(self, summary: dict) -> dict:
        response = self._checked(self._request("post", self._url("/end"), json={"summary": summary}))
        return response.json()

    def close(self) -> None:
        self.client.close()
        self._heartbeat.close()
