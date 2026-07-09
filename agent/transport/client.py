import httpx

from shared.types import ProctorEvent


class SessionClient:
    def __init__(self, server_url: str, token: str) -> None:
        self.base = server_url.rstrip("/")
        self.token = token
        self.client = httpx.Client(timeout=30.0)

    def _url(self, path: str) -> str:
        return f"{self.base}/api/sessions/token/{self.token}{path}"

    def fetch_session(self) -> dict:
        response = self.client.get(f"{self.base}/api/sessions/token/{self.token}")
        response.raise_for_status()
        return response.json()

    def accept_consent(self, accepted: bool) -> dict:
        response = self.client.post(self._url("/consent"), json={"accepted": accepted, "policy_version": "1.0"})
        response.raise_for_status()
        return response.json()

    def submit_identity(self, embedding: list[float]) -> dict:
        response = self.client.post(self._url("/identity"), json={"embedding": embedding})
        response.raise_for_status()
        return response.json()

    def start(self) -> dict:
        response = self.client.post(self._url("/start"))
        response.raise_for_status()
        return response.json()

    def heartbeat(self, agent_version: str, status: str = "active", payload: dict | None = None) -> None:
        self.client.post(
            self._url("/heartbeat"),
            json={"agent_version": agent_version, "status": status, "payload": payload or {}},
        )

    def post_event(self, event: ProctorEvent) -> dict:
        response = self.client.post(self._url("/events"), json=event.to_dict())
        response.raise_for_status()
        return response.json()

    def upload_evidence(self, evidence_type: str, path: str, violation_id: str | None = None) -> dict:
        with open(path, "rb") as handle:
            files = {"file": (path.split("/")[-1], handle, "application/octet-stream")}
            data = {"evidence_type": evidence_type}
            if violation_id:
                data["violation_id"] = violation_id
            response = self.client.post(self._url("/evidence"), data=data, files=files)
        response.raise_for_status()
        return response.json()

    def upload_chunk(self, source: str, chunk_index: int, content: bytes) -> dict:
        files = {"file": (f"{source}_{chunk_index}.bin", content, "application/octet-stream")}
        data = {"source": source, "chunk_index": str(chunk_index)}
        response = self.client.post(self._url("/chunks"), data=data, files=files)
        response.raise_for_status()
        return response.json()

    def end(self, summary: dict) -> dict:
        response = self.client.post(self._url("/end"), json={"summary": summary})
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.client.close()
