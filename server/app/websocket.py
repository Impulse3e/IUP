import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active: dict[str, set[WebSocket]] = defaultdict(set)
        self.lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self.lock:
            self.active[session_id].add(websocket)

    async def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        async with self.lock:
            self.active[session_id].discard(websocket)
            if not self.active[session_id]:
                del self.active[session_id]

    async def broadcast(self, session_id: str, message: dict[str, Any]) -> None:
        async with self.lock:
            sockets = list(self.active.get(session_id, set()))
        for socket in sockets:
            try:
                await socket.send_json(message)
            except Exception:
                await self.disconnect(session_id, socket)

    async def broadcast_all(self, message: dict[str, Any]) -> None:
        async with self.lock:
            all_sessions = list(self.active.items())
        for _, sockets in all_sessions:
            for socket in list(sockets):
                try:
                    await socket.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()
