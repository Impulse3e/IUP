from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.app.api import router
from server.app.auth import hash_password
from server.app.config import settings
from server.app.database import SessionLocal, init_db
from server.app.models import User
from server.app.secret import ensure_secret_key
from server.app.watchdog import heartbeat_watchdog


def seed_admin() -> None:
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == settings.seed_admin_email).first()
        if existing:
            return
        db.add(
            User(
                email=settings.seed_admin_email,
                password_hash=hash_password(settings.seed_admin_password),
                full_name="IUP Admin",
                role="admin",
            )
        )
        db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Path("data/storage").mkdir(parents=True, exist_ok=True)
    ensure_secret_key()
    init_db()
    seed_admin()
    task = asyncio.create_task(heartbeat_watchdog())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


LAN_ORIGIN_RE = (
    r"https?://(localhost|127\.0\.0\.1|\[::1\]|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?"
)

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_origin_regex=LAN_ORIGIN_RE,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")
dashboard_dir = Path(__file__).resolve().parents[2] / "dashboard"
if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")

    @app.get("/")
    async def dashboard_index():
        return FileResponse(dashboard_dir / "index.html")

    @app.get("/student")
    async def student_portal():
        return FileResponse(dashboard_dir / "student.html")
