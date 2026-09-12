from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "IUP Proctoring"
    secret_key: str = "change-me-in-production"
    database_url: str = "sqlite:///./data/iup.db"
    storage_backend: str = "local"
    storage_path: str = "./data/storage"
    s3_endpoint: str = ""
    s3_bucket: str = "iup-evidence"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    jwt_expire_minutes: int = 60 * 24
    retention_days: int = 90
    webhook_timeout: float = 5.0
    max_upload_bytes: int = 20 * 1024 * 1024
    lti_client_id: str = ""
    lti_deployment_id: str = ""
    lti_launch_secret: str = ""
    seed_admin_email: str = "admin@iup.local"
    seed_admin_password: str = "admin123"
    identity_distance_threshold: float = 0.32
    heartbeat_timeout_sec: int = 30
    heartbeat_check_interval_sec: int = 10
    cleanup_interval_sec: int = 180
    chunk_keep_webcam: int = 4
    chunk_keep_screen: int = 4


settings = Settings()
