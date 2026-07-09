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
    cors_origins: str = "*"
    jwt_expire_minutes: int = 60 * 24
    retention_days: int = 90
    webhook_timeout: float = 5.0
    lti_client_id: str = ""
    lti_deployment_id: str = ""
    seed_admin_email: str = "admin@example.com"
    seed_admin_password: str = "admin123"


settings = Settings()
