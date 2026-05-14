from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Tesseract
    tessdata_prefix: str
    tesseract_path: str
    base_sin_model: str = "sin"

    # Storage
    storage_path: str = "./storage"

    # Database
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "nic_ocr"
    db_user: str
    db_password: str

    # CORS — comma-separated origins in .env
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
