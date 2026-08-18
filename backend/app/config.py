"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "CV Agent Backend"
    ENV: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = (
        "postgresql+psycopg://cvuser:cvpassword@localhost:5432/cvapp"
    )

    # Model
    # Relative to the process CWD: `backend/` for local `uvicorn`, `/app` in
    # the container. Both resolve to the models dir shipped beside the app.
    MODEL_PATH: str = "app/models/model.pt"
    LABELS_PATH: str = "app/models/labels.json"
    MODEL_VERSION: str = "1.0.0"

    # CORS
    CORS_ORIGINS: str = "*"

    # Uploads
    MAX_UPLOAD_MB: int = 8
    ALLOWED_IMAGE_TYPES: str = "image/jpeg,image/png,image/webp"

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # OpenAI (speech)
    OPENAI_API_KEY: str | None = None
    STT_MODEL: str = "whisper-1"
    TTS_MODEL: str = "tts-1"
    TTS_VOICE: str = "alloy"


    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_image_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_IMAGE_TYPES.split(",") if t.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
