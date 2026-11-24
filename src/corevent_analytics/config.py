from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_url: str | None = Field(default=None, alias="DB_URL")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_user: str = Field(default="", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_name: str = Field(default="", alias="DB_NAME")
    db_ssl: bool = Field(default=False, alias="DB_SSL")
    db_ssl_ca: str | None = Field(default=None, alias="DB_SSL_CA")
    db_ssl_reject_unauthorized: bool = Field(
        default=True,
        alias="DB_SSL_REJECT_UNAUTHORIZED",
    )

    synthetic_n_users: int = Field(default=3000, alias="SYNTHETIC_N_USERS")
    synthetic_n_events: int = Field(default=200, alias="SYNTHETIC_N_EVENTS")
    synthetic_seed: int = Field(default=42, alias="SYNTHETIC_SEED")

    output_dir: Path = Field(default=ROOT_DIR / "outputs", alias="OUTPUT_DIR")
    min_support: float = Field(default=0.05, alias="MIN_SUPPORT")
    min_confidence: float = Field(default=0.3, alias="MIN_CONFIDENCE")
    kmeans_k_max: int = Field(default=8, alias="KMEANS_K_MAX")
    test_size: float = Field(default=0.25, alias="TEST_SIZE")
    random_state: int = Field(default=42, alias="RANDOM_STATE")

    def build_database_url(self) -> str | None:
        if self.db_url:
            return self.db_url
        if not self.db_user or not self.db_name:
            return None
        user = quote_plus(self.db_user)
        if self.db_password:
            auth = f"{user}:{quote_plus(self.db_password)}"
        else:
            auth = user
        return f"postgresql+psycopg://{auth}@{self.db_host}:{self.db_port}/{self.db_name}"


def get_settings() -> Settings:
    return Settings()
