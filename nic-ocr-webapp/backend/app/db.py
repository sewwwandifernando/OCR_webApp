from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.config import settings
from app.schema import metadata


def _build_async_url() -> str:
    return (
        f"mysql+aiomysql://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        "?charset=utf8mb4"
    )


def _build_sync_url() -> str:
    return (
        f"mysql+pymysql://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        "?charset=utf8mb4"
    )


engine = create_async_engine(_build_async_url(), pool_pre_ping=True, echo=False)
sync_engine = create_engine(_build_sync_url(), pool_pre_ping=True, echo=False)


async def get_db() -> AsyncGenerator[AsyncConnection, None]:
    async with engine.connect() as conn:
        yield conn


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
