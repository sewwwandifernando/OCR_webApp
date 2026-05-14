import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


async def get_next_training_id(conn: AsyncConnection) -> str:
    result = await conn.execute(
        text("SELECT id FROM training_data ORDER BY id DESC LIMIT 1 FOR UPDATE")
    )
    row = result.fetchone()
    if row is None:
        return "sin_id_001"
    suffix = int(row[0].split("_")[-1])
    return f"sin_id_{suffix + 1:03d}"


async def get_next_run_id(conn: AsyncConnection) -> str:
    result = await conn.execute(
        text("SELECT id FROM training_runs ORDER BY id DESC LIMIT 1 FOR UPDATE")
    )
    row = result.fetchone()
    if row is None:
        return "run_001"
    suffix = int(row[0].split("_")[-1])
    return f"run_{suffix + 1:03d}"


def get_model_file_size(model_path: str) -> int | None:
    try:
        return os.path.getsize(model_path)
    except OSError:
        return None


def ensure_storage_dirs(storage_path: str) -> None:
    subdirs = [
        "training_data/images",
        "training_data/tiff",
        "training_data/gt",
        "training_data/box",
        "training_data/lstmf",
        "models/active",
        "models/history",
        "output",
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(storage_path, subdir), exist_ok=True)
