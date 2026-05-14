import os
import shutil
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import settings
from app.db import get_db
from app.schema import training_runs as tr_table
from app.services.file_service import get_model_file_size

router = APIRouter()

DB = Annotated[AsyncConnection, Depends(get_db)]


@router.get("")
async def list_models(conn: DB):
    result = await conn.execute(
        select(tr_table).order_by(tr_table.c.started_at.desc())
    )
    rows = []
    for r in result.fetchall():
        row = dict(r._mapping)
        row["file_size"] = get_model_file_size(row["model_path"]) if row.get("model_path") else None
        rows.append(row)
    return rows


@router.post("/{run_id}/activate")
async def activate_model(run_id: str, conn: DB):
    result = await conn.execute(select(tr_table).where(tr_table.c.id == run_id))
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    row = dict(row._mapping)
    if row["status"] != "completed":
        raise HTTPException(status_code=400, detail="Can only activate a completed run")

    model_path = row.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    # Copy to active dir and tessdata prefix
    active_dir = os.path.join(settings.storage_path, "models", "active")
    os.makedirs(active_dir, exist_ok=True)
    dest_active = os.path.join(active_dir, "sin_id.traineddata")
    dest_tess = os.path.join(settings.tessdata_prefix, "sin_id.traineddata")

    shutil.copy2(model_path, dest_active)
    shutil.copy2(model_path, dest_tess)

    await conn.execute(text("UPDATE training_runs SET is_active=FALSE"))
    await conn.execute(
        text("UPDATE training_runs SET is_active=TRUE WHERE id=:id"),
        {"id": run_id},
    )
    await conn.commit()

    return {"activated": run_id}


@router.get("/{run_id}/download")
async def download_model(run_id: str, conn: DB):
    result = await conn.execute(select(tr_table).where(tr_table.c.id == run_id))
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = row._mapping.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"sin_id_{run_id}.traineddata",
    )
