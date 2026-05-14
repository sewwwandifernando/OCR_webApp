import asyncio
import os
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import settings
from app.db import engine, sync_engine, get_db
from app.schema import training_data as td_table
from app.schema import training_runs as tr_table
from app.services.file_service import get_next_run_id
from app.services import training_service

router = APIRouter()

DB = Annotated[AsyncConnection, Depends(get_db)]

_LSTMF_LIST_FILENAME = "train.list"
_MIN_READY = 10


class StartTrainingRequest(BaseModel):
    iterations: int


@router.post("/start")
async def start_training(body: StartTrainingRequest, conn: DB):
    if training_service.is_training():
        raise HTTPException(status_code=409, detail="Training already in progress")

    if body.iterations < 1:
        raise HTTPException(status_code=400, detail="iterations must be >= 1")

    # Count lstmf-ready rows
    result = await conn.execute(
        select(td_table.c.lstmf_path).where(td_table.c.status_lstmf == "done")
    )
    lstmf_rows = result.fetchall()
    if len(lstmf_rows) < _MIN_READY:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {_MIN_READY} ready lstmf files, have {len(lstmf_rows)}",
        )

    # Write lstmf list file — use absolute paths so lstmtraining resolves them
    # regardless of its working directory
    list_dir = os.path.abspath(os.path.join(settings.storage_path, "output"))
    os.makedirs(list_dir, exist_ok=True)
    lstmf_list_file = os.path.join(list_dir, _LSTMF_LIST_FILENAME)
    with open(lstmf_list_file, "w", encoding="utf-8") as f:
        for (path,) in lstmf_rows:
            f.write(os.path.abspath(path) + "\n")

    run_id = await get_next_run_id(conn)
    await conn.execute(
        tr_table.insert().values(
            id=run_id,
            status="running",
            iterations=body.iterations,
            started_at=datetime.utcnow(),
        )
    )
    await conn.commit()

    training_service.start_training(run_id, body.iterations, lstmf_list_file, settings, sync_engine)

    return {"run_id": run_id, "status": "running"}


@router.get("/runs")
async def get_runs(conn: DB):
    result = await conn.execute(
        select(tr_table).order_by(tr_table.c.started_at.desc())
    )
    rows = [dict(r._mapping) for r in result.fetchall()]
    return rows


@router.get("/runs/{run_id}/logs")
async def stream_logs(run_id: str, conn: DB):
    result = await conn.execute(
        select(tr_table).where(tr_table.c.id == run_id)
    )
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    row = dict(row._mapping)

    # If this run is the active one, stream from queue
    if training_service.get_active_run_id() == run_id and training_service.is_training():
        async def _live_stream():
            loop = asyncio.get_event_loop()
            for line in training_service.get_log_lines():
                yield f"data: {line}\n\n"
                if line == "[DONE]":
                    break
                await asyncio.sleep(0)

        return StreamingResponse(_live_stream(), media_type="text/event-stream")

    # Already completed — return stored log from DB
    stored_log = row.get("log") or ""

    async def _static_stream():
        for line in stored_log.splitlines():
            yield f"data: {line}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_static_stream(), media_type="text/event-stream")


@router.get("/status")
async def get_status():
    return {
        "is_training": training_service.is_training(),
        "active_run_id": training_service.get_active_run_id(),
    }
