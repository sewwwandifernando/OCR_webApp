import os
import uuid
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import settings
from app.db import get_db
from app.services.ocr_service import run_ocr

router = APIRouter()

DB = Annotated[AsyncConnection, Depends(get_db)]


@router.post("/ocr")
async def run_ocr_endpoint(
    conn: DB,
    file: UploadFile = File(...),
    nic_type: str = Form(default="new"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if nic_type not in ("new", "old"):
        raise HTTPException(status_code=400, detail="nic_type must be 'new' or 'old'")

    output_dir = os.path.join(settings.storage_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "upload.png")[1] or ".png"
    tmp_path = os.path.join(output_dir, f"ocr_{uuid.uuid4().hex}{ext}")

    contents = await file.read()
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(contents)

    try:
        result = run_ocr(tmp_path, nic_type)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return result
