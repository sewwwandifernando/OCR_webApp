import os
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.config import settings
from app.db import get_db
from app.schema import training_data as td_table
from app.services.file_service import get_next_training_id
from app.services.tesseract_service import generate_box_file, generate_lstmf_file

router = APIRouter()

DB = Annotated[AsyncConnection, Depends(get_db)]


def _row_to_dict(row) -> dict:
    return dict(row._mapping)


@router.get("")
async def list_training_data(conn: DB):
    result = await conn.execute(
        select(td_table).order_by(td_table.c.uploaded_at.desc())
    )
    rows = [_row_to_dict(r) for r in result.fetchall()]

    total = len(rows)
    ready = sum(
        1 for r in rows
        if r["status_tif"] == "done"
        and r["status_box"] == "done"
        and r["status_lstmf"] == "done"
    )
    errors = sum(
        1 for r in rows
        if "failed" in (r["status_tif"], r["status_box"], r["status_lstmf"])
    )

    return {"items": rows, "summary": {"total": total, "ready": ready, "errors": errors}}


@router.post("/upload")
async def upload_training_data(
    conn: DB,
    file: UploadFile = File(...),
    ground_truth: str = Form(...),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    storage = settings.storage_path

    async with conn.begin():
        new_id = await get_next_training_id(conn)

        png_path = os.path.join(storage, "training_data", "images", f"{new_id}.png")
        gt_path = os.path.join(storage, "training_data", "gt", f"{new_id}.gt.txt")

        # Save PNG
        contents = await file.read()
        async with aiofiles.open(png_path, "wb") as f:
            await f.write(contents)

        # Save ground truth (UTF-8, no BOM, no trailing newline)
        async with aiofiles.open(gt_path, "w", encoding="utf-8") as f:
            await f.write(ground_truth.strip())

        await conn.execute(
            td_table.insert().values(
                id=new_id,
                png_path=png_path,
                gt_path=gt_path,
                ground_truth=ground_truth.strip(),
                status_gt="done",
                status_tif="pending",
                status_box="pending",
                status_lstmf="pending",
            )
        )

    # --- post-commit processing (best-effort, no transaction) ---
    tif_path = os.path.join(storage, "training_data", "tiff", f"{new_id}.tif")
    box_base = os.path.join(storage, "training_data", "box", new_id)
    lstmf_base = os.path.join(storage, "training_data", "lstmf", new_id)
    box_path = f"{box_base}.box"
    lstmf_path = f"{lstmf_base}.lstmf"

    # Convert PNG → TIFF
    try:
        img = Image.open(png_path)
        img.save(tif_path, format="TIFF")
        await conn.execute(
            td_table.update()
            .where(td_table.c.id == new_id)
            .values(tif_path=tif_path, status_tif="done")
        )
    except Exception as exc:
        await conn.execute(
            td_table.update()
            .where(td_table.c.id == new_id)
            .values(status_tif="failed", error_tif=str(exc))
        )
        await conn.commit()
        result = await conn.execute(select(td_table).where(td_table.c.id == new_id))
        return _row_to_dict(result.fetchone())

    # Generate .box file
    box_ok, box_err = generate_box_file(tif_path, box_base, ground_truth.strip())
    if box_ok:
        await conn.execute(
            td_table.update()
            .where(td_table.c.id == new_id)
            .values(box_path=box_path, status_box="done")
        )
    else:
        await conn.execute(
            td_table.update()
            .where(td_table.c.id == new_id)
            .values(status_box="failed", error_box=box_err)
        )

    # Generate .lstmf file
    lstmf_ok, lstmf_err = generate_lstmf_file(
        tif_path, gt_path, box_path, lstmf_base, settings.tessdata_prefix, settings.tesseract_path
    )
    if lstmf_ok:
        await conn.execute(
            td_table.update()
            .where(td_table.c.id == new_id)
            .values(lstmf_path=lstmf_path, status_lstmf="done")
        )
    else:
        await conn.execute(
            td_table.update()
            .where(td_table.c.id == new_id)
            .values(status_lstmf="failed", error_lstmf=lstmf_err)
        )

    await conn.commit()

    result = await conn.execute(select(td_table).where(td_table.c.id == new_id))
    return _row_to_dict(result.fetchone())


@router.put("/{item_id}/ground-truth")
async def update_ground_truth(item_id: str, conn: DB, ground_truth: str = Form(...)):
    result = await conn.execute(select(td_table).where(td_table.c.id == item_id))
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")

    row = _row_to_dict(row)

    # Rewrite .gt.txt
    async with aiofiles.open(row["gt_path"], "w", encoding="utf-8") as f:
        await f.write(ground_truth.strip())

    await conn.execute(
        td_table.update()
        .where(td_table.c.id == item_id)
        .values(
            ground_truth=ground_truth.strip(),
            status_box="pending",
            status_lstmf="pending",
            error_box=None,
            error_lstmf=None,
        )
    )
    await conn.commit()

    # Regenerate box + lstmf if tif exists
    if row.get("tif_path") and os.path.exists(row["tif_path"]):
        storage = settings.storage_path
        box_base = os.path.join(storage, "training_data", "box", item_id)
        lstmf_base = os.path.join(storage, "training_data", "lstmf", item_id)
        box_path = f"{box_base}.box"
        lstmf_path = f"{lstmf_base}.lstmf"

        box_ok, box_err = generate_box_file(row["tif_path"], box_base, ground_truth.strip())
        if box_ok:
            await conn.execute(
                td_table.update()
                .where(td_table.c.id == item_id)
                .values(box_path=box_path, status_box="done", error_box=None)
            )
        else:
            await conn.execute(
                td_table.update()
                .where(td_table.c.id == item_id)
                .values(status_box="failed", error_box=box_err)
            )

        lstmf_ok, lstmf_err = generate_lstmf_file(
            row["tif_path"], row["gt_path"], box_path, lstmf_base, settings.tessdata_prefix, settings.tesseract_path
        )
        if lstmf_ok:
            await conn.execute(
                td_table.update()
                .where(td_table.c.id == item_id)
                .values(lstmf_path=lstmf_path, status_lstmf="done", error_lstmf=None)
            )
        else:
            await conn.execute(
                td_table.update()
                .where(td_table.c.id == item_id)
                .values(status_lstmf="failed", error_lstmf=lstmf_err)
            )

        await conn.commit()

    result = await conn.execute(select(td_table).where(td_table.c.id == item_id))
    return _row_to_dict(result.fetchone())


@router.delete("/{item_id}")
async def delete_training_data(item_id: str, conn: DB):
    result = await conn.execute(select(td_table).where(td_table.c.id == item_id))
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")

    row = _row_to_dict(row)

    for path_col in ("png_path", "gt_path", "tif_path", "box_path", "lstmf_path"):
        path = row.get(path_col)
        if path and os.path.exists(path):
            os.remove(path)

    await conn.execute(td_table.delete().where(td_table.c.id == item_id))
    await conn.commit()

    return {"deleted": item_id}


@router.post("/regenerate-lstmf")
async def regenerate_all_lstmf(conn: DB):
    """Regenerate all lstmf files using the current tessdata.
    Required whenever tessdata is replaced (e.g. fast → best model)."""
    result = await conn.execute(select(td_table).order_by(td_table.c.id))
    rows = [_row_to_dict(r) for r in result.fetchall()]

    results = []
    for row in rows:
        item_id = row["id"]
        tif_path = row.get("tif_path")
        gt_path = row.get("gt_path")
        ground_truth = row.get("ground_truth", "")

        if not tif_path or not os.path.exists(tif_path):
            results.append({"id": item_id, "status": "skipped", "reason": "no tif"})
            continue
        if not gt_path or not os.path.exists(gt_path):
            results.append({"id": item_id, "status": "skipped", "reason": "no gt"})
            continue

        storage = settings.storage_path
        box_base = os.path.join(storage, "training_data", "box", item_id)
        lstmf_base = os.path.join(storage, "training_data", "lstmf", item_id)
        box_path = f"{box_base}.box"
        lstmf_path = f"{lstmf_base}.lstmf"

        # Regenerate box file in WordStr format first
        box_ok, box_err = generate_box_file(tif_path, box_base, ground_truth)
        if not box_ok:
            await conn.execute(
                td_table.update()
                .where(td_table.c.id == item_id)
                .values(status_box="failed", error_box=box_err)
            )
            results.append({"id": item_id, "status": "failed", "reason": f"box: {box_err}"})
            continue

        await conn.execute(
            td_table.update()
            .where(td_table.c.id == item_id)
            .values(box_path=box_path, status_box="done", error_box=None)
        )

        lstmf_ok, lstmf_err = generate_lstmf_file(
            tif_path, gt_path, box_path, lstmf_base,
            settings.tessdata_prefix, settings.tesseract_path,
        )
        if lstmf_ok:
            await conn.execute(
                td_table.update()
                .where(td_table.c.id == item_id)
                .values(lstmf_path=lstmf_path, status_lstmf="done", error_lstmf=None)
            )
            results.append({"id": item_id, "status": "ok"})
        else:
            await conn.execute(
                td_table.update()
                .where(td_table.c.id == item_id)
                .values(status_lstmf="failed", error_lstmf=lstmf_err)
            )
            results.append({"id": item_id, "status": "failed", "reason": lstmf_err})

    await conn.commit()
    ok = sum(1 for r in results if r["status"] == "ok")
    return {"regenerated": ok, "total": len(results), "details": results}


@router.get("/{item_id}/preview")
async def preview_training_data(item_id: str, conn: DB):
    result = await conn.execute(select(td_table).where(td_table.c.id == item_id))
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")

    png_path = row._mapping["png_path"]
    if not os.path.exists(png_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(png_path, media_type="image/png")
