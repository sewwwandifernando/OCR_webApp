from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db
from app.routers import models, testing, training, training_data
from app.services.file_service import ensure_storage_dirs


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    ensure_storage_dirs(settings.storage_path)
    yield


app = FastAPI(title="NIC OCR Training System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training_data.router, prefix="/api/training-data", tags=["Training Data"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(testing.router, prefix="/api/testing", tags=["Testing"])


@app.get("/health")
async def health():
    return {"status": "ok"}
