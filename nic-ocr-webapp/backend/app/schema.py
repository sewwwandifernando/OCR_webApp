from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.sql import func

metadata = MetaData()

training_data = Table(
    "training_data",
    metadata,
    Column("id", String(20), primary_key=True),
    Column("png_path", String(500), nullable=False),
    Column("gt_path", String(500), nullable=False),
    Column("tif_path", String(500)),
    Column("box_path", String(500)),
    Column("lstmf_path", String(500)),
    Column("ground_truth", Text, nullable=False),
    Column("status_gt", String(20), nullable=False, server_default="done"),
    Column("status_tif", String(20), nullable=False, server_default="pending"),
    Column("status_box", String(20), nullable=False, server_default="pending"),
    Column("status_lstmf", String(20), nullable=False, server_default="pending"),
    Column("error_tif", Text),
    Column("error_box", Text),
    Column("error_lstmf", Text),
    Column("uploaded_at", DateTime, server_default=func.now()),
    Column("notes", String(1000), server_default=""),
)

training_runs = Table(
    "training_runs",
    metadata,
    Column("id", String(20), primary_key=True),
    Column("started_at", DateTime),
    Column("completed_at", DateTime),
    Column("status", String(20), nullable=False, server_default="queued"),
    Column("iterations", Integer, nullable=False),
    Column("file_count", Integer),
    Column("model_path", String(500)),
    Column("is_active", Boolean, nullable=False, server_default="0"),
    Column("log", Text),
)
