from celery import Celery
from backend.core.config import settings
import os

# Create broker directories needed for the filesystem transport
for folder in ["./broker/out", "./broker/processed"]:
    os.makedirs(folder, exist_ok=True)

celery_app = Celery(
    "ai_trainer_tasks",
    broker=settings.BROKER_URL,
    backend=settings.RESULT_BACKEND,
    include=["backend.workers.tasks"]
)

celery_app.conf.update(
    broker_transport_options={
        "data_folder_in": "./broker/out",
        "data_folder_out": "./broker/out",
        "data_folder_processed": "./broker/processed"
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
