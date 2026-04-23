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

conf = {
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "timezone": "UTC",
    "enable_utc": True,
}

# Filesystem broker needs explicit folder paths; Redis/Rabbit do not.
if settings.BROKER_URL.startswith("filesystem://"):
    conf["broker_transport_options"] = {
        "data_folder_in": "./broker/out",
        "data_folder_out": "./broker/out",
        "data_folder_processed": "./broker/processed"
    }

celery_app.conf.update(**conf)
