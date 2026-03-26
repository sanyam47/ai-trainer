from sqlalchemy import Column, String, JSON
from backend.db.database import Base
import uuid

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default="queued") # queued, running, completed, failed
    intent = Column(JSON, nullable=True) # {modality, task, target_classes}
    model_path = Column(String, nullable=True)
    message = Column(String, nullable=True)
