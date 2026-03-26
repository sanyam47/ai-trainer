from pydantic import BaseModel
from typing import List, Optional

class TrainingRequest(BaseModel):
    task: str  # User's prompt, e.g. "I want to train a text classifier for legal documents"

class IntentAnalysis(BaseModel):
    modality: str  # text, image, audio, numeric
    task: str      # classification, regression
    target_classes: Optional[List[str]] = None # Only for classification

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    intent: Optional[dict] = None
    model_path: Optional[str] = None
    message: Optional[str] = None
