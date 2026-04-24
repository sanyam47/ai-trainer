from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from backend.core.schemas import TrainingRequest, JobResponse, JobStatusResponse
from backend.core.router import interpret_intent
from backend.db.database import engine, Base, get_db
from backend.db.models import Job
from backend.workers.tasks import run_auto_train_pipeline, run_manual_train_pipeline
from backend.api import lab_router
from backend.core.paths import FRONTEND_DIR
import os
import shutil
from uuid import uuid4

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lab_router.router)

@app.post("/interpret")
def interpret(req: TrainingRequest):
    intent = interpret_intent(req.task)
    return intent

@app.post("/auto-train", response_model=JobResponse)
def auto_train(req: TrainingRequest, db: Session = Depends(get_db)):
    intent = interpret_intent(req.task)
    intent_payload = intent.model_dump()
    intent_payload["user_prompt"] = req.task
    job = Job(intent=intent_payload)
    db.add(job)
    db.commit()
    db.refresh(job)
    run_auto_train_pipeline.delay(job.id)
    return JobResponse(job_id=job.id, status=job.status, message="Job queued successfully")

@app.post("/train/manual", response_model=JobResponse)
async def train_manual(
    task: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    intent = interpret_intent(task)
    intent_payload = intent.model_dump()
    intent_payload["user_prompt"] = task
    os.makedirs("uploads", exist_ok=True)
    dfs = []
    import pandas as pd
    for file in files:
        temp_path = os.path.join("uploads", f"part_{uuid4().hex}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            dfs.append(pd.read_csv(temp_path))
        except: continue
            
    if not dfs: raise HTTPException(status_code=400, detail="No valid CSV files uploaded")
    merged_df = pd.concat(dfs, ignore_index=True)
    final_path = os.path.join("uploads", f"manual_merged_{uuid4().hex}.csv")
    merged_df.to_csv(final_path, index=False)
    job = Job(intent=intent_payload)
    db.add(job)
    db.commit()
    db.refresh(job)
    run_manual_train_pipeline.delay(job.id, final_path)
    return JobResponse(job_id=job.id, status=job.status, message="Manual training queued")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job.id, status=job.status, intent=job.intent, model_path=job.model_path, message=job.message)

@app.get("/download_model")
def download_model(job_id: str = None, db: Session = Depends(get_db)):
    from fastapi.responses import FileResponse
    if job_id:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job and job.model_path and os.path.exists(job.model_path):
            return FileResponse(job.model_path, filename=os.path.basename(job.model_path))
    model_path = os.path.join("models", "model.pkl")
    if os.path.exists(model_path):
        return FileResponse(model_path, filename="model.pkl")
    raise HTTPException(status_code=404, detail="No trained models found")

# Serve frontend assets from project-relative path (works locally and on cloud hosts)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
