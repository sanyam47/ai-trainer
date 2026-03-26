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
import os
import shutil
from uuid import uuid4

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Trainer API")

# Add static file serving
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
app.mount("/static", StaticFiles(directory=frontend_path), name="frontend")

@app.get("/")
def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/script.js")
def read_js():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(frontend_path, "script.js"))

@app.get("/style.css")
def read_css():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(frontend_path, "style.css"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/interpret")
def interpret(req: TrainingRequest):
    intent = interpret_intent(req.task)
    return intent

@app.post("/auto-train", response_model=JobResponse)
def auto_train(req: TrainingRequest, db: Session = Depends(get_db)):
    # 1. Interpret
    intent = interpret_intent(req.task)
    
    # 2. Create Job in DB
    job = Job(intent=intent.model_dump())
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # 3. Push to queue
    run_auto_train_pipeline.delay(job.id)
    
    return JobResponse(
        job_id=job.id,
        status=job.status,
        message="Job queued successfully"
    )

@app.post("/train/manual", response_model=JobResponse)
async def train_manual(
    task: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    # 1. Interpret Intent
    intent = interpret_intent(task)
    
    # 2. Save Uploaded Files & Merge
    import pandas as pd
    dfs = []
    os.makedirs("uploads", exist_ok=True)
    
    for file in files:
        temp_path = os.path.join("uploads", f"part_{uuid4().hex}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            df = pd.read_csv(temp_path)
            dfs.append(df)
        except:
            continue
            
    if not dfs:
        raise HTTPException(status_code=400, detail="No valid CSV files uploaded")
        
    merged_df = pd.concat(dfs, ignore_index=True)
    final_path = os.path.join("uploads", f"manual_merged_{uuid4().hex}.csv")
    merged_df.to_csv(final_path, index=False)
        
    # 3. Create Job
    job = Job(intent=intent.model_dump())
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # 4. Push to Queue
    run_manual_train_pipeline.delay(job.id, final_path)
    
    return JobResponse(
        job_id=job.id,
        status=job.status,
        message=f"Manual training job ({len(files)} files merged) queued"
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        intent=job.intent,
        model_path=job.model_path,
        message=job.message
    )

@app.get("/download_model")
def download_model():
    from fastapi.responses import FileResponse
    # For now, return the most common text model path.
    # We could evolve this to take a job_id.
    model_path = os.path.join(os.getcwd(), "models", "model.pkl")
    if os.path.exists(model_path):
        return FileResponse(model_path, filename="model.pkl")
    else:
        # Fallback to image/audio paths if they exist
        for m in ["image_model.pkl", "audio_model.pkl"]:
            path = os.path.join(os.getcwd(), "models", m)
            if os.path.exists(path):
                return FileResponse(path, filename=m)
        
        raise HTTPException(status_code=404, detail="No trained models found")
