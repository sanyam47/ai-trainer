from celery import shared_task
from backend.workers.celery_app import celery_app
from backend.db.database import SessionLocal
from backend.db.models import Job
from backend.data_fetcher import build_dataset
from backend.pipelines.text_pipeline import TextClassificationPipeline
from backend.pipelines.image_pipeline import ImageClassificationPipeline
from backend.pipelines.audio_pipeline import AudioClassificationPipeline
from backend.pipelines.regression_pipeline import NumericRegressionPipeline
import os

def get_pipeline(modality: str, task: str = "classification"):
    if task == "regression":
        return NumericRegressionPipeline()
        
    if modality == "image":
        return ImageClassificationPipeline()
    elif modality == "audio":
        return AudioClassificationPipeline()
    return TextClassificationPipeline()

@celery_app.task
def run_auto_train_pipeline(job_id: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        db.close()
        return
        
    job.status = "running"
    db.commit()
    
    try:
        intent = job.intent or {}
        modality = intent.get("modality", "text")
        task_type = intent.get("task", "classification")
        
        # 1. Fetch data (Auto mode always uses internet)
        df = build_dataset(intent.get("task", ""), intent.get("target_classes", []))
        if df.empty:
            raise ValueError("No data fetched from internet")
            
        # 2. Save temp csv
        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", f"auto_{job_id}.csv")
        df.to_csv(temp_path, index=False)
        
        # 3. Train model
        pipeline = get_pipeline(modality, task_type)
        model_path, accuracy, message = pipeline.train(temp_path, intent)
        
        if model_path is None:
            raise ValueError(f"Training failed: {message}")
            
        job.status = "completed"
        job.model_path = model_path
        job.message = f"Auto-training ({modality}) completed. Accuracy: {accuracy}"
        db.commit()
    except Exception as e:
        job.status = "failed"
        job.message = str(e)
        db.commit()
    finally:
        db.close()

@celery_app.task
def run_manual_train_pipeline(job_id: str, filepath: str):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    
    if not job:
        db.close()
        return
        
    job.status = "running"
    db.commit()
    
    try:
        intent = job.intent or {}
        modality = intent.get("modality", "text")
        task_type = intent.get("task", "classification")
        
        pipeline = get_pipeline(modality, task_type)
        model_path, accuracy, message = pipeline.train(filepath, intent)
        
        if model_path is None:
            raise ValueError(f"Training failed: {message}")
            
        job.status = "completed"
        job.model_path = model_path
        job.message = f"Manual training ({modality}) completed. Accuracy: {accuracy}"
        db.commit()
    except Exception as e:
        job.status = "failed"
        job.message = str(e)
        db.commit()
    finally:
        db.close()
