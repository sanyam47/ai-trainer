# 🚀 AI Model Trainer: Professional Async Engine

A production-ready, multi-modal AI training workstation built with **FastAPI**, **Celery**, and **Scikit-Learn**. 

This application allows anyone—regardless of technical background—to describe an AI task in plain English and have the machine automatically fetch data, label it, and train a professional model in the background.

## ✨ Key Features

- **🧠 Smart Intent Router:** Uses LLMs (GPT-4o) to automatically extract data modality and task requirements from natural language.
- **⚡ Asynchronous Architecture:** Powered by **FastAPI** and **Celery** with a local filesystem broker. Training never blocks the web server.
- **🌐 Multi-Modal Training:**
  - **Text:** Automated internet scraping, AI-labeling, and TF-IDF classification.
  - **Image:** Feature-extraction based grayscale classification.
  - **Audio:** MFCC analysis for sound-wave signature recognition.
  - **Numeric Regression:** Multi-file merging and numerical prediction using Random Forests.
- **📁 Manual Mode:** Support for uploading multiple CSV datasets with automatic merging.
- **💹 Progress Tracking:** Real-time polling UI with live job status updates.
- **📥 Model Export:** Instant download of trained `.pkl` models for immediate deployment.

## 🛠️ Technology Stack

- **Backend:** FastAPI (Python 3.10+)
- **Task Queue:** Celery (Filesystem Broker)
- **Database:** SQLAlchemy + SQLite
- **Machine Learning:** Scikit-Learn, Librosa (Audio), PIL (Image), Pandas
- **Frontend:** Vanilla JS + CSS (Premium Dark Theme)

## 🚦 Getting Started

### 1. Requirements
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and add your keys:
```bash
OPENAI_API_KEY=your_key_here
SERP_API_KEY=your_key_here
```

### 3. Run the System (Separate Terminals)

**Start the Worker:**
```bash
python -m celery -A backend.workers.celery_app worker --loglevel=info --pool=solo --purge
```

**Start the API Server:**
```bash
python -m uvicorn backend.api.main:app --reload --port 8001
```

Access the UI at: `http://localhost:8001`

## Deploy Free on Render

This repo includes `render.yaml`, so you can deploy without keeping your PC on:

1. Push latest code to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select this repository and deploy.
4. Add secrets when prompted:
   - `OPENAI_API_KEY`
   - `SERP_API_KEY` (optional)
5. Open the generated URL and share it.

Notes:
- The web service runs FastAPI + Celery worker in one free instance for simple demos/reviews.
- Free instances may sleep after inactivity; first request can take ~30-60s.

## 🔐 Security
This project is configured to safely exclude sensitive data (`.db`, `.env`, `models/`, `uploads/`) from version control via `.gitignore`. 
