"""Comprehensive API Test Suite for AI Trainer"""
import requests
import json
import time

BASE = "http://localhost:8001"
results = []

def test(name, func):
    try:
        ok, detail = func()
        status = "PASS" if ok else "FAIL"
        results.append((name, status, detail))
        print(f"  {'✅' if ok else '❌'} {name}: {detail}")
    except Exception as e:
        results.append((name, "ERROR", str(e)))
        print(f"  💥 {name}: ERROR - {e}")

print("=" * 60)
print("AI TRAINER - COMPREHENSIVE TEST SUITE")
print("=" * 60)

# ---- TEST 1: Homepage ----
print("\n[1/9] HOMEPAGE")
def t1():
    r = requests.get(f"{BASE}/")
    has_train = "train-section" in r.text
    has_lab = "lab-section" in r.text
    has_playground = "trainTestingSection" in r.text
    has_refine = "trainRefineInstruction" in r.text
    has_lab_playground = "labTestInput" in r.text
    has_lab_file = "labTestFileInput" in r.text
    return r.status_code == 200 and has_train and has_lab and has_playground and has_refine and has_lab_playground and has_lab_file, \
        f"status={r.status_code}, train={has_train}, lab={has_lab}, playground={has_playground}, refine_loop={has_refine}, lab_playground={has_lab_playground}, lab_file_input={has_lab_file}"
test("Homepage loads with all sections", t1)

# ---- TEST 2: Interpret ----
print("\n[2/9] INTERPRET (Train AI - Analyze)")
def t2a():
    r = requests.post(f"{BASE}/interpret", json={"task": "I want to classify text as spam or ham"})
    d = r.json()
    return r.status_code == 200 and d.get("modality") == "text" and d.get("task") == "classification", f"status={r.status_code}, response={d}"
test("Text classification intent", t2a)

def t2b():
    r = requests.post(f"{BASE}/interpret", json={"task": "I want to classify images of dogs and cats"})
    d = r.json()
    return r.status_code == 200 and d.get("modality") == "image", f"status={r.status_code}, response={d}"
test("Image classification intent", t2b)

def t2c():
    r = requests.post(f"{BASE}/interpret", json={"task": "predict house prices"})
    d = r.json()
    return r.status_code == 200 and d.get("task") == "regression", f"status={r.status_code}, response={d}"
test("Regression intent", t2c)

def t2d():
    r = requests.post(f"{BASE}/interpret", json={"task": "classify audio sounds like bark or meow"})
    d = r.json()
    return r.status_code == 200 and d.get("modality") == "audio", f"status={r.status_code}, response={d}"
test("Audio classification intent", t2d)

# ---- TEST 3: Lab Models ----
print("\n[3/9] LAB MODELS LIST")
def t3():
    r = requests.get(f"{BASE}/lab/models")
    d = r.json()
    return r.status_code == 200 and isinstance(d, list) and len(d) > 0, \
        f"status={r.status_code}, count={len(d)}, models={d[:5]}"
test("List lab models", t3)

# ---- TEST 4: Lab Analyze ----
print("\n[4/9] LAB ANALYZE")
def t4a():
    r = requests.post(f"{BASE}/lab/analyze", json={
        "model_name": "model.pkl",
        "instruction": "Add a finance class to this model"
    })
    d = r.json()
    has_action = "action" in d
    has_new_labels = "new_labels" in d
    return r.status_code == 200 and has_action and has_new_labels, \
        f"status={r.status_code}, action={d.get('action')}, new_labels={d.get('new_labels')}"
test("Analyze - add class", t4a)

def t4b():
    r = requests.post(f"{BASE}/lab/analyze", json={
        "model_name": "model.pkl",
        "instruction": "Remove the spam class"
    })
    d = r.json()
    return r.status_code == 200, f"status={r.status_code}, action={d.get('action')}"
test("Analyze - remove class", t4b)

# ---- TEST 5: Lab Predict (Multimodal) ----
print("\n[5/9] LAB PREDICT (Live Testing Playground)")

# Use professor_7B_sim.pkl which matches vectorizer.pkl (55 features)
def t5a():
    r = requests.post(f"{BASE}/lab/predict", data={
        "model_name": "professor_7B_sim.pkl",
        "text": "I have a headache and need medicine"
    })
    d = r.json()
    has_prediction = "prediction" in d
    has_confidence = "confidence" in d
    has_probs = "all_probs" in d
    return r.status_code == 200 and has_prediction and has_confidence and has_probs, \
        f"status={r.status_code}, prediction={d.get('prediction')}, confidence={round(d.get('confidence', 0), 3)}, num_classes={len(d.get('all_probs', {}))}"
test("Text prediction (professor model)", t5a)

def t5b():
    r = requests.post(f"{BASE}/lab/predict", data={
        "model_name": "professor_7B_sim.pkl",
        "text": "The patient needs cardiac surgery"
    })
    d = r.json()
    return r.status_code == 200 and "prediction" in d, \
        f"status={r.status_code}, prediction={d.get('prediction')}, confidence={round(d.get('confidence', 0), 3)}"
test("Text prediction (different input)", t5b)

# Orphaned model should give a clear error, not a cryptic 500
def t5c():
    r = requests.post(f"{BASE}/lab/predict", data={
        "model_name": "model.pkl",
        "text": "test text"
    })
    d = r.json()
    # This model has no matching vectorizer, so it should fail with 500 but with a clear message
    return r.status_code == 500 and "features" in d.get("detail", ""), \
        f"status={r.status_code}, detail={d.get('detail', '')[:80]}"
test("Orphaned model gives clear error", t5c)

def t5d():
    r = requests.post(f"{BASE}/lab/predict", data={
        "model_name": "nonexistent_model.pkl",
        "text": "test"
    })
    return r.status_code == 404, f"status={r.status_code}, detail={r.json().get('detail')}"
test("Predict - missing model (404)", t5d)

def t5e():
    r = requests.post(f"{BASE}/lab/predict", data={
        "model_name": "professor_7B_sim.pkl"
    })
    return r.status_code == 400, f"status={r.status_code}, detail={r.json().get('detail')}"
test("Predict - no text or file (400)", t5e)

# ---- TEST 6: Jobs endpoint ----
print("\n[6/9] JOBS STATUS")
def t6a():
    r = requests.get(f"{BASE}/jobs/nonexistent-id")
    return r.status_code == 404, f"status={r.status_code}"
test("Job status - nonexistent (404)", t6a)

# ---- TEST 7: Download model ----
print("\n[7/9] DOWNLOAD MODEL")
def t7a():
    r = requests.get(f"{BASE}/download_model")
    return r.status_code == 200 and len(r.content) > 100, \
        f"status={r.status_code}, size={len(r.content)} bytes"
test("Download default model", t7a)

def t7b():
    r = requests.get(f"{BASE}/download_model?job_id=nonexistent")
    return r.status_code in [200, 404], f"status={r.status_code}"
test("Download with bad job_id (graceful)", t7b)

# ---- TEST 8: Static files ----
print("\n[8/9] STATIC FILES")
def t8a():
    r = requests.get(f"{BASE}/static/style.css")
    return r.status_code == 200 and len(r.text) > 50, f"status={r.status_code}, size={len(r.text)}"
test("CSS loads", t8a)

def t8b():
    r = requests.get(f"{BASE}/static/script.js")
    has_testLabModel = "testLabModel" in r.text
    has_testTrainModel = "testTrainModel" in r.text
    has_refineTrainModel = "refineTrainModel" in r.text
    has_formdata = "FormData" in r.text
    return r.status_code == 200 and has_testLabModel and has_testTrainModel and has_refineTrainModel and has_formdata, \
        f"status={r.status_code}, testLab={has_testLabModel}, testTrain={has_testTrainModel}, refine={has_refineTrainModel}, formData={has_formdata}"
test("JS loads with all new functions", t8b)

# ---- TEST 9: End-to-end auto-train trigger ----
print("\n[9/9] AUTO-TRAIN TRIGGER")
def t9():
    r = requests.post(f"{BASE}/auto-train", json={"task": "classify text as positive or negative"})
    d = r.json()
    has_job_id = "job_id" in d
    return r.status_code == 200 and has_job_id, f"status={r.status_code}, job_id={d.get('job_id', 'NONE')}"
test("Auto-train queues a job", t9)

# ---- SUMMARY ----
print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
errors = sum(1 for _, s, _ in results if s == "ERROR")
total = len(results)
print(f"RESULTS: {passed}/{total} PASSED | {failed} FAILED | {errors} ERRORS")
print("=" * 60)

if failed + errors > 0:
    print("\nFailed/Error details:")
    for name, status, detail in results:
        if status != "PASS":
            print(f"  >> {name}: {detail}")
else:
    print("\nALL TESTS PASSED! The application is fully operational.")
