import requests

BASE = "https://ai-trainer-0upz.onrender.com"
passed = 0
total = 0

def test(name, fn):
    global passed, total
    total += 1
    try:
        ok, detail = fn()
        if ok:
            passed += 1
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        print(f"       {detail}")
    except Exception as e:
        print(f"  [ERR] {name}: {e}")

print("=== LIVE RENDER DEPLOYMENT TESTS ===\n")

print("[1] INTENT DETECTION (Gemini)")

def t_spam():
    r = requests.post(BASE+"/interpret", json={"task":"classify emails as spam or ham"}, timeout=30)
    d = r.json()
    ok = r.status_code == 200 and "spam" in d.get("target_classes", [])
    return ok, f"modality={d.get('modality')}, classes={d.get('target_classes')}"
test("Spam/ham detection", t_spam)

def t_sentiment():
    r = requests.post(BASE+"/interpret", json={"task":"detect if a movie review is positive negative or neutral"}, timeout=30)
    d = r.json()
    classes = d.get("target_classes", [])
    ok = r.status_code == 200 and len(classes) >= 2 and classes != ["class_a", "class_b"]
    return ok, f"classes={classes}"
test("Sentiment analysis intent", t_sentiment)

def t_image():
    r = requests.post(BASE+"/interpret", json={"task":"classify images of cats dogs and birds"}, timeout=30)
    d = r.json()
    ok = r.status_code == 200 and d.get("modality") == "image"
    return ok, f"modality={d.get('modality')}, classes={d.get('target_classes')}"
test("Image classification intent", t_image)

def t_regression():
    r = requests.post(BASE+"/interpret", json={"task":"predict house prices based on number of rooms"}, timeout=30)
    d = r.json()
    ok = r.status_code == 200 and d.get("task") == "regression"
    return ok, f"task={d.get('task')}, modality={d.get('modality')}"
test("Regression intent", t_regression)

def t_fraud():
    r = requests.post(BASE+"/interpret", json={"task":"I want to detect fraudulent bank transactions vs legitimate ones"}, timeout=30)
    d = r.json()
    classes = d.get("target_classes", [])
    ok = r.status_code == 200 and len(classes) >= 2 and classes != ["class_a", "class_b"]
    return ok, f"classes={classes}"
test("Fraud detection intent", t_fraud)

print("\n[2] LAB ANALYZE (Gemini)")

def t_add_class():
    r = requests.post(BASE+"/lab/analyze", json={
        "model_name": "model.pkl",
        "instruction": "add a phishing class to this spam detector"
    }, timeout=30)
    d = r.json()
    ok = r.status_code == 200 and d.get("action") == "ADD_CLASS"
    return ok, f"action={d.get('action')}, new_labels={d.get('new_labels')}"
test("Add class instruction", t_add_class)

def t_refine():
    r = requests.post(BASE+"/lab/analyze", json={
        "model_name": "model.pkl",
        "instruction": "improve the model accuracy"
    }, timeout=30)
    d = r.json()
    ok = r.status_code == 200 and "action" in d
    return ok, f"action={d.get('action')}, reasoning={d.get('reasoning', '')[:60]}"
test("Refine instruction", t_refine)

print("\n[3] STATIC ASSETS")

def t_css():
    r = requests.get(BASE+"/static/style.css", timeout=10)
    return r.status_code == 200, f"{len(r.text)} chars"
test("CSS stylesheet", t_css)

def t_js():
    r = requests.get(BASE+"/static/script.js", timeout=10)
    has_test = "testLabModel" in r.text
    has_refine = "refineTrainModel" in r.text
    return r.status_code == 200 and has_test and has_refine, f"testLabModel={has_test}, refineTrainModel={has_refine}"
test("JavaScript bundle", t_js)

print("\n[4] ERROR HANDLING")

def t_404():
    r = requests.post(BASE+"/lab/predict", data={"model_name": "notfound.pkl", "text": "test"}, timeout=15)
    return r.status_code == 404, f"status={r.status_code}"
test("Missing model returns 404", t_404)

def t_job_404():
    r = requests.get(BASE+"/jobs/fake-id-123", timeout=10)
    return r.status_code == 404, f"status={r.status_code}"
test("Missing job returns 404", t_job_404)

print(f"\n=== RESULTS: {passed}/{total} PASSED on live Render ===")
if passed == total:
    print("All tests passed! App is fully operational in the cloud.")
else:
    print(f"{total - passed} test(s) need attention.")
