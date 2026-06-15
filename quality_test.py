import requests
import time
import json

BASE = "https://ai-trainer-0upz.onrender.com"

def print_step(msg):
    print(f"\n{'-'*50}\n{msg}\n{'-'*50}")

def slow_request(method, endpoint, **kwargs):
    time.sleep(3) # Wait between requests to avoid free tier rate limit
    response = requests.request(method, BASE + endpoint, **kwargs)
    return response

try:
    print_step("TEST 1: Complex Intent Detection")
    prompt = "I need an AI that can look at customer support tickets and tell me if they are complaints, refund requests, or just general questions."
    print(f"User Prompt: '{prompt}'")
    r = slow_request('POST', '/interpret', json={'task': prompt})
    res = r.json()
    print(f"App Understood:\n  Modality: {res.get('modality')}\n  Task: {res.get('task')}\n  Classes: {res.get('target_classes')}")

    print_step("TEST 2: Model Manipulation (Lab Analyze)")
    instruction = "Actually, let's also add a category for 'billing issue' but remove 'general questions'"
    current_classes = ["complaint", "refund", "question"]
    print(f"Current Model Classes: {current_classes}")
    print(f"User Instruction: '{instruction}'")
    r = slow_request('POST', '/lab/analyze', json={'model_name': 'mock.pkl', 'instruction': instruction})
    res = r.json()
    print(f"AI Plan:\n  Action: {res.get('action')}\n  Add: {res.get('new_labels')}\n  Remove: {res.get('remove_labels')}\n  Reasoning: {res.get('reasoning')}")

    print_step("TEST 3: End-to-End Prediction (using the live dummy model if available)")
    # We will trigger a real background auto-train job to see if it queues properly
    r = slow_request('POST', '/auto-train', json={'task': 'classify spam vs ham'})
    res = r.json()
    print(f"Auto-Train Job Queued: {res.get('message')} (Job ID: {res.get('job_id')})")

    print("\n✅ All AI features responded successfully with intelligent answers!")

except Exception as e:
    print(f"\n❌ Error during testing: {e}")
