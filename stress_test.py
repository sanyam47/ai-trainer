import requests
import time
import json

BASE = "https://ai-trainer-0upz.onrender.com"

# Prompts designed to test the limits of Gemini's understanding
INTENT_PROMPTS = [
    "I want to automatically route IT tickets to hardware, software, or network teams based on the description.",
    "Can you build a model that looks at an X-ray and tells me if there's a fracture or if it's healthy?",
    "I need an AI to estimate the resale value of used cars given their mileage and year.",
    "We need to detect if customer audio calls are angry, neutral, or happy.",
    "Categorize these social media posts into: politics, sports, entertainment, or technology."
]

LAB_PROMPTS = [
    "I changed my mind, please remove the 'technology' category and just focus on the others.",
    "This is great, but can we also add a 'finance' category to track market news?"
]

def slow_request(method, endpoint, **kwargs):
    # Wait 5 seconds to avoid Gemini 15 RPM Free Tier rate limit
    time.sleep(5)
    return requests.request(method, BASE + endpoint, **kwargs)

print("🚀 STARTING AI STRESS TEST (Running slowly to respect rate limits...)\n")

print("--- PHASE 1: INTENT & MODALITY DETECTION ---")
for i, prompt in enumerate(INTENT_PROMPTS, 1):
    print(f"\n[Test {i}] Prompt: '{prompt}'")
    try:
        r = slow_request('POST', '/interpret', json={'task': prompt})
        if r.status_code == 200:
            res = r.json()
            print(f"  ✅ Result: Modality={res.get('modality')} | Task={res.get('task')} | Classes={res.get('target_classes')}")
        else:
            print(f"  ❌ Failed (Status {r.status_code}): {r.text}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n--- PHASE 2: MODEL MANIPULATION (LAB) ---")
mock_classes = ["politics", "sports", "entertainment", "technology"]
for i, prompt in enumerate(LAB_PROMPTS, 1):
    print(f"\n[Test {i}] Model Classes: {mock_classes}")
    print(f"Instruction: '{prompt}'")
    try:
        r = slow_request('POST', '/lab/analyze', json={'model_name': 'mock.pkl', 'instruction': prompt})
        if r.status_code == 200:
            res = r.json()
            print(f"  ✅ Result: Action={res.get('action')}")
            print(f"      Add Labels: {res.get('new_labels')}")
            print(f"      Remove Labels: {res.get('remove_labels')}")
        else:
            print(f"  ❌ Failed (Status {r.status_code}): {r.text}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n🎉 STRESS TEST COMPLETE!")
