from serpapi import GoogleSearch
import pandas as pd
from openai import OpenAI
from backend.core.config import settings
from typing import List

client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.USE_OPENAI else None

# ==========================
# FETCH DATA FROM INTERNET
# ==========================
def fetch_data(query: str):
    params = {
        "engine": "google",
        "q": query,
        "api_key": settings.SERP_API_KEY,
        "num": 50
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    texts = []

    for res in results.get("organic_results", []):
        snippet = res.get("snippet")
        if snippet and len(snippet) > 30:
            texts.append(snippet)

    return texts

# ==========================
# FALLBACK LABELING
# ==========================
def simple_label(text: str, target_classes: List[str], fallback_index: int = 0) -> str:
    text_lower = text.lower()

    for cls in target_classes:
        if cls.lower() in text_lower:
            return cls.lower()

    return target_classes[fallback_index % len(target_classes)] if target_classes else "general"

# ==========================
# AI LABELING (WITH FALLBACK)
# ==========================
def ai_label(text: str, target_classes: List[str], fallback_index: int = 0) -> str:
    try:
        if client is None:
            raise RuntimeError("OpenAI disabled by configuration")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""
Classify this text into ONE label.

Possible labels:
{', '.join(target_classes)}

Rules:
- Return ONLY one word
- No explanation

Text:
{text}
"""
            }],
            temperature=0
        )

        label = response.choices[0].message.content.strip().lower()

        # safety check
        valid_targets = [c.lower() for c in target_classes]
        if label not in valid_targets:
            return simple_label(text, target_classes, fallback_index)

        return label

    except Exception as e:
        print("Label error:", e)
        return simple_label(text, target_classes, fallback_index)

# ==========================
# BUILD DATASET
# ==========================
def build_dataset(prompt: str, target_classes: list, modality: str = "text"):
    """
    Creates a dummy dataset for training based on the modality.
    """
    import pandas as pd
    
    safe_classes = [str(c).strip() for c in (target_classes or []) if str(c).strip()]
    if not safe_classes:
        safe_classes = ["class_a", "class_b"]

    if modality == "image":
        data = {
            "image_path": [f"demo_{i}.jpg" for i in range(50)],
            "label": [safe_classes[i % len(safe_classes)] for i in range(50)]
        }
        return pd.DataFrame(data)

    elif modality == "audio":
        data = {
            "audio_path": [f"demo_{i}.wav" for i in range(50)],
            "label": [safe_classes[i % len(safe_classes)] for i in range(50)]
        }
        return pd.DataFrame(data)

    else:
        # Text - try Gemini for realistic synthetic data
        try:
            from backend.core.gemini_client import generate_training_data_gemini, gemini_available
            if gemini_available():
                print(f"Using Gemini to generate training data for: {safe_classes}")
                samples = generate_training_data_gemini(prompt, safe_classes, samples_per_class=25)
                if samples:
                    return pd.DataFrame(samples)
        except Exception as e:
            print(f"Gemini data generation failed, using fallback: {e}")

        # Fallback: better dummy data with class-specific phrases
        rows = []
        templates = {
            "spam": ["Win a free {cls} now!", "Urgent: claim your {cls} prize", "Click here for {cls} deal"],
            "ham": ["Hey, are you free today?", "Meeting at 3pm", "Thanks for your help"],
            "positive": ["I love this product!", "Amazing experience", "Highly recommend"],
            "negative": ["Terrible service", "Very disappointed", "Would not recommend"],
        }
        for i in range(50):
            cls = safe_classes[i % len(safe_classes)]
            tmpl = templates.get(cls, [f"This is a {cls} example text sample"])
            text = tmpl[i % len(tmpl)].replace("{cls}", cls)
            rows.append({"text": text, "label": cls})
        return pd.DataFrame(rows)