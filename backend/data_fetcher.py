from serpapi import GoogleSearch
import pandas as pd
from openai import OpenAI
from backend.core.config import settings
from typing import List

client = OpenAI(api_key=settings.OPENAI_API_KEY)

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
    
    if modality == "image":
        # Create dummy image metadata
        data = {
            "image_path": [f"demo_{i}.jpg" for i in range(50)],
            "label": [target_classes[i % len(target_classes)] for i in range(50)]
        }
    elif modality == "audio":
        data = {
            "audio_path": [f"demo_{i}.wav" for i in range(50)],
            "label": [target_classes[i % len(target_classes)] for i in range(50)]
        }
    else:
        # Default Text
        data = {
            "text": [f"Sample text for {prompt} - {i}" for i in range(50)],
            "label": [target_classes[i % len(target_classes)] for i in range(50)]
        }
    
    return pd.DataFrame(data)