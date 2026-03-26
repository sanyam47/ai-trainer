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
def build_dataset(query: str, target_classes: List[str]):
    texts = fetch_data(query)

    if not texts:
        return pd.DataFrame()

    # increase limit to have enough data points for 3 overlapping labels
    texts = texts[:30]

    data = []

    for i, t in enumerate(texts):
        label = ai_label(t, target_classes, i)
        data.append({
            "text": t,
            "label": label
        })

    return pd.DataFrame(data)