from backend.core.config import settings
from backend.core.schemas import IntentAnalysis

def interpret_intent(user_text: str) -> IntentAnalysis:
    """
    Interprets ML intent from user text.
    Priority: Gemini (free) -> OpenAI (if enabled) -> Offline keyword fallback
    """
    # --- 1. Try Gemini (Free, no credit card) ---
    try:
        from backend.core.gemini_client import interpret_intent_gemini, gemini_available
        if gemini_available():
            result = interpret_intent_gemini(user_text)
            return IntentAnalysis(**result)
    except Exception as e:
        print(f"Gemini intent failed, trying fallback: {e}")

    # --- 2. Try OpenAI (if enabled) ---
    try:
        if settings.USE_OPENAI and settings.OPENAI_API_KEY != "FILL_ME_IN":
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that determines the machine learning intent from a user's prompt. Extract the data modality (e.g., text, image), the specific task (e.g., classification), and any specific target classes mentioned or implied."},
                    {"role": "user", "content": user_text}
                ],
                response_format=IntentAnalysis,
                temperature=0
            )
            return completion.choices[0].message.parsed
    except Exception as e:
        print(f"OpenAI intent failed: {e}")

    # --- 3. Smart Offline Keyword Fallback ---
    print("Using offline keyword fallback for intent detection")
    text = user_text.lower()

    # Modality detection
    modality = "text"
    if any(w in text for w in ["image", "picture", "photo", "dog", "cat", "animal", "face", "object", "visual", "detect object"]):
        modality = "image"
    elif any(w in text for w in ["audio", "sound", "voice", "song", "bark", "music", "speech"]):
        modality = "audio"
    elif any(w in text for w in ["price", "demand", "number", "score", "predict value", "forecast", "amount"]):
        modality = "numeric"

    # Task detection
    task = "classification"
    if any(w in text for w in ["price", "demand", "value", "how much", "forecast", "amount", "regression"]):
        task = "regression"

    # Smart class extraction - look for "X vs Y", "X or Y", "X and Y" patterns
    import re
    found_classes = []

    # Pattern: "between X and Y", "X vs Y", "X or Y"
    vs_pattern = re.findall(r'\b(\w+)\s+(?:vs|versus|or|and)\s+(\w+)\b', text)
    for pair in vs_pattern:
        for w in pair:
            if len(w) > 2 and w not in {"the", "and", "for", "its", "this", "that"}:
                found_classes.append(w)

    # Pattern: "classify as X" or "detect X"
    classify_pattern = re.findall(r'(?:classify|detect|identify|recognize)\s+(?:as\s+)?(\w+)', text)
    found_classes.extend(classify_pattern)

    # Common domain classes
    domain_classes = {
        "spam": "spam", "ham": "ham", "phishing": "phishing",
        "positive": "positive", "negative": "negative", "neutral": "neutral",
        "dog": "dog", "cat": "cat", "bird": "bird", "lion": "lion",
        "criminal": "criminal", "financial": "financial", "property": "property",
        "fake": "fake", "real": "real", "fraud": "fraud", "legit": "legit",
        "happy": "happy", "sad": "sad", "angry": "angry",
        "cancer": "cancer", "benign": "benign", "malignant": "malignant",
    }
    for k, v in domain_classes.items():
        if k in text and v not in found_classes:
            found_classes.append(v)

    # Deduplicate and clean
    seen = set()
    clean_classes = []
    for c in found_classes:
        if c not in seen and len(c) > 1:
            seen.add(c)
            clean_classes.append(c)

    if not clean_classes:
        clean_classes = ["class_a", "class_b"]

    return IntentAnalysis(
        modality=modality,
        task=task,
        target_classes=clean_classes
    )
