from openai import OpenAI
from backend.core.config import settings
from backend.core.schemas import IntentAnalysis

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def interpret_intent(user_text: str) -> IntentAnalysis:
    """Uses OpenAI's structured outputs to extract intent from user text."""
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that determines the machine learning intent from a user's prompt. Extract the data modality (e.g., text, image), the specific task (e.g., classification), and any specific target classes mentioned or implied. Provide a comprehensive list of target classes."},
                {"role": "user", "content": user_text}
            ],
            response_format=IntentAnalysis,
            temperature=0
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Offline Mode or API Error: {e}")
        
        # --- Smart Keyword Detection (FREE!) ---
        text = user_text.lower()
        
        # 1. Modality detection
        modality = "text"
        if any(w in text for w in ["image", "picture", "photo", "dog", "cat", "animal", "face"]):
            modality = "image"
        elif any(w in text for w in ["audio", "sound", "voice", "song", "bark"]):
            modality = "audio"
        elif any(w in text for w in ["price", "demand", "number", "score", "predict"]):
            modality = "numeric"

        # 2. Task detection
        task = "classification"
        if any(w in text for w in ["price", "demand", "value", "how much"]):
            task = "regression"

        # 3. Target Class extraction (Simple list-based)
        common_classes = {
            "dog": "dog", "cat": "cat", "animal": "animal",
            "spam": "spam", "urgent": "urgent", "ham": "ham",
            "criminal": "criminal", "property": "property", "financial": "financial",
            "positive": "positive", "negative": "negative"
        }
        found_classes = [v for k, v in common_classes.items() if k in text]
        
        # Fallback if no classes found
        if not found_classes:
            found_classes = ["class_a", "class_b"]

        return IntentAnalysis(
            modality=modality,
            task=task,
            target_classes=found_classes
        )
