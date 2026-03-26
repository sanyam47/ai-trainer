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
        print(f"Error extracting intent: {e}")
        # Default fallback
        return IntentAnalysis(
            modality="text",
            task="classification",
            target_classes=["criminal", "property", "financial"]
        )
