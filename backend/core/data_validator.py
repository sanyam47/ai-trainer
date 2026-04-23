import pandas as pd
from typing import List, Dict, Any

class DataValidator:
    def __init__(self, threshold: int = 30):
        self.threshold = threshold

    def validate_csv(self, file_path: str, target_classes: List[str], goal_description: str = "") -> Dict[str, Any]:
        """Analyzes a CSV for sample count, balance, and semantic relevance."""
        try:
            df = pd.read_csv(file_path)
            
            if 'label' not in df.columns or 'text' not in df.columns:
                return {
                    "is_valid": False,
                    "message": "CSV must have 'text' and 'label' columns.",
                    "stats": {}
                }

            counts = df['label'].value_counts().to_dict()
            warnings = []
            
            # 1. Check Relevance (Semantic Audit)
            if goal_description:
                relevance_score = self._check_relevance(df, goal_description)
                if relevance_score < 0.5:
                    warnings.append(f"CRITICAL: Data seems irrelevant to your goal ({int(relevance_score*100)}% match).")

            # 2. Check for missing classes
            for cls in target_classes:
                count = counts.get(cls, 0)
                if count == 0:
                    warnings.append(f"Class '{cls}' has ZERO samples.")
                elif count < self.threshold:
                    warnings.append(f"Class '{cls}' is thin ({count} samples).")

            is_sufficient = len(warnings) == 0
            
            return {
                "is_valid": True,
                "is_sufficient": is_sufficient,
                "samples_per_class": counts,
                "warnings": warnings,
                "relevance_score": relevance_score if goal_description else 1.0,
                "message": "Data looks good!" if is_sufficient else "Data found but has issues."
            }

        except Exception as e:
            return {"is_valid": False, "message": str(e)}

    def _check_relevance(self, df: pd.DataFrame, goal: str) -> float:
        """Heuristic/AI check if samples match the goal."""
        samples = df['text'].head(5).tolist()
        match_count = 0
        
        # Simple keyword overlap for offline, or we could call ai_label
        goal_words = set(goal.lower().split())
        for s in samples:
            sample_words = set(s.lower().split())
            if goal_words.intersection(sample_words):
                match_count += 1
        
        return match_count / len(samples) if samples else 0
