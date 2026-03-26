from abc import ABC, abstractmethod

class ModelPipeline(ABC):
    
    @abstractmethod
    def train(self, filepath: str, intent: dict) -> tuple[str, float, str]:
        """
        Trains the model based on the dataset and parsed intent.
        Returns: (model_path, accuracy, message)
        """
        pass
