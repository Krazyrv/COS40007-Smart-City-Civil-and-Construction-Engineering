from abc import ABC, abstractmethod

class Trainer(ABC):
    """Abstract trainer interface"""
    @abstractmethod
    def train(self):
        """run trainning loop; return a result dict or object"""
        raise NotImplementedError
    
    @abstractmethod
    def validate(self):
        """run validation on held-out set; return metrics dict."""
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, sources):
        """Run inference on one or more image paths; return predictions"""
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path):
        """save model/checkpoints to path"""
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path):
        """load model/checkpoints from path"""
        raise NotImplementedError