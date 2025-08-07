from abc import ABC, abstractmethod

class AbstractTimeSeriesModel(ABC):
    """Base type for generative time series models.
    """
    
    @abstractmethod
    def prior(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def step(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def simulate(self, *args, **kwargs):
        pass
