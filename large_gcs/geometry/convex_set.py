from abc import ABC, abstractmethod

class ConvexSet(ABC):
    """
    Abstract wrapper class for convex sets. Base implementations will be drake implementations,
    but this allows for other data and methods to the convex set as well.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    def plot(self, **kwargs):
        """
        Plots the convex set using matplotlib.
        """
        if self.dimension != 2:
            raise NotImplementedError
        options = {'facecolor':'mintcream', 'edgecolor':'k', 'zorder':1}
        options.update(kwargs)
        self._plot(**options)
    
    @property
    def dimension(self):
        return self._dimension