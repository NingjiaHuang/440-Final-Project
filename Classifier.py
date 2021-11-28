"""top level module for classifier abstractions
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Classifier(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]) -> None:
        """fit the classifier given a set of examples X with shape (num_examples, num_features) and labels y with shape (num_examples,).
        Args:
            X (np.ndarray): the example set with shape (num_examples, num_features)
            y (np.ndarray): the labels with shape (num_examples,)
            weights (Optional[np.ndarray]): the example weights, if necessary
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """produce a list of output labels for a set of examples X with shape (num_examples, num_features).
        Args:
            X (np.ndarray): examples for which outputs should be provided
        Returns:
            np.ndarray: the predicted outputs with shape (num_examples,)
        """
        pass