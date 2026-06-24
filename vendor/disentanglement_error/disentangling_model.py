from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, mean_absolute_error


class DisentanglingModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict_disentangling(self, x_test):
        pass

    def score(self, y_true, y_pred):
        if self.is_regression:
            return 1 - mean_absolute_error(y_true, y_pred)
        return accuracy_score(y_true, y_pred)

    @property
    def is_regression(self):
        return False
