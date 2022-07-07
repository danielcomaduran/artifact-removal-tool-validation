
# Import libraries
import sys
from black import transform_line
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
sys.path.append("..") # Adds higher directory to python modules path.
from Functions import artifact_removal_tools as art


class ART():
    """
    """

    def __init__(self, random_state=15, srate=128):
        self.random_state = random_state
        self.srate = srate
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def transform(self, X):
        # [x_train, _, y_train, _] = train_test_split(X, self.y, test_size=0.9, random_state=self.random_state, stratify=self.y)
        # self.X = x_train
        # self.y = y_train

        [X, _, _] = art.remove_eyeblinks_cpu(self.X, srate=self.srate, window_length=2*self.srate)
        self.X = X
        return X

    # def fit_transform(self, X, y):
    #     self.fit(self, X, y)
    #     self.transform(self, X)
    #     return self
