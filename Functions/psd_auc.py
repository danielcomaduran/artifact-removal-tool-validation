import numpy as np

class PSD_AUC():
    """
    """

    def __init__(self):
      
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def transform(self, X):
        # [x_train, _, y_train, _] = train_test_split(X, self.y, test_size=0.9, random_state=self.random_state, stratify=self.y)
        # self.X = x_train
        # self.y = y_train

        # Average over frequencies
        # Look at what people do with \mu band
        X = np.mean(X, axis=-1)
        # X = np.trapz(X, axis=-1)
        return X

    # def fit_transform(self, X, y):
    #     self.fit(self, X, y)
    #     self.transform(self, X)
    #     return self