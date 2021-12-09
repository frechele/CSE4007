import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class FDASolver:
  def __init__(self):
    self.model = LinearDiscriminantAnalysis(solver='lsqr')

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    self.clf = self.model.fit(X, y)

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    return self.clf.score(X, y)
