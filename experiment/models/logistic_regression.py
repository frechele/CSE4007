import numpy as np

from sklearn.linear_model import LogisticRegression


class LogisticRegressionSolver:
  def __init__(self):
    self.model = LogisticRegression(n_jobs=-1, max_iter=int(1e+10), solver='lbfgs', multi_class='ovr')

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    self.clf = self.model.fit(X, y)

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    return self.clf.score(X, y)
