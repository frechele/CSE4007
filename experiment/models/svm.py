import numpy as np

from sklearn.svm import SVC


class SVMLinear:
  def __init__(self):
    self.model = SVC(decision_function_shape='ovr', kernel='linear')

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    self.clf = self.model.fit(X, y)

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    return self.clf.score(X, y)


class SVMRBF:
  def __init__(self):
    self.model = SVC(decision_function_shape='ovr', kernel='rbf')

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    self.clf = self.model.fit(X, y)

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    return self.clf.score(X, y)


class SVMSigmoid:
  def __init__(self):
    self.model = SVC(decision_function_shape='ovr', kernel='sigmoid')

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    self.clf = self.model.fit(X, y)

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    return self.clf.score(X, y)


class SVMPolyDeg2:
  def __init__(self):
    self.model = SVC(decision_function_shape='ovr', kernel='poly', degree=2)

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    self.clf = self.model.fit(X, y)

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    return self.clf.score(X, y)


class SVMPolyDeg3:
  def __init__(self):
    self.model = SVC(decision_function_shape='ovr', kernel='poly', degree=3)

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    self.clf = self.model.fit(X, y)

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)

    return self.clf.score(X, y)
