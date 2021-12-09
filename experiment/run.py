import argparse
import pickle
import time
import random
import numpy as np

from sklearn.preprocessing import StandardScaler

from experiment.models.logistic_regression import LogisticRegressionSolver
from experiment.models.svm import *
from experiment.models.fda import FDASolver
from experiment.models.nn import MLPSolver, CNNSolver

SOLVERS = {
  'logistic_reg': LogisticRegressionSolver,
  'svm_linear': SVMLinear,
  'svm_rbf': SVMRBF,
  'svm_sigmoid': SVMSigmoid,
  'svm_poly_deg2': SVMPolyDeg2,
  'svm_poly_deg3': SVMPolyDeg3,
  'fda': FDASolver,
  'mlp': MLPSolver,
  'cnn': CNNSolver,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--solver', choices=list(SOLVERS.keys()), required=True)
  parser.add_argument('--normalize', action='store_true')
  parser.add_argument('--translate', action='store_true')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  with open('data/train.pkl', 'rb') as f:
    train_data = pickle.load(f)
    x_train, y_train = train_data['images'], train_data['targets']

    indicies = list(range(x_train.shape[0]))
    random.shuffle(indicies)

    x_train = x_train[indicies]
    y_train = y_train[indicies]

    del train_data

  with open('data/validation.pkl', 'rb') as f:
    val_data = pickle.load(f)
    x_val, y_val = val_data['images'], val_data['targets']
    del val_data

  if args.normalize:
    x_train = StandardScaler().fit(x_train).transform(x_train)
    x_val = StandardScaler().fit(x_val).transform(x_val)

  if args.translate:
    left_x_val = np.zeros(x_val.shape)
    left_x_val[:, :, 1:] = x_val[:, :, :-1]

    right_x_val = np.zeros(x_val.shape)
    right_x_val[:, :, :-1] = x_val[:, :, 1:]

    x_val = np.concatenate((left_x_val, right_x_val), axis=0)
    y_val = np.concatenate((y_val, y_val), axis=0)

  train_times = []
  inference_speeds = []
  val_accs = []

  for trial in range(100):
    solver = SOLVERS[args.solver]()

    train_begin_time = time.time()
    solver.train(x_train, y_train)
    train_end_time = time.time()

    val_begin_time = time.time()
    acc = solver.score(x_val, y_val)
    val_end_time = time.time()

    train_times.append(train_end_time - train_begin_time)
    inference_speeds.append(x_val.shape[0] / (val_end_time - val_begin_time))
    val_accs.append(acc)

  print(f'[SOLVER {args.solver}]')
  print(f'- training time: {np.mean(train_times):.2f} seconds')
  print(f'- inference speed: {np.mean(inference_speeds):.2f} samples/second')
  print(f'- validation acc: {np.mean(val_accs):.2f}')
  print(flush=True)
