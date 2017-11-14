import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from utils import INPUT_SHAPE, batch_generator

import argparse
import os


# For reproducible results
np.random.seed(0)


def load_data(args):
  """ Loads human-generated training data and splits it into training and
      validation set.
  """
  # 1) Load data from .csv file
  data = pd.read_csv(
    os.path.join(
      os.getcwd(), args.data_dir, "driving_log.csv"
    ),
    names=[
      "center", "left", "right", "steering", "throttle", "reverse", "speed"
    ]
  )

  # 2) Separate into inputs and outputs
  X = data[["center", "left", "right"]].values
  y = data["steering"].values

  # 3) Split data into training (80%) and validation (20%) sets 
  X_train, X_validate, y_train, y_validate = train_test_split(
    X, y, test_size=args.test_size, random_state=0
  )

  return X_train, X_validate, y_train, y_validate

def build_model(args):
  """ Builds model using Keras.
  """
  model = Sequential()

  # Image normalization
  model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
  
  # Feature detection
  model.add(Conv2D(
    24, 5, 5, activation="elu", subsample(2, 2)
  ))
  model.add(Conv2D(
    36, 5, 5, activation="elu", subsample(2, 2)
  ))
  model.add(Conv2D(
    48, 5, 5, activation="elu", subsample(2, 2)
  ))
  model.add(Conv2D(
    64, 3, 3, activation="elu"
  ))
  model.add(Conv2D(
    64, 3, 3, activation="elu"
  ))

  # Avoid overfitting
  model.add(Dropout(args.keep_prob))

  # Flatten
  model.add(Flatten())

  # Predict steering angle
  model.add(Dense(100, activation="elu"))
  model.add(Dense(50, activation="elu"))
  model.add(Dense(10, activation="elu"))
  model.add(Dense(1))

  model.summary()

  return model

def train_model(model, args, X_train, X_validate, y_train, y_validate):
  pass

def main():
  pass

if __name__ == "__main__":
  main()
