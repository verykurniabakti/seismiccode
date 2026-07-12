import tensorflow as tf
import os
from datetime import datetime

SEED = 2023
SAMPLING_RATE = 100

# network
# SPACIAL_RESOLUTION = 100
MARGIN = 0.5
ALPHA = 1

ACTIVITION = "relu"

# train
BATCH_SIZE = 32
BUFFER_SIZE = BATCH_SIZE * 2
AUTO = tf.data.AUTOTUNE   # define autotune
LEARNING_RATE = 0.001

EPOCHS = 15 # 100
STEPS_PER_EPOCH = 30 #  100
VALIDATION_STEPS = 5 #  20

# metrics
FEATURE_DISTANCE = "Wasserstein"  # Wasserstein or Euclidean

# test
TEST_BATCH_SIZE = 128
CROP_CONFIDENCE = 0.5

# save
# create save folder
now = datetime.now()
time_str = now.strftime("%d%H%M")
OUTPUT_PATH = "output {}".format(time_str)


