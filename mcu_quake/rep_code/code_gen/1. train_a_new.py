# -*- coding: utf-8 -*-
"""
A demonstration of parsing a chromosome and training a model using CPUs on a personal computer. The training and validation dataset contains only 256 randomly selected records for illustration purposes. Full data are provided in the Data Availability section of the manuscript.

Before starting the training (by running '1. train_one.py'), please adjust the settings in the config.py script as needed. For example:

EPOCHS = 5 # 100
STEPS_PER_EPOCH = 50 #  100
VALIDATION_STEPS = 10 #  20


Other requirements:
python == 3.11
tensorflow == 2.13.0 / 2.15.0

"""

import pandas as pd
import config
import os
import time
import logging
import tensorflow as tf
from tensorflow import keras

from Library.model import Embedding_Network, Contrastive_Network, Contrastive_Model
from Library.dataset import Data_Generator
from Library import utils


#--------------
# Configuration
#--------------

save_dir = f"{config.OUTPUT_PATH} train a new"
Contrastive_SAVE_PATH = os.path.join(save_dir, "Contrastive_Network, {}".format(config.FEATURE_DISTANCE))
EMBEDDING_SAVE_PATH = os.path.join(save_dir, "Embedding_Network, {}".format(config.FEATURE_DISTANCE))

# Create output folder if it does not exist
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# Create and configure logger
logging.basicConfig(filename=os.path.join(save_dir, "Train log.txt"),
                    level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    datefmt = "%Y-%m-%d %H:%M:%S",
                    filemode='w'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


#-------------
# Load dataset
#-------------
logger.info("Load train data ...")

# Define paths for training and validation datasets
# 7 s period records
train_file_path = config.TRAIN_DATA_PATH   # 
val_file_path = config.VAL_DATA_PATH #

# --- load data ---
train_file = pd.read_pickle(train_file_path)
val_file = pd.read_pickle(val_file_path)


logger.info("Build tensor dataset ...")

signal_length = len(train_file["data"].iloc[0][0])
train_generator = Data_Generator(train_file, seed=config.SEED)
val_generator = Data_Generator(val_file, seed=config.SEED)

train_ds = tf.data.Dataset.from_generator(
                            generator=train_generator.get_next_record,
                            output_signature=(
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                )
                            ).batch(config.BATCH_SIZE).prefetch(config.AUTO)

val_ds = tf.data.Dataset.from_generator(
                            generator=val_generator.get_next_record,
                            output_signature=(
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                tf.TensorSpec(shape=(signal_length,), dtype=tf.float32),
                                )
                            ).batch(config.BATCH_SIZE).prefetch(config.AUTO)

#------------
# Build model
#------------

logger.info("Build model ...")


# --- specific chromosome ---

# MCU-Quake, chromosome ID 5-20
chromosome = [8.097911880061828, -0.30367944386676804, -0.35225592325209076, -0.5840144913416598, -0.13185410244369244, 0.7712233111496816, 7.997406552070309, 3.961237228964, 22.835719873542722, 32.38916312891132, 1.146460404651366]

# Model architecture
embedding_network = Embedding_Network(input_size = signal_length,
                                    chromosome = chromosome,
                                    )


embedding_network.summary(print_fn=logger.info)

Contrastive_network = Contrastive_Network(signal_length, embedding_network)

Contrastive_model = Contrastive_Model(
                            Contrastive_network = Contrastive_network,
                            batch_size = config.BATCH_SIZE,
                            loss_tracker = keras.metrics.Mean(name="loss"),
                            feature_distance = config.FEATURE_DISTANCE,
			                metric_acc = keras.metrics.Accuracy(name="acc"),
                            margin = config.MARGIN,
                            alpha = config.ALPHA,
                            )

logger.info(f"Architecture chromosome: {chromosome}")

logger.info("Training setting:\n"
	    f"data_window_length={signal_length/config.SAMPLING_RATE}s, "
	    f"margin={config.MARGIN}, alpha={config.ALPHA}, activation={config.ACTIVITION}, "
	    f"learning_rate={config.LEARNING_RATE}, epochs={config.EPOCHS}, "
		f"steps_per_epoch={config.STEPS_PER_EPOCH}, "
		f"validate_steps={config.VALIDATION_STEPS}, "
		f"test_batch_size={config.TEST_BATCH_SIZE}, "
	    f"feature distance: {config.FEATURE_DISTANCE}.")

# Compile the Contrastive model
Contrastive_model.compile(
					optimizer=keras.optimizers.Adam(config.LEARNING_RATE)
					)


# Call back for the best instance
checkpoint_filepath = os.path.join(Contrastive_SAVE_PATH, "checkpoint/best")
checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                                    filepath=checkpoint_filepath,
                                                    save_weights_only=True,
                                                    monitor='val_acc',  # not accuracy in the conventional sense
                                                    mode='max',
                                                    save_best_only=True)


#-------------------------------------
# Train and validate the Contrastive model
#-------------------------------------

logger.info("Training the contrastive model...")

history = Contrastive_model.fit(
						train_ds,
						steps_per_epoch=config.STEPS_PER_EPOCH,
						validation_data=val_ds,
						validation_steps=config.VALIDATION_STEPS,
						epochs=config.EPOCHS,
						callbacks=[checkpoint_callback],
						)


# Evaluate inference speed
infer_start = time.time()
for i in range(config.VALIDATION_STEPS):
	data = next(iter(val_ds))
	_ = Contrastive_network(data)
infer_elapsed =  time.time() - infer_start

infer_speed = round(infer_elapsed*1000/(config.TEST_BATCH_SIZE*config.VALIDATION_STEPS), 4)   # ms/sample
logger.info(f"Inference speed: {infer_speed} ms/sample.")


# -----------------------------
# Save Best Contrastive Network
# -----------------------------

# Reload the best checkpoint
logger.info(f"Reload the best weights from checkpoint ...")
Contrastive_model.load_weights(checkpoint_filepath)

logger.info(f"Save the best Contrastive network to {save_dir}...")
keras.models.save_model(
						model=Contrastive_model.Contrastive_network,
						filepath=save_dir,
						include_optimizer=False,
						)

# Save embedding network for feature codes
logger.info(f"Saving the best embedding network to {EMBEDDING_SAVE_PATH}...")
keras.models.save_model(
						model=embedding_network,
						filepath=EMBEDDING_SAVE_PATH,
						include_optimizer=False,
						)


#--------------------------
# Plot the training history
#--------------------------
logger.info(f"Plotting training history...")
utils.plot_training(history, save_dir)