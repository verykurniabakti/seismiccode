# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm


class Data_Generator:
	def __init__(self, data_frame, seed=None):
		self.data_frame = data_frame
		self.seed = seed
		if self.seed != None:
			np.random.seed(self.seed)

	def get_next_record(self):
		while True:
			random_index = np.random.choice(np.arange(len(self.data_frame)))
			(refer, positive, negative, refer_silence) = self.data_frame["data"].iloc[random_index]

			refer = tf.convert_to_tensor(refer, dtype=tf.float32)
			positive = tf.convert_to_tensor(positive, dtype=tf.float32)
			negative = tf.convert_to_tensor(negative, dtype=tf.float32)
			refer_silence = tf.convert_to_tensor(refer_silence, dtype=tf.float32)

			yield (refer, positive, negative, refer_silence)

	def get_next_test(self):
		while True:
			random_index = np.random.choice(np.arange(len(self.data_frame)))
			(refer, positive, negative, refer_silence) = self.data_frame["data"].iloc[random_index]
			typpe_str = self.data_frame["type"].iloc[random_index]
			(refer_names, positive_names, negative_names) = self.data_frame["choices"].iloc[random_index]
			

			refer = tf.convert_to_tensor(refer, dtype=tf.float32)
			positive = tf.convert_to_tensor(positive, dtype=tf.float32)
			negative = tf.convert_to_tensor(negative, dtype=tf.float32)
			refer_silence = tf.convert_to_tensor(refer_silence, dtype=tf.float32)

			yield (refer, positive, negative, refer_silence,
	  				typpe_str,
			   		refer_names, positive_names, negative_names)
			


def load_json_data(file_path, id=None):
	with open(file_path, "r") as read_file:
		dataset = json.load(read_file)

	if id is not None:
		return dataset[str(id)]

	return dataset



def save_json_data(file_path, dataset, indent="\t"):
    with open(file_path, "w") as outfile: 
        json.dump(dataset, outfile, indent=indent)


def load_embedding_data(data_dir, name):
	return load_json_data(os.path.join(data_dir, name))

