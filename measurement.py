import tensorflow as tf 
import numpy as np
import scipy.stats as st

#A randomly chosen patch is set to zero
def block_patch(input, k_size=32):
	shape = input.get_shape().as_list()

	#for training images
	if len(shape) == 3:
		patch = tf.zeros([k_size, k_size, shape[-1]], dtype=tf.float32)
	 
		rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-k_size, dtype=tf.int32)
		h_, w_ = rand_num[0], rand_num[1]

		padding = [[h_, shape[0]-h_-k_size], [w_, shape[1]-w_-k_size], [0, 0]]
		padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

		res = tf.multiply(input, padded) + (1-padded)
	#for generated images
	else:
		patch = tf.zeros([k_size, k_size, shape[-1]], dtype=tf.float32)
	 
		res = []
		for idx in range(0,shape[0]):
			rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-k_size, dtype=tf.int32)
			h_, w_ = rand_num[0], rand_num[1]

			padding = [[h_, shape[0]-h_-k_size], [w_, shape[1]-w_-k_size], [0, 0]]
			padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

			res.append(tf.multiply(input[idx], padded) + (1-padded))
		res = tf.stack(res)

	return res, padded

#All pixels outside a randomly chosen patch are set to zero
def keep_patch(input, k_size=32):
	shape = input.get_shape().as_list()
	#for training images
	if len(shape) == 3:
		#generate a patch
		patch = tf.ones([k_size, k_size, shape[-1]], dtype=tf.float32)
	 	
	 	#add padding of 0 randomly to all sides (size should not be greater than the image)
		rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-k_size, dtype=tf.int32)
		h_, w_ = rand_num[0], rand_num[1]
		padding = [[h_, shape[0]-h_-k_size], [w_, shape[1]-w_-k_size], [0, 0]]
		padded = tf.pad(patch, padding, "CONSTANT", constant_values=0)
		res = tf.multiply(input, padded) + (1-padded) 

	#for generated images
	else:
		patch = tf.ones([k_size, k_size, shape[-1]], dtype=tf.float32)
	 
		res = []
		for idx in range(0,shape[0]):
			rand_num = tf.random_uniform([2], minval=0, maxval=shape[0]-k_size, dtype=tf.int32)
			h_, w_ = rand_num[0], rand_num[1]

			padding = [[h_, shape[0]-h_-k_size], [w_, shape[1]-w_-k_size], [0, 0]]
			padded = tf.pad(patch, padding, "CONSTANT", constant_values=0)

			res.append(tf.multiply(input[idx], padded) + (1-padded))
		res = tf.stack(res)

	return res, padded


