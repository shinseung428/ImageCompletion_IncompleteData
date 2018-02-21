
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import cv2 

from measurement import *

#function to get training data
def load_train_data(args):
	paths = os.path.join(args.data, "img_align_celeba/*.jpg")
	data_count = len(glob(paths))
	
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))

	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	images = tf.image.decode_jpeg(image_file, channels=3)

	#input image range from -1 to 1
	#center crop 32x32 since raw images are not center cropped.
	images = tf.image.central_crop(images, 0.5)
	images = tf.image.resize_images(images ,[args.input_height, args.input_width])
	images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1

	#apply measurement models
	# if args.measurement == "block_pixels":
	# 	images = block_pixels(images, p=0.5)
	# elif args.measurement == "block_patch":
		# images, mask = block_patch(images, k_size=28)
	# elif args.measurement == "keep_patch":
	# 	images = keep_patch(images, k_size=32)
	# elif args.measurement == "conv_noise":
	# 	images = conv_noise(images, k_size=3, stddev=0.1)	

	if args.measurement == "block_patch":
		images, mask = block_patch(images, k_size=28)
	elif args.measurement == "keep_patch":
		images, mask = keep_patch(images, k_size=32)
	
	mask = tf.reshape(mask, [args.input_height, args.input_height, 3])
	mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

	train_batch, masks = tf.train.shuffle_batch([images, mask],
										 batch_size=args.batch_size,
										 capacity=args.batch_size*2,
										 min_after_dequeue=args.batch_size
										)


	return train_batch, masks, data_count


#function to save images in tile
#comment this function block if you don't have opencv
def img_tile(epoch, args, imgs, aspect_ratio=1.0, tile_shape=[8,8], border=1, border_color=0, name="input"):
	if imgs.ndim != 3 and imgs.ndim != 4:
		raise ValueError('imgs has wrong number of dimensions.')
	n_imgs = imgs.shape[0]

	#tile_shape = None
	# Grid shape
	img_shape = np.array(imgs.shape[1:3])
	if tile_shape is None:
		img_aspect_ratio = img_shape[1] / float(img_shape[0])
		aspect_ratio *= img_aspect_ratio
		tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
		tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
		grid_shape = np.array((tile_height, tile_width))
	else:
		assert len(tile_shape) == 2
		grid_shape = np.array(tile_shape)

	# Tile image shape
	tile_img_shape = np.array(imgs.shape[1:])
	tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

	# Assemble tile image
	tile_img = np.empty(tile_img_shape)
	tile_img[:] = border_color
	for i in range(grid_shape[0]):
		for j in range(grid_shape[1]):
			img_idx = j + i*grid_shape[1]
			if img_idx >= n_imgs:
				# No more images - stop filling out the grid.
				break
			img = imgs[img_idx]
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			yoff = (img_shape[0] + border) * i
			xoff = (img_shape[1] + border) * j
			tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img

	cv2.imwrite(args.images_path+"/img_"+str(epoch)+"_"+name+ ".jpg", (tile_img + 1)*127.5)
