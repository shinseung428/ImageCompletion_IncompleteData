import tensorflow as tf
import numpy as np

from ops import *
from architecture import *
from measurement import *

class network():
    def __init__(self, args):
        self.measurement = args.measurement
        self.batch_size = args.batch_size

        #prepare training data
        self.Y_r, self.masks, self.data_count = load_train_data(args)

        self.build_model()
        self.build_loss()

        #summary
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss) 
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.Y_r_sum = tf.summary.image("input_img", self.Y_r, max_outputs=5)
        self.X_g_sum = tf.summary.image("X_g", self.X_g, max_outputs=5)
        self.Y_g_sum = tf.summary.image("Y_g", self.Y_g, max_outputs=5)

    #structure of the model
    def build_model(self):
        self.X_g, self.g_nets = self.completion_net(self.Y_r, name="generator")

        self.X_g = (1 - self.masks)*self.X_g + self.masks*self.Y_r

        self.Y_g, _ = self.measurement_fn(self.X_g, name="measurement_fn")
        self.fake_d_logits, self.fake_d_net = self.discriminator(self.Y_g, name="discriminator")
        self.real_d_logits, self.real_d_net = self.discriminator(self.Y_r, name="discriminator", reuse=True)

        trainable_vars = tf.trainable_variables()
        self.g_vars = []
        self.d_vars = []
        for var in trainable_vars:
            if "generator" in var.name:
                self.g_vars.append(var)
            else:
                self.d_vars.append(var)

    #loss function setting 
    def build_loss(self):
        def calc_loss(logits, label):
            if label==1:
                y = tf.ones_like(logits)
            else:
                y = tf.zeros_like(logits)
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

        #GAN loss
        #self.real_d_loss = calc_loss(self.real_d_logits, 1)
        #self.fake_d_loss = calc_loss(self.fake_d_logits, 0)
        
        # WGAN loss
        self.real_d_loss = -tf.reduce_mean(self.real_d_logits)
        self.fake_d_loss = tf.reduce_mean(self.fake_d_logits)

        self.d_loss = self.real_d_loss + self.fake_d_loss
        #self.g_loss = calc_loss(self.fake_d_logits, 1)
        self.g_loss = -self.fake_d_loss

    # Completion network in the GLCIC paper
    def completion_net(self, input, name="generator", reuse=False):
        input_shape = input.get_shape().as_list()
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv1 = conv2d(input, 64,
                          kernel=5,
                          stride=1,
                          padding="SAME",
                          name="conv1"
                          )
            conv1 = batch_norm(conv1, name="conv_bn1")
            conv1 = tf.nn.relu(conv1)
            
            conv2 = conv2d(conv1, 128,
                          kernel=3,
                          stride=2,
                          padding="SAME",
                          name="conv2"
                          )
            conv2 = batch_norm(conv2, name="conv_bn2")
            conv2 = tf.nn.relu(conv2)

            conv3 = conv2d(conv2, 128,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv3"
                          )
            conv3 = batch_norm(conv3, name="conv_bn3")
            conv3 = tf.nn.relu(conv3)

            conv4 = conv2d(conv3, 256,
                          kernel=3,
                          stride=2,
                          padding="SAME",
                          name="conv4"
                          )
            conv4 = batch_norm(conv4, name="conv_bn4")
            conv4 = tf.nn.relu(conv4)

            conv5 = conv2d(conv4, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv5"
                          )
            conv5 = batch_norm(conv5, name="conv_bn5")
            conv5 = tf.nn.relu(conv5)

            conv6 = conv2d(conv5, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv6"
                          )
            conv6 = batch_norm(conv5, name="conv_bn6")
            conv6 = tf.nn.relu(conv5)

            #Dilated conv from here
            dilate_conv1 = dilate_conv2d(conv6, 
                                        [self.batch_size, conv6.get_shape()[1], conv6.get_shape()[2], 256],
                                        rate=2,
                                        name="dilate_conv1")

            dilate_conv2 = dilate_conv2d(dilate_conv1, 
                                        [self.batch_size, dilate_conv1.get_shape()[1], dilate_conv1.get_shape()[2], 256],
                                        rate=4,
                                        name="dilate_conv2")

            dilate_conv3 = dilate_conv2d(dilate_conv2, 
                                        [self.batch_size, dilate_conv2.get_shape()[1], dilate_conv2.get_shape()[2], 256],
                                        rate=8,
                                        name="dilate_conv3")

            dilate_conv4 = dilate_conv2d(dilate_conv3, 
                                        [self.batch_size, dilate_conv3.get_shape()[1], dilate_conv3.get_shape()[2], 256],
                                        rate=16,
                                        name="dilate_conv4")                                                                                              

            #resize back
            conv7 = conv2d(dilate_conv4, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv7"
                          )
            conv7 = batch_norm(conv7, name="conv_bn7")
            conv7 = tf.nn.relu(conv7)

            conv8 = conv2d(conv7, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv8"
                          )
            conv8 = batch_norm(conv8, name="conv_bn8")
            conv8 = tf.nn.relu(conv8)

            deconv1 = deconv2d(conv8, 4, [self.batch_size, input_shape[1]/2, input_shape[2]/2, 128], name="deconv1")
            deconv1 = batch_norm(deconv1, name="deconv_bn1")
            deconv1 = tf.nn.relu(deconv1)

            conv9 = conv2d(deconv1, 128,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv9"
                          )
            conv9 = batch_norm(conv9, name="conv_bn9")
            conv9 = tf.nn.relu(conv9)

            deconv2 = deconv2d(conv9, 4, [self.batch_size, input_shape[1], input_shape[2], 64], name="deconv2")
            deconv2 = batch_norm(deconv2, name="deconv_bn2")
            deconv2 = tf.nn.relu(deconv2)

            conv10 = conv2d(deconv2, 32,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv10"
                          )
            conv10 = batch_norm(conv10, name="conv_bn10")
            conv10 = tf.nn.relu(conv10)

            conv11 = conv2d(conv10, 3,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv11"
                          )
            conv11 = batch_norm(conv11, name="conv_bn11")
            conv11 = tf.nn.tanh(conv11)

            return conv11, nets

    # D network from DCGAN
    def discriminator(self, input, name="discriminator", reuse=False):
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv1 = tf.contrib.layers.conv2d(input, 64, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv1")
            conv1 = batch_norm(conv1, name="bn1")
            conv1 = tf.nn.relu(conv1)
            nets.append(conv1)

            conv2 = tf.contrib.layers.conv2d(conv1, 128, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv2")
            conv2 = batch_norm(conv2, name="bn2")
            conv2 = tf.nn.relu(conv2)
            nets.append(conv2)

            conv3 = tf.contrib.layers.conv2d(conv2, 256, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv3")
            conv3 = batch_norm(conv1, name="bn3")
            conv3 = tf.nn.relu(conv3)
            nets.append(conv3)

            conv4 = tf.contrib.layers.conv2d(conv3, 512, 5, 2,
                                     padding="VALID",
                                     activation_fn=None,
                                     scope="conv4")
            conv4 = batch_norm(conv4, name="bn4")                                                                                                                           
            conv4 = tf.nn.relu(conv4)
            nets.append(conv4)
            
            flatten = tf.contrib.layers.flatten(conv4)

            output = linear(flatten, 1, name="linear")

            return output, nets


    #pass generated image to measurment model
    def measurement_fn(self, input, name="measurement_fn"):
        with tf.variable_scope(name) as scope:            
            if self.measurement == "block_patch":
                return block_patch(input, k_size=28)
            elif self.measurement == "keep_patch":
                return keep_patch(input, k_size=32)













