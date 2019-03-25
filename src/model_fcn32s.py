from __future__ import print_function
import os
import math
import random
import numpy as np
import tensorflow as tf
from data_loader import dataLoader
from vgg19 import VGG_ILSVRC_19_layer as vgg19_train
from vgg19_inference import VGG_ILSVRC_19_layer as vgg19_test
from utils import *
slim = tf.contrib.slim
# Init values
bilinear_wts  = get_upsampling_weight(in_channels=21, out_channels=21, kernel_size=64)
bilinear_init = tf.constant_initializer(value=bilinear_wts, dtype=tf.float32)
VGG_MEAN = [103.939, 116.779, 123.68]

class FCN32s(object):
    def __init__(self, opt):
        self.opt         = opt
        self.mode        = tf.placeholder(tf.bool, name='Train_or_Test_Mode')
        self.images      = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input_image')
        self.labels      = tf.placeholder(tf.float32, [None, 224, 224, opt.num_classes], name='labels')
        self.drop_prob   = tf.placeholder(tf.float32, name='dropout_prob')
        self.global_step = tf.Variable(0, dtype=tf.float32, trainable=False)

    def build_train_graph(self, l_rate_decay_step, re_use=False):
        # Input Pre-processing
        with tf.name_scope('Input_preprocess'):
            b, g, r = tf.split(self.images, 3, 3)
            images_ = tf.concat([b - VGG_MEAN[0],\
                                 g - VGG_MEAN[1],\
                                 r - VGG_MEAN[2]], 3)
            images_ = tf.pad(images_,  [[0, 0], [100, 100], [100, 100], [0, 0]],\
                             mode='CONSTANT', name='Input_Pad', constant_values=0)
        # VGG Model
        vgg_net = vgg19_train({'data': images_})
        vgg_out = vgg_net.layers['drop7']
        # Score Layer
        with tf.variable_scope('score_fr'):
            score_fr = slim.conv2d(vgg_out, opt.num_classes, [1, 1],
                                   activation_fn=None,
                                   padding='SAME',
                                   weights_initializer=tf.zeros_initializer(),
                                   stride=1, scope='score_fr')
        # Upsample
        with tf.variable_scope('upscore'):
            upscore = slim.conv2d_transpose(score_fr, opt.num_classes, [64, 64], stride=32,
                                  activation_fn=None, padding='SAME',
                                  weights_initializer=bilinear_init,
                                  biases_initializer=None,
                                  trainable=False, scope='h_embdd_1_1')
            upscore = upscore[:, 19: (19 + 224), 19: (19 + 224), :] # Crop to match input
        #-------------------------------------------------------------------
        # Softmax-Cross entropy Loss
        upsc_rz = tf.reshape(upscore, (-1, opt.num_classes))
        labl_rz = tf.reshape(self.labels, (-1, opt.num_classes))
        softmax = tf.nn.softmax(upsc_rz) + 0.0001
        loss    = -tf.reduce_sum(labl_rz * tf.log(softmax), reduction_indices=[1])
        loss    = tf.reduce_mean(loss, name='loss_mean')
        #-----------------------------------------------------------------------
        # Weight Decay
        # Add l2-loss for weights only and ignore bias and temperature variables
        if opt.l2 > 0:
            print('L2 regularization:')
            for var in tf.trainable_variables():
                tf_var = var.name
                if tf_var[-8:-2] !=  'biases':
                    print(tf_var)
                    loss = loss + (opt.l2 * tf.nn.l2_loss(var))
        print('...............................................................')
        #-----------------------------------------------------------------------
        # Total Loss Norm
        loss = loss/tf.to_float(opt.batch_size)
        #-----------------------------------------------------------------------
        # Learning Rate Decay
        decay_l_rate = tf.train.exponential_decay(opt.l_rate, self.global_step,\
                                                  l_rate_decay_step, 0.9, staircase=True)
        optim = tf.train.AdamOptimizer(opt.l_rate, beta1=0.9, beta2=0.999,\
                                       epsilon=1e-08, name='Adam')
        #-----------------------------------------------------------------------
        # Other Parameters
        incr_glbl_stp = tf.assign(self.global_step, self.global_step+1)
        #-----------------------------------------------------------------------
        # Gradients
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))

        # Get the gradients
        train_op = optim.minimize(loss, var_list=tf.trainable_variables())

        self.upscore  = upscore
        self.vgg_net  = vgg_net
        self.train_op = train_op
        self.incr_glbl_stp = incr_glbl_stp

    def pixela_acc(self, pred)
    def deprocess_pred(self, pred):
        opt = self.opt
        # Assuming input image is float32
        upsc_rs = tf.reshape(self.upscore, (-1, opt.num_classes))
        softmax = tf.nn.softmax(upsc_rs)
        pred  = tf.reshape(softmax, [opt.batch_size, 244, 244, opt.num_classes])
        pred  = tf.math.argmax(softmax, axis=-1, output_type=tf.dtypes.int32)
        alpha = tf.image.convert_image_dtype(alpha, dtype=tf.uint8)
        alpha = tf.image.resize_images(alpha, (100, 100), method=ResizeMethod.NEAREST_NEIGHBOR)

        return alpha

    def train(self):
        opt = self.opt
        # Some large value
        best_pred = 0.0

        # Get Loss
        train_loss = []
        test_loss  = []

        # Checkpoint_path
        ckpt_dir_path = os.path.join(opt.exp_dir, opt.dataset_name, opt.checkpoint_dir)

        # Train Data Loader
        train_loader = dataLoader(opt.dataset_dir, opt.train_name, 224, 224,
                                  mode='Train', dtype=opt.type)
        train_gen    = train_loader.gen_data_batch(batch_size=opt.batch_size)
        # Val Data Loader
        val_loader   = dataLoader(opt.dataset_dir, opt.val_name, 224, 224,
                                  mode='Test', dtype=opt.type)
        val_gen      = val_loader.gen_data_batch(batch_size=opt.vald_batch_size)
        # Test Data Loader
        test_loader  = dataLoader(opt.dataset_dir, opt.test_name, 224, 224,
                                  mode='Test', dtype=opt.type)
        test_gen     = test_loader.gen_data_batch(batch_size=opt.test_batch_size)

        # Compute steps
        n_examples        = train_loader.max_steps
        n_iters_per_epoch = int(np.ceil(float(n_examples)/opt.batch_size))
        training_steps    = int(opt.epochs             * n_iters_per_epoch)
        decay_step        = int(opt.temperature_decay  * n_iters_per_epoch)
        l_rate_decay_step = int(opt.l_rate_decay_epoch * n_iters_per_epoch)
        print('Total Training Steps: ', training_steps)
        print('Temperature decay step: ', decay_step)
        print('Learning rate decay step: ', l_rate_decay_step)

        # Build graph
        self.build_train_graph_gnet(l_rate_decay_step=l_rate_decay_step)

        # Setup loss scalars
        tf.summary.scalar("Loss", self.loss)
        # Predictions
        tf.summary.image("Prediction ", self.deprocess_pred(self.alphas[i]), max_outputs=4)

        # Merge all summaries into a single "operation"
        summary_op = tf.summary.merge_all()

        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        # To save model
        init  = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # create log writer object
            writer = tf.summary.FileWriter(opt.logs_path, graph=sess.graph)

            # Intialize the graph
            sess.run(init)

            # Load the pre-trainined googlenet weights
            self.vgg_net.load('./googlenet_weights/googlenet_places.npy', sess)
            print('Pre-trainined Googlenet weights loaded')

            # Check if training has to be continued
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    print('Enter a valid checkpoint file')
                else:
                    load_model = os.path.join(checkpoint_dir_path, opt.init_checkpoint_file)
                    saver.restore(sess, load_model)
                    sess.run(tf.assign(self.global_step, opt.global_step))
                    print("Resume training from previous checkpoint: %s" % opt.init_checkpoint_file)

            if median_result[0] > best_pred:
                best_pred  = median_result[0]
                model_name = 'posenet_bp_' + str(i)
                checkpoint_path = os.path.join(ckpt_dir_path, model_name)
                saver.save(sess, checkpoint_path)
                print("Intermediate file saved")
                sw_path = os.path.join(ckpt_dir_path, model_name + '_sw.npy')
                switch_values = np.array(switch_values)
                print(switch_values.shape)
                np.save(sw_path, switch_values)
                print("Switch Values saved!")
