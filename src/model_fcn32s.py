from __future__ import print_function
import os
import math
import random
import numpy as np
import tensorflow as tf
from data_loader import dataLoader
from vgg19 import VGG_ILSVRC_19_layer
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
        opt = self.opt
        # Input Pre-processing
        with tf.name_scope('Input_preprocess'):
            b, g, r = tf.split(self.images, 3, 3)
            images_ = tf.concat([b - VGG_MEAN[0],\
                                 g - VGG_MEAN[1],\
                                 r - VGG_MEAN[2]], 3)
            images_ = tf.pad(images_,  [[0, 0], [100, 100], [100, 100], [0, 0]],\
                             mode='CONSTANT', name='Input_Pad', constant_values=0)
        # VGG Model
        vgg_net = VGG_ILSVRC_19_layer({'data': images_})
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
        upsc_rz = tf.reshape(upscore,     (-1, opt.num_classes))
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

        self.loss     = loss
        self.upscore  = upscore
        self.vgg_net  = vgg_net
        self.train_op = train_op
        self.incr_glbl_stp = incr_glbl_stp

    def build_test_graph(self, re_use=False):
        opt = self.opt
        # Input Pre-processing
        with tf.name_scope('Input_preprocess'):
            b, g, r = tf.split(self.images, 3, 3)
            images_ = tf.concat([b - VGG_MEAN[0],\
                                 g - VGG_MEAN[1],\
                                 r - VGG_MEAN[2]], 3)
            images_ = tf.pad(images_,  [[0, 0], [100, 100], [100, 100], [0, 0]],\
                             mode='CONSTANT', name='Input_Pad', constant_values=0)
        # VGG Model
        vgg_net = VGG_ILSVRC_19_layer({'data': images_})
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
        # Assuming input image is float32
        upscore = tf.reshape(upscore, (-1, opt.num_classes))
        upscore = tf.nn.softmax(upscore)
        upscore = tf.reshape(upscore, [tf.shape(self.images)[0], 224, 224, opt.num_classes])

        self.vgg_net = vgg_net
        self.upscore = upscore

    def pixel_acc(self):
        opt = self.opt
        # Assuming input image is float32
        upsc_rz = tf.reshape(self.upscore, (-1, opt.num_classes))
        softmax = tf.nn.softmax(upsc_rz)
        softmax = tf.reshape(softmax, [tf.shape(self.labels)[0], 224, 224, opt.num_classes])
        # Argmax input and predictions
        preds  = tf.argmax(softmax,     axis=-1, output_type=tf.int32)
        labels = tf.argmax(self.labels, axis=-1, output_type=tf.int32)
        # Pixel accuracy
        self.px_acc = tf.reduce_mean(tf.metrics.accuracy(labels=labels, predictions=preds))

    def deprocess_pred(self, pred):
        opt = self.opt
        # Assuming input image is float32
        upsc_rs = tf.reshape(self.upscore, (-1, opt.num_classes))
        softmax = tf.nn.softmax(upsc_rs)
        softmax = tf.reshape(softmax, [tf.shape(self.labels)[0], 224, 224, opt.num_classes])
        predict = tf.argmax(softmax, axis=-1, output_type=tf.int32)
        alpha   = colorize(value=predict, name='pred_to_image', opt=opt)
        alpha   = tf.image.convert_image_dtype(predict, dtype=tf.uint8)
        alpha   = tf.image.resize_images(alpha, (100, 100), method=ResizeMethod.NEAREST_NEIGHBOR)

        return alpha

    def train(self):
        opt = self.opt
        # Some large value
        best_acc = 0.0

        # Checkpoint_path
        ckpt_dir_path = os.path.join(opt.exp_dir, opt.dataset_name, opt.checkpoint_dir)

        # Train Data Loader
        train_loader = dataLoader(opt.train_dataset_dir, opt.train_name, 224, 224,
                                  mode='Train', dtype=opt.type1)
        train_gen    = train_loader.gen_data_batch(batch_size=opt.batch_size)
        # Test Data Loader
        test_loader  = dataLoader(opt.test_dataset_dir, opt.test_name, 224, 224,
                                  mode='Test', dtype=opt.type2)
        test_gen     = test_loader.gen_data_batch(batch_size=opt.test_batch_size)

        # Compute steps
        n_examples        = train_loader.max_steps
        n_iters_per_epoch = int(np.ceil(float(n_examples)/opt.batch_size))
        training_steps    = int(opt.epochs             * n_iters_per_epoch)
        l_rate_decay_step = int(opt.l_rate_decay_epoch * n_iters_per_epoch)
        print('Total Training Steps: ', training_steps)
        print('Learning rate decay step: ', l_rate_decay_step)

        # Build graph
        self.build_train_graph(l_rate_decay_step=l_rate_decay_step)
        self.pixel_acc()

        # Setup loss scalars
        tf.summary.scalar("Loss", self.loss)
        # Predictions
        tf.summary.image("Ground_truth", self.deprocess_pred(self.labels), max_outputs=4)
        tf.summary.image("Prediction",   self.deprocess_pred(self.upscore), max_outputs=4)

        # Merge all summaries into a single "operation"
        summary_op = tf.summary.merge_all()

        # Set GPU options
        config = tf.GPUOptions(allow_growth=True)

        # To save model
        init_op = tf.group(tf.global_variables_initializer(),\
                           tf.local_variables_initializer())
        saver   = tf.train.Saver(max_to_keep=5)

        with tf.Session(config=tf.ConfigProto(gpu_options=config)) as sess:
            # create log writer object
            writer = tf.summary.FileWriter(opt.logs_path, graph=sess.graph)

            # Intialize the graph
            sess.run(init_op)

            # Load the pre-trainined googlenet weights
            self.vgg_net.load('./imagenet_weights/vgg19.npy', sess)
            print('Pre-trainined Googlenet weights loaded')

            # Check if training has to be continued
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    print('Enter a valid checkpoint file')
                else:
                    load_model = os.path.join(ckpt_dir_path, opt.init_checkpoint_file)
                    saver.restore(sess, load_model)
                    sess.run(tf.assign(self.global_step, opt.global_step))
                    print("Resume training from previous checkpoint: %s" % opt.init_checkpoint_file)

            # Begin training
            step = 0
            for epoch in range(opt.epochs):
                print('Epoch {}/{}'.format(epoch, opt.epochs))
                print('-' * 10)
                curr_loss = 0.0
                for i in range(n_iters_per_epoch):
                    image_batch, label_batch = next(train_gen)
                    feed = {self.images: image_batch,
                            self.labels: label_batch,
                            self.vgg_net.keep_prob: 0.5}

                    _, l, _ = sess.run([self.train_op, self.loss, self.incr_glbl_stp], feed_dict=feed)
                    curr_loss += l
                    # Increment step
                    step += 1
                    # Run test
                    if i % opt.summary_freq == 0:
                        # Print global step
                        run_global_step = sess.run([self.global_step])
                        # Estimate loss at global time step
                        interm_loss = sess.run(summary_op, feed_dict=feed)
                        print('Global Step:' + str(run_global_step))
                        # Write log
                        writer.add_summary(interm_loss, step)
                        # Validation Accuracy
                        print('Estimating Testing Accuracy....')
                        total_acc = 0.0
                        total_steps = test_loader.max_steps//opt.test_batch_size
                        for j in range(total_steps):
                            val_images, val_labels = next(test_gen)
                            feed = {self.images: val_images,
                                    self.labels: val_labels,
                                    self.vgg_net.keep_prob: 1.0}
                            acc = sess.run(self.px_acc, feed_dict=feed)
                            total_acc += acc
                        # Final accuracy
                        final_accuracy = (total_acc/total_steps) * 100
                        print('Pixel Accuracy: ', final_accuracy)
                        # Save
                        if final_accuracy > best_acc:
                            best_acc  = final_accuracy
                            model_name = 'fcn32s_bp_' + str(step)
                            checkpoint_path = os.path.join(ckpt_dir_path, model_name)
                            saver.save(sess, checkpoint_path)
                            print("Intermediate file saved")

                if i%opt.print_every == 0:
                    print('Epoch Completion..{%d/%d} and loss = %d' % (i, n_iters_per_epoch, curr_loss))
