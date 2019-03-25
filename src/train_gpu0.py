from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
import pprint
import argparse
import numpy as np
import tensorflow as tf
from model_fcn32s import FCN32s

pp = pprint.PrettyPrinter(indent=1)
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default='./training', help="Load config file")
args = parser.parse_args()
print('FLAGS:')
print('----------------------------------------')
for arg in vars(args):
    print(arg, ':' , getattr(args, arg))
print('----------------------------------------')
#-------------------------------------------------------------------------------
with open(args.config_file) as f:
        config = yaml.load(f)
#-------------------------------------------------------------------------------
flags = tf.app.flags
flags.DEFINE_string("dataset_name",         config['dataset_name'],          "Dataset Name")
flags.DEFINE_string("exp_dir",              config['exp_dir'],               "Experiment directory")
flags.DEFINE_string("train_dataset_dir",    config['train_dataset_dir'],     "Train Dataset directory")
flags.DEFINE_string("test_dataset_dir",     config['test_dataset_dir'],      "Test Dataset directory")
flags.DEFINE_string("train_name",           config['train_name'],            "Train File")
flags.DEFINE_string("test_name",            config['test_name'],             "Test File")
flags.DEFINE_integer('num_classes',         config['num_classes'],           "Total classes")
flags.DEFINE_string("type1",                config['type1'],                 "Dataset type-- sbd/voc")
flags.DEFINE_string("type2",                config['type2'],                 "Dataset type-- sbd/voc")
flags.DEFINE_string("checkpoint_dir",       config['ckpt_dir'],              "Dir to save the checkpoints")
flags.DEFINE_bool("continue_train",         config['continue_train'],       "Continue Train")
flags.DEFINE_string("init_checkpoint_file", config['init_checkpoint_file'],  "Checkpoint file")
flags.DEFINE_string("logs_path",            config['log_dir'],               "Tensorboard log path")
flags.DEFINE_string("mode",                 "Train",                         "Train/Test")
flags.DEFINE_integer("batch_size",          config['batch_size'],            "The size of of a sample batch")
flags.DEFINE_integer("test_batch_size",     config['test_batch_size'],       "The size of of a sample batch")
flags.DEFINE_string("train_model",          config['train_model'],           "Which model to train")
flags.DEFINE_integer("global_step",         config['global_step'],           "Starting Global step")
flags.DEFINE_integer("start_step",          config['start_step'],            "Starting training step")
flags.DEFINE_integer("epochs",              config['epochs'],                "Maximum number of epochs")
flags.DEFINE_integer("summary_freq",        config['summary_freq'],          "Logging every log_freq iterations")
flags.DEFINE_integer("l_rate_decay_epoch",  config['l_rate_decay_epoch'],    "Learning rate decay epoch")
flags.DEFINE_float("l2",                    config['l2'],                    "Weight Decay")
flags.DEFINE_float("l_rate",                config['l_rate'],                "Learning Rate")
flags.DEFINE_integer("print_every",         config['print_every'],           "Print Every n epochs")
FLAGS = flags.FLAGS
#-------------------------------------------------------------------------------
def main(_):
    # Experiments directory
    if not os.path.exists(FLAGS.exp_dir):
        os.makedirs(FLAGS.exp_dir)

    # Make dataset name
    data_name_path = os.path.join(FLAGS.exp_dir, FLAGS.dataset_name)
    if not os.path.exists(data_name_path):
        os.makedirs(data_name_path)

    # Make checkpoint_dir
    checkpoint_dir_path = os.path.join(data_name_path, FLAGS.checkpoint_dir)
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)

    # Print the arguments
    pp.pprint(FLAGS.__flags)

    # Save the config parameters
    config_param_path = os.path.join(checkpoint_dir_path, 'train_params.txt')
    f = open(config_param_path, 'w')
    for param in config:
        write_param = param + ' : ' + str(config[param])
        f.write(write_param)
        f.write('\n')
    f.close()

    if FLAGS.train_model == 'model1':
        fcn = FCN32s(FLAGS)
        fcn.train()

if __name__ == '__main__':
    tf.app.run()
