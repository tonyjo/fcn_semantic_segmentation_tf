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
flags.DEFINE_string("test_dataset_dir",     config['test_dataset_dir'],      "Test Dataset directory")
flags.DEFINE_string("test_name",            config['test_name'],             "Test File")
flags.DEFINE_integer('num_classes',         config['num_classes'],           "Total classes")
flags.DEFINE_string("type1",                config['type1'],                 "Dataset type-- sbd/voc")
flags.DEFINE_string("type2",                config['type2'],                 "Dataset type-- sbd/voc")
flags.DEFINE_string("checkpoint_dir",       config['ckpt_dir'],              "Dir to save the checkpoints")
flags.DEFINE_bool("continue_train",         config['continue_train'],        "Continue Train")
flags.DEFINE_string("init_checkpoint_file", config['init_checkpoint_file'],  "Checkpoint file")
flags.DEFINE_string("mode",                 "Test",                          "Train/Test")
flags.DEFINE_integer("test_batch_size",     config['test_batch_size'],       "The size of of a sample batch")
flags.DEFINE_string("train_model",          config['train_model'],           "Which model to train")
flags.DEFINE_integer("print_every",         config['print_every'],           "Print Every n epochs")
FLAGS = flags.FLAGS
#-------------------------------------------------------------------------------
def main(_):
    # Print the arguments
    pp.pprint(FLAGS.__flags)

    if FLAGS.train_model == 'model1':
        fcn = FCN32s(FLAGS)
        fcn.test()

if __name__ == '__main__':
    tf.app.run()
