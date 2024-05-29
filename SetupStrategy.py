import os
import time
import math
import random
import datetime
import logging
from pathlib import Path

import tensorflow as tf
import transformers

from datasets import load_dataset

def CheckGpu():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # reduce the amount of console output from TF

    print('TF version',tf.__version__)
    #well i will not using GPU but this is for you ===>
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # check GPU available

#Lets setup xla, distribution strategy, float16 ==>
def setup_strategy(xla, fp16, no_cuda):
    print(" Tensorflow: setting up strategy")
    
    # setup xla
    if xla:
        print(" XLA Enabled")
        tf.config.optimizer.set_jit(True)
    
    # setup mixed precision training
    if fp16:
        # Set to float16 at first
        print(" Mixed Precision Training Enabled")
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
    
    # setup distribution strategy
    gpus = tf.config.list_physical_devices("GPU")
    if no_cuda:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        if len(gpus) == 0:
            print(" One Device Strategy [CPU] Enabled")
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        elif len(gpus) == 1:
            print(" One Device Strategy [GPU] Enabled")
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        elif len(gpus) > 1:
            print(" Mirrored Strategy Enabled")
            # If only want to use a specific subset of GPUs use CUDA_VISIBLE_DEVICES=0`
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

    return strategy

def n_replicas(strategy):
    # return number of devices
    return strategy.num_replicas_in_sync