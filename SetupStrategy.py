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