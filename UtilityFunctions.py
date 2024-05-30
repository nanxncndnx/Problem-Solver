import os
import time
import random
import logging
from pathlib import Path

import tensorflow as tf

def fix_all_seeds(seed):
    #set random seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

#initialize logger for tracking events and save in file
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if isinstance(log_file, Path):
        log_file = str(log_file)

    log_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)

        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        
    return logger

#custom progress bar to display metrics ==>
class ProgressBar(object): 
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['

        if recv_per >= 1:
            recv_per = 1

        prog_width = int(self.width * recv_per)

        if prog_width > 0:
            bar += '=' * (prog_width - 1)

            if current< self.n_total:
                bar += ">"

            else:
                bar += '='

        bar += '.' * (self.width - prog_width)
        bar += ']'

        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current

        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)

            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
                
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)

            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'

        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'

            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'

            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info

        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' if key != "learning_rate" else f' {key}: {value:.8f} ' for key, value in info.items()])
            print(show_info, end='')

        else:
            print(show_bar, end='')