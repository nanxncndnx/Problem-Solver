import os
import time
from pathlib import Path

from SetupStrategy import CheckGpu, setup_strategy
from UtilityFunctions import init_logger, fix_all_seeds
from Run import run

#first of all we will checking Tensorflow version and GPU ==>
CheckGpu()

# huggingface TF-T5 implementation has issues when mixed precision is enabled
# we will disable FP16 for this but can be used for training any other model
strategy = setup_strategy(xla=True, fp16=False, no_cuda=False)

#lets Execute everuthing we have and build the model ==>
class Args():
    # define training arguments
    
    # MODEL
    model_type = 't5'
    tokenizer_name = 'Salesforce/codet5-base'
    model_name_or_path = 'Salesforce/codet5-base'
    
    # DATA
    train_batch_size = 8
    validation_batch_size = 8
    max_input_length = 48
    max_target_length = 128
    prefix = "Generate Python: "    

    # OPTIMIZER
    learning_rate = 3e-4
    weight_decay = 1e-4
    warmup_ratio = 0.2
    adam_epsilon = 1e-8

    # TRAINING
    seed = 2022
    epochs = 20

    # DIRECTORIES
    output_dir = "runs/"
    logging_dir = f"{output_dir}/logs/"
    checkpoint_dir = f"checkpoint"
    save_dir = f"{output_dir}/saved_model/"
    cache_dir = '../working/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    

# initialize training arguments
args = Args()
# initialize logger
logger = init_logger(log_file=os.path.join(args.logging_dir, f"{args.model_type}-{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.log"))
# fix all seeds
fix_all_seeds(args.seed)

if __name__ == "__main__":
    # run training and evaluation
    dataset = run(args, strategy, logger)