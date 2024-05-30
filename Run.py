import logging
from datasets import load_dataset

from transformers import RobertaTokenizer, TFT5ForConditionalGeneration

from DatasetProcessing import download_dataset
from DatasetProcessing import convert_examples_to_features, get_train_tfdataset, get_validation_tfdataset
from CustomTrainingLoop import Trainer

def run(args, strategy, logger):
    logger.info(" Starting training / evaluation")
    
    logger.info(" Downloading Data Files")
    dataset_path = download_dataset(args.cache_dir) 

    logger.info(" Loading Data Files")
    dataset = load_dataset('json', data_files=dataset_path) 
    # train test split
    dataset = dataset['train'].train_test_split(0.1, shuffle=False) 
        
    logger.info(" Initializing Tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name) 
    
    logger.info(" Preparing Features")
    dataset = dataset.map(convert_examples_to_features, batched=True, fn_kwargs={"tokenizer":tokenizer, "args":args})

    logger.info(" Intializing training and validation dataset ")
    train_dataset = dataset['train']
    num_train_examples = len(dataset['train'])
    # create tf train dataset
    tf_train_dataset = get_train_tfdataset(train_dataset, num_train_examples, args, strategy) 
    
    validation_dataset = dataset['test']
    num_validation_examples = len(dataset['test'])
    # create tf validation dataset
    tf_validation_dataset = get_validation_tfdataset(train_dataset, num_validation_examples, args, strategy) 
    
    logger.info(f' Intializing model | {args.model_type.upper()} ')
    with strategy.scope():
        # model must be created under `strategy.scope`
        model = TFT5ForConditionalGeneration.from_pretrained(args.model_name_or_path, from_pt=True)
    
    # custom training loop
    trainer = Trainer(model, args, tf_train_dataset, tf_validation_dataset, num_train_examples, num_validation_examples) 
    trainer.train(strategy, logger)
    
    # save pretrained model and tokenizer
    logger.info(f" Saving model in {args.save_dir}")
    trainer.model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)