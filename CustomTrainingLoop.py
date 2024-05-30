import os
import math
import datetime
import logging

import tensorflow as tf
import transformers
from transformers import create_optimizer

from UtilityFunctions import ProgressBar
from SetupStrategy import n_replicas

class Trainer:
    def __init__(
        self, model, args, train_dataset, validation_dataset, 
        num_train_examples, num_validation_examples
    ):
        
        self.model = model
        self.args = args
        
        self.train_dataset = train_dataset
        self.num_train_examples = num_train_examples
        
        self.validation_dataset = validation_dataset
        self.num_validation_examples = num_validation_examples
        
        self.global_step = 0
        self.eval_loss = tf.keras.metrics.Sum()
        
    #creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.
    def create_optimizer_and_scheduler(self, num_training_steps):
        num_warmup_steps = math.ceil(num_training_steps * self.args.warmup_ratio)

        self.optimizer, self.lr_scheduler = create_optimizer(
            init_lr=self.args.learning_rate,
            num_train_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            weight_decay_rate=self.args.weight_decay,
            adam_epsilon=self.args.adam_epsilon
        )
    
    def evaluation_step(self, features, labels, nb_instances_in_global_batch):
        #forward pass
        outputs = self.model(input_ids=features['input_ids'], attention_mask=features['attention_mask'], labels=labels, training=False)[:2]
        loss, logits = outputs[:2]

        #loss scaling
        scaled_loss = loss / tf.cast(nb_instances_in_global_batch, dtype=loss.dtype)
        
        #add current batch loss
        self.eval_loss.update_state(scaled_loss)
    
    @tf.function

    def distributed_evaluation_steps(self, batch, strategy):
        features = {k: v for k, v in batch.items() if 'labels' not in k}
        labels = batch['labels']
        nb_instances = tf.reduce_sum(tf.cast(labels != -100, dtype=tf.int32))
        
        #strategy.run() expects args to be a list or tuple
        inputs = (features, labels, nb_instances)

        #`run` replicates the provided computation and runs with the distributed input
        strategy.run(self.evaluation_step, inputs)

    def evaluate(self, strategy):
        #calculate total validation steps
        steps = math.ceil(self.num_validation_examples / self.args.validation_batch_size)
        
        #reset eval loss after every epoch
        self.eval_loss.reset_states()
        logs = {}
        pbar = ProgressBar(n_total=steps, desc='Evaluating')
        
        #iterate over validation dataset
        for step, batch in enumerate(self.validation_dataset): 
            #distributed evaluation step
            self.distributed_evaluation_steps(batch, strategy) 
            logs["eval_loss"] = self.eval_loss.result() / (step + 1)
            pbar(step=step, info=logs)
            
            if step == steps - 1:
                break

        print("\n------------- validation result -----------------")
        
    def apply_gradients(self, features, labels, nb_instances_in_global_batch):
        #forward pass
        outputs = self.model(input_ids=features['input_ids'], attention_mask=features['attention_mask'], labels=labels, training=True)[:2] 
        loss, logits = outputs[:2]
        
        #loss scaling
        scaled_loss = loss / tf.cast(nb_instances_in_global_batch, dtype=loss.dtype) 
        
        #calculate gradients
        gradients = tf.gradients(scaled_loss, self.model.trainable_variables) 
        
        #convert gradients with nan value
        gradients = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)] 
        
        #optimize the model
        self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables))) 
        
        #add current batch loss
        self.train_loss.update_state(scaled_loss) 
    
    @tf.function

    def distributed_training_steps(self, batch, strategy):
        with strategy.scope():
            features = {k: v for k, v in batch.items() if 'labels' not in k}
            labels = batch['labels']
            nb_instances = tf.reduce_sum(tf.cast(labels != -100, dtype=tf.int32))
            
            #strategy.run() expects args to be a list or tuple
            inputs = (features, labels, nb_instances)
            
            #`run` replicates the provided computation and runs with the distributed input.
            strategy.run(self.apply_gradients, inputs)
    
    def train(self, strategy, logger):
        #calculate total training steps
        num_updates_per_epoch = self.num_train_examples // self.args.train_batch_size 
        self.steps_per_epoch = num_updates_per_epoch
        t_total = self.steps_per_epoch * self.args.epochs
        
        with strategy.scope():
            #optimizer, and checkpoint must be created under `strategy.scope`
            #create optimizer and scheduler
            self.create_optimizer_and_scheduler(num_training_steps=t_total) 
            
            #create checkpoint manager
            folder = os.path.join(self.args.output_dir, self.args.checkpoint_dir)
            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model) 
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=1)
            iterations = self.optimizer.iterations
            
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {self.num_train_examples}")
            logger.info(f"  Num Epochs = {self.args.epochs}")
            logger.info(f"  Total train batch size (w. parallel & distributed) = {self.args.train_batch_size * n_replicas(strategy)}")
            logger.info(f"  Steps per epoch = {self.steps_per_epoch}")
            logger.info(f"  Total optimization steps = {t_total}")
            
            self.train_loss = tf.keras.metrics.Sum(name="training_loss")
            start_time = datetime.datetime.now()
            
            for epoch_iter in range(self.args.epochs):
                #training loop
                logger.info(f"Epoch {epoch_iter + 1}/{self.args.epochs}")
                
                pbar = ProgressBar(n_total=self.steps_per_epoch, desc='Training')
                #iterate over training dataset
                
                for step, batch in enumerate(self.train_dataset):    
                    #distributed training step
                    self.distributed_training_steps(batch, strategy) 
                    
                    self.global_step = iterations.numpy()
                    training_loss = self.train_loss.result() / (step + 1)
                    
                    logs = {}
                    logs["training_loss"] = training_loss.numpy()
                    logs["learning_rate"] = self.lr_scheduler(self.global_step).numpy()
                    pbar(step=step, info=logs)
                    
                    if self.global_step % self.steps_per_epoch == 0:
                        print("\n------------- train result -----------------")
                        
                        #call to evaluation loop
                        self.evaluate(strategy)
                        
                        #save checkpoint
                        ckpt_save_path = self.model.ckpt_manager.save()
                        logger.info(f"Saving checkpoint at {ckpt_save_path}")
                        break
                
                #reset train loss after every epoch
                self.train_loss.reset_states()
            
            end_time = datetime.datetime.now()
            logger.info(f"Training took: {str(end_time - start_time)}")