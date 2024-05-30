import tensorflow as tf
import transformers

#okay lets download data using a keras utility ==>
def download_dataset(cache_dir):
    _url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl" # download mbpp dataset (Mostly Basic Python Problems)
    dataset_path = tf.keras.utils.get_file("mbpp.jsonl", origin=_url, cache_dir=cache_dir, cache_subdir=cache_dir)
    return dataset_path

#encode text-code pairs ==>
def convert_examples_to_features(examples, tokenizer, args):
    texts = examples['text']
    codes = examples['code']
    tests = [" ".join(test) for test in examples['test_list']] # convert list of test cases to single string

    #encode texts by prepending the task for input sequence
    inputs = [args.prefix + text for text in texts]
    model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding="max_length", truncation=True)

    #encode texts by prepending the task for input sequence and appending the test sequence
    #but we dont need it right now ==============================>
    #inputs = [args.prefix + text + " " + test for text, test in zip(texts, tests)]
    #model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding="max_length", truncation=True)

    #encode texts by prepending the task for input sequence
    labels = tokenizer(codes, max_length=args.max_target_length, padding="max_length", truncation=True).input_ids

    #we need to replace the index of the padding tokens by -100
    #such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    model_inputs["labels"] = labels_with_ignore_index
    
    # return features
    return model_inputs

#convert the dataset to tensorflow tf.data.Dataset object ==>
def get_train_tfdataset(train_dataset, num_train_examples, args, strategy):
    #select feature columns
    columns = ['input_ids', 'attention_mask', 'labels'] 
    #set to tensorflow format
    train_dataset.set_format(type='tensorflow', columns=columns)

    #specify return types
    return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 'labels':tf.int32}
    #specify return shapes
    return_shapes = {'input_ids': tf.TensorShape([None]),'attention_mask': tf.TensorShape([None]), 'labels': tf.TensorShape([None])}
    #initialize dataset 
    tf_dataset = tf.data.Dataset.from_generator(lambda : train_dataset, return_types, return_shapes)

    #turn off auto-sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_dataset = tf_dataset.with_options(options)

    #repeat, shuffle, batch, prefetch
    ds = (
        tf_dataset.repeat()
        .shuffle(num_train_examples, seed=args.seed)
        .batch(args.train_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    #distribute dataset to devices
    return strategy.experimental_distribute_dataset(ds)

def get_validation_tfdataset(eval_dataset, num_validation_examples, args, strategy):
    #select feature columns
    columns = ['input_ids', 'attention_mask', 'labels'] 
    #set to tensorflow format
    eval_dataset.set_format(type='tensorflow', columns=columns) 
    
    #specify return types
    return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 'labels':tf.int32} 
    #specify return shapes
    return_shapes = {'input_ids': tf.TensorShape([None]),'attention_mask': tf.TensorShape([None]), 'labels': tf.TensorShape([None])} 
    #initialize dataset 
    tf_dataset = tf.data.Dataset.from_generator(lambda : eval_dataset, return_types, return_shapes) 
    
    #turn off auto-sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_dataset = tf_dataset.with_options(options)
    
    #repeat, batch, prefetch
    ds = (
        tf_dataset.repeat()
        .batch(args.validation_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    #distribute dataset to devices
    return strategy.experimental_distribute_dataset(ds)