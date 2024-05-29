from SetupStrategy import CheckGpu, setup_strategy

#first of all we will checking Tensorflow version and GPU ==>
CheckGpu()

# huggingface TF-T5 implementation has issues when mixed precision is enabled
# we will disable FP16 for this but can be used for training any other model
strategy = setup_strategy(xla=True, fp16=False, no_cuda=False)