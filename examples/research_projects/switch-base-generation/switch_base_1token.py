from transformers import AutoTokenizer, SwitchTransformersModel, SwitchTransformersModelYRJ, ExpertManager
# from transformers.models.switch_transformers import LoadFunc, globalMultiEvent, globalMultiThreadRun, globalMultiParam, globalEM, globalMultiLock
# import multiprocessing
import torch
import pickle
import time
import os
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 1

# ExpertManager.FTpwd = "-samsum"
# base_num = 8

ExpertManager.FTpwd = "-samsum"
base_num = 16

# ExpertManager.FTpwd = ""
# base_num = 32


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")
    model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")
    # model = SwitchTransformersModel.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"")


    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("t", return_tensors="pt").input_ids  # Batch size 1


    decoder_input_ids = model._shift_right(decoder_input_ids)


    time_ = 0
    loop = 1
    for i in range(0, loop):
        start = time.time()
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        # print(outputs[0])
        end = time.time()
        time_ += (end - start)/loop
    print("Inference Time:", time_, "s")


