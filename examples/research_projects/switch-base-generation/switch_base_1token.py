import os
import time
import torch
import pickle
import resource
from transformers import AutoTokenizer, SwitchTransformersModel, SwitchTransformersModelYRJ, ExpertManager

ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-samsum"
base_num = 8

def get_memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

if __name__ == '__main__':
    start_memory = get_memory_usage()

    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")
    model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")

    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids
    decoder_input_ids = tokenizer("t", return_tensors="pt").input_ids
    decoder_input_ids = model._shift_right(decoder_input_ids)

    time_ = 0
    loop = 1
    for i in range(0, loop):
        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        end = time.time()
        time_ += (end - start)/loop

    end_memory = get_memory_usage()
    peak_memory = end_memory - start_memory
    print("Inference Time:", time_, "s")
    print("Peak Memory Usage:", peak_memory/1024.0, "MB")