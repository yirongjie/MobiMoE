from transformers import AutoTokenizer, SwitchTransformersModel, SwitchTransformersModelYRJ, ExpertManager
from transformers.models.switch_transformers import LoadFunc, globalMultiEvent, globalMultiThreadRun, globalMultiParam, globalMultiLock
import multiprocessing
# import torch
# import pickle
import time
import os
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 12


if __name__ == '__main__':
    expert_manager = ExpertManager()

    # loadmutli = multiprocessing.Process(target=LoadFunc,args=(8, expert_manager, globalMultiEvent, globalMultiThreadRun, globalMultiParam, globalMultiLock))
    # loadmutli.start()
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj")
    model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj", expert_manager)

    loadmutli = multiprocessing.Process(target=LoadFunc,args=(8, expert_manager, globalMultiEvent, globalMultiThreadRun, globalMultiParam, globalMultiLock))
    loadmutli.start()
    
    time.sleep(10)

    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1


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


