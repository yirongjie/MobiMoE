from transformers import AutoTokenizer, SwitchTransformersModel, SwitchTransformersModelYRJ, ExpertManager
import time
import os
import torch

ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cuda"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-samsum"
base_num = 8

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")
    model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj", device_map="auto", offload_folder="offload")

    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(0)  # Batch size 1

    decoder_input_ids = model._shift_right(decoder_input_ids)

    time_ = 0
    loop = 1
    for i in range(0, loop):
        torch.cuda.synchronize()  # Wait for all CUDA cores to finish their tasks
        start = time.time()
        with torch.no_grad():  # Avoid saving intermediate layer outputs
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        torch.cuda.synchronize()  # Wait for all CUDA cores to finish their tasks
        end = time.time()
        time_ += (end - start)/loop

    peak_memory = torch.cuda.max_memory_allocated()  # Get peak GPU memory usage
    print("Inference Time:", time_, "s")
    print("Peak GPU Memory Usage:", peak_memory / 1024 ** 2, "MB")  # Print peak GPU memory usage