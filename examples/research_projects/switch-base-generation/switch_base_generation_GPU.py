import torch 
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration, ExpertManager
import os
import time
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cuda"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-samsum"
# ExpertManager.FTpwd = ""
tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj", device_map="auto", offload_folder="offload")


input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."

input_ids = tokenizer(input_text, return_tensors="pt").input_ids
input_ids = input_ids.to('cuda:0')
torch.cuda.synchronize()  # Wait for all CUDA cores to finish their tasks
startT = time.time()
with torch.no_grad():  # Avoid saving intermediate layer outputs
    outputs = model.generate(input_ids)
torch.cuda.synchronize()  # Wait for all CUDA cores to finish their tasks
endT = time.time()
print(tokenizer.decode(outputs[0]))
print(endT-startT)
# >>> <pad> <extra_id_0> man<extra_id_1> beer<extra_id_2> a<extra_id_3> salt<extra_id_4>.</s>





######################


# from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

# tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models//switch-base-8-samsum")
# model = SwitchTransformersForConditionalGeneration.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models//switch-base-8-samsum")

# input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))
# # >>> <pad> <extra_id_0> man<extra_id_1> beer<extra_id_2> a<extra_id_3> salt<extra_id_4>.</s>




