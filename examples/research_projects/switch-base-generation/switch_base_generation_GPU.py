import torch 
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration, ExpertManager
import os
import time
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cuda"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-samsum"
# ExpertManager.FTpwd = ""

ExpertManager.bwList = ['2_0', '3_5', '0_0', '0_5', '5_5', '4_0', '5_0', '4_5', '1_0', '4_3', '2_3', '3_3', '1_5', '3_2', '0_2', '0_3', '2_2', '4_2', '2_5', '3_0', '5_2', '1_2', '1_3', '5_3', '3_6', '4_6', '0_6', '2_6', '2_4', '3_4', '1_6', '5_6', '1_4', '5_1', '1_1', '3_1', '0_4', '0_1', '1_7', '4_4', '2_1', '4_1', '2_7', '5_7', '5_4', '4_7', '0_7', '3_7']
if __name__ == '__main__':
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




