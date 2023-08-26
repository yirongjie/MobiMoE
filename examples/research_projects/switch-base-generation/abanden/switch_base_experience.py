from transformers import SwitchTransformersModel, SwitchTransformersModelYRJ, ExpertManager, AutoModel
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration, ExpertManager
# from transformers.models.switch_transformers import LoadFunc, globalMultiEvent, globalMultiThreadRun, globalMultiParam, globalEM, globalMultiLock
# import multiprocessing
import torch
import pickle
import time
import difflib
import os
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-xsum"


# if __name__ == '__main__':
#     tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj")
#     model = AutoModel.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj")
#     print(model)
#     quantization_bit = 4
#     print(f"Quantized to {quantization_bit} bit")
#     model = model.quantize(quantization_bit)

from datasets import load_dataset
# from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, SwitchTransformersTop1Router

dataset = load_dataset("glue", name="cola", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
# dataset = load_dataset("wikitext", 'wikitext-103-v1', cache_dir='./datasets')
# print(dataset)

# dataset = load_dataset("xsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
# exit()
inputSens = dataset['test']['sentence'][0:200]

# for sen in dataset['test']['text']:
#     print(len(sen))
model_q = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj")
tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8")
model = SwitchTransformersForConditionalGeneration.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8")
lst = []
for i, input_text in enumerate(inputSens):
    if len(input_text) > 0:
        # input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
        # input_text = "A <extra_id_0> walks into a bar ."
        # input_text = "summarize:"+input_text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        sen = tokenizer.decode(outputs[0])
        outputs_q = model_q.generate(input_ids)
        sen_q = tokenizer.decode(outputs_q[0])
        # print(sen, type(sen))
        # print(difflib.SequenceMatcher(None, sen, sen_q).quick_ratio())
        lst.append(difflib.SequenceMatcher(None, sen, sen_q).quick_ratio())
        # print("-----------------")
        if (i+1)%5 == 0:
            print(i+1, sum(lst)/len(lst))
            # print(input_text)
            # print(sen)


print(sum(lst)/len(lst))

