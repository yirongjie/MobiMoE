from datasets import load_dataset
import sys
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration,ExpertManager,SwitchTransformersModelYRJ
import os
dataset = load_dataset("xsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
inputSens = dataset['test']
textt = "document"
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 1
ExpertManager.FTpwd = ""
# ExpertManager.FTpwd = "-xsum"
paramList = sys.argv


tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-32"+ExpertManager.FTpwd+"-yrj")
model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-32"+ExpertManager.FTpwd+"-yrj")
output_lns= []
reference_lns = []
ii = 0
for i, input in enumerate(inputSens):
    if i > 30:
        break
    if i < 1:
        continue
    input_text = str(input[textt])
    # input_text=input_text.replace("\n"," ")
    if len(input_text.split(" "))< 300:

        # print("\"", input_text, "\",")
        # input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        # outputs = model.generate(input_ids)#, max_new_tokens=100)
        # output_text = tokenizer.decode(outputs[0])
        ii = ii+1
        print(ii,"   ------------------------------")
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # Batch size 1
        decoder_input_ids = tokenizer("<\s>", return_tensors="pt").input_ids  # Batch size 1
        decoder_input_ids = model._shift_right(decoder_input_ids)
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
