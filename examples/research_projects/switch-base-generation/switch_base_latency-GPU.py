from datasets import load_dataset
import sys
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration,ExpertManager,SwitchTransformersModelYRJ
import os
# dataset = load_dataset("xsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
# inputSens = dataset['test']
# textt = "document"
# dataset = load_dataset("samsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
dataset = load_dataset(os.path.dirname(os.path.realpath(__file__))+"/datasets/samsum")#, cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
inputSens = dataset['test']
textt = "dialogue"

ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cuda"
ExpertManager.CacheLimit = 1
# ExpertManager.FTpwd = ""
# ExpertManager.FTpwd = "-xsum"
ExpertManager.FTpwd = "-samsum"
paramList = sys.argv


tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-16"+ExpertManager.FTpwd+"-yrj")
model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-16"+ExpertManager.FTpwd+"-yrj", device_map="auto", offload_folder="offload")
output_lns= []
reference_lns = []
ii = 0
for i, input in enumerate(inputSens):
    if i > 30:
        break
    input_text = str(input[textt])
    if len(input_text.split(" "))< 300:
        # input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        # outputs = model.generate(input_ids)#, max_new_tokens=100)
        # output_text = tokenizer.decode(outputs[0])
        ii = ii+1
        print(ii,"   ------------------------------")
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # Batch size 1
        decoder_input_ids = tokenizer("</s>", return_tensors="pt").input_ids.to(0)  # Batch size 1
        decoder_input_ids = model._shift_right(decoder_input_ids)
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
