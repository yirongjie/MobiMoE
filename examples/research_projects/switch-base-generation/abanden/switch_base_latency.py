from datasets import load_dataset
import sys
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration,ExpertManager
import os
dataset = load_dataset("xsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
inputSens = dataset['test']
textt = "document"

# dataset = load_dataset("samsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
# inputSens = dataset['test']
# textt = "dialogue"
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-xsum"
# ExpertManager.FTpwd = "-samsum"
paramList = sys.argv
# ExpertManager.onlyBLk = int(sys.argv[1])
# ExpertManager.onlyEpt = int(sys.argv[2])


tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
# tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd)
# model = SwitchTransformersForConditionalGeneration.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd)
output_lns= []
reference_lns = []
ii = 0
for i, input in enumerate(inputSens):
    if i > 30:
        break
    # input_text = "summarize: "+input[textt]
    input_text = str(input[textt])
    if len(input_text.split(" "))< 300:
        ii = ii+1
        # print(ii,"   ------------------------------")
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)#, max_new_tokens=100)
        output_text = tokenizer.decode(outputs[0])
        print(ii, len(output_text.split(" ")))
