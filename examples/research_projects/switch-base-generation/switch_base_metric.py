from rouge_score import rouge_scorer, scoring
import torch

def cal_rouge2(sen1,sen2):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    scores = scorer.score(sen1,sen2)['rouge2']
    return scores[0], scores[1], scores[2]


def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}

import sys
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration,ExpertManager
import os
from datasets import load_dataset
# dataset = load_dataset("xsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
# inputSens = dataset['test']
# textt = "document"

dataset = load_dataset("samsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
inputSens = dataset['test']
textt = "dialogue"


# paramList = sys.argv
# ExpertManager.onlyBLk = int(sys.argv[1])
# ExpertManager.onlyEpt = int(sys.argv[2])

ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
# ExpertManager.Device = "cpu"
# ExpertManager.CacheLimit = 12
# # ExpertManager.FTpwd = "-xsum"
# ExpertManager.FTpwd = "-samsum"
# tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
# model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")

ExpertManager.Device = "cuda"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-samsum"
# ExpertManager.FTpwd = ""
tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj", device_map="auto", offload_folder="offload")

output_lns= []
reference_lns = []
sum_cont= 0
for i, input in enumerate(inputSens):
    if sum_cont >= 200:
        break
    # input_text = "summarize: "+input[textt]
    input_text = input[textt]
    if len(input_text.split(" "))< 300:
        label_text = input["summary"]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda:0')
        with torch.no_grad():  # Avoid saving intermediate layer outputs
            outputs = model.generate(input_ids, max_new_tokens=100)
        output_text = tokenizer.decode(outputs[0])
        output_text=output_text.replace("<pad> ","")
        output_text=output_text.replace("</s>","")
        output_lns.append(output_text)
        reference_lns.append(label_text)
        sum_cont += 1
result = calculate_rouge(output_lns, reference_lns, use_stemmer=True)
print(result)
print("total cnt:", sum_cont)
