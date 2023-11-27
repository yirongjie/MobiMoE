from rouge_score import rouge_scorer, scoring
def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}

import os
from datasets import load_dataset
dataset = load_dataset("xsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
inputSens = dataset['test']
textt = "document"

# dataset = load_dataset("samsum", cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
# inputSens = dataset['test']
# textt = "dialogue"
import sys
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration,ExpertManager
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-xsum"
# ExpertManager.FTpwd = "-samsum"
# paramList = sys.argv
# ExpertManager.onlyBLk = 0#int(sys.argv[1])

for jEpt in range(6):
    ExpertManager.onlyBLk = jEpt#int(sys.argv[1])
    for iEpt in range(8):
        ExpertManager.onlyEpt = iEpt#int(sys.argv[2])
        #/home/ey/personal/program/MoE/huggingface/transformers/examples/research_projects/switch-base-generation/switch_base_experience.py
        tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
        model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
        # tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd)
        # model = SwitchTransformersForConditionalGeneration.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd)
        output_lns= []
        reference_lns = []
        for i, input in enumerate(inputSens):
            if i > 1000:
                break
            # input_text = "summarize: "+input[textt]
            input_text = input[textt]
            if len(input_text.split(" "))< 300:
                label_text = input["summary"]
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_new_tokens=100)
                output_text = tokenizer.decode(outputs[0])
                output_text=output_text.replace("<pad> ","")
                output_text=output_text.replace("</s>","")
                output_lns.append(output_text)
                reference_lns.append(label_text)
        result = calculate_rouge(output_lns, reference_lns, use_stemmer=True)
        print(ExpertManager.onlyBLk, ExpertManager.onlyEpt, ":  ", result)
