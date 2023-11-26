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

dataset_name = "samsum"

dataset = load_dataset(dataset_name, cache_dir=os.path.dirname(os.path.realpath(__file__))+'/datasets')
inputSens = dataset['test']
textt = "dialogue"
print(inputSens.__len__())


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
ExpertManager.FTpwd = "-" + dataset_name
base_num = 8



def get_rouge2(model, endCnt= 200, inputSens= inputSens):
    output_lns= []
    reference_lns = []
    sum_cont= 0
    for i, input in enumerate(inputSens):
        if sum_cont >= endCnt:
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
    # print(result)
    return result["rouge2"]

# get from ./experts/summary_metric.py
bwList_pool =['2_0', '3_5', '0_0', '0_5', '5_5', '4_0', '5_0', '4_5', '1_0', '4_3', '2_3', '3_3', '1_5', '3_2', '0_2', '0_3', '2_2', '4_2', '2_5', '3_0', '5_2', '1_2', '1_3', '5_3', '3_6', '4_6', '0_6', '2_6', '2_4', '3_4', '1_6', '5_6', '1_4', '5_1', '1_1', '3_1', '0_4', '0_1', '1_7', '4_4', '2_1', '4_1', '2_7', '5_7', '5_4', '4_7', '0_7', '3_7']
exp_size = bwList_pool.__len__()

def binary_search_bw_len(min_bw_len, max_bw_len):
    count = 0
    while min_bw_len < max_bw_len:
        count += 1
        mid_bw_len = (min_bw_len + max_bw_len) // 2
        ExpertManager.bwList = bwList_pool[:mid_bw_len]
        relative_rouge2 = get_rouge2(model, endCnt=outline_endCnt, inputSens=inputSens) / origin_rouge2
        print(count,"|4-bit:", min_bw_len, "/", max_bw_len, "   |relative rouge2",relative_rouge2)
        if relative_rouge2 >= 0.95:
            max_bw_len = mid_bw_len
            return min_bw_len
        else:
            min_bw_len = mid_bw_len + 1
    return min_bw_len

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")
    model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj", device_map="auto", offload_folder="offload")
    origin_moael = SwitchTransformersForConditionalGeneration.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-"+str(base_num)+ExpertManager.FTpwd, device_map="auto", offload_folder="offload")

    outline_endCnt = 200
    
    origin_rouge2 = get_rouge2(origin_moael, endCnt= outline_endCnt, inputSens= inputSens)
    print("origin rouge2:", origin_rouge2)    

    
    ExpertManager.bwList = bwList_pool
    relative_rouge2 = get_rouge2(model, endCnt=outline_endCnt, inputSens=inputSens) / origin_rouge2
    print("relative rouge2",relative_rouge2)
    if relative_rouge2 < 0.95:
        exp_size = exp_size
        exit()

    bw_len = binary_search_bw_len(0, exp_size)

    print("bw_len:", bw_len)
    print("bwList:", bwList_pool[:bw_len])
    ExpertManager.bwList = bwList_pool[:bw_len]

    # bw_len = exp_size
    # ExpertManager.bwList = bwList_pool[:bw_len]
    # relative_rouge2 = get_rouge2(model, endCnt= outline_endCnt, inputSens= inputSens)/origin_rouge2
    # print("relative rouge2",relative_rouge2)

    # while bw_len < exp_size:
    #     mid_bw_len = (bw_len + exp_size) // 2
    #     ExpertManager.bwList = bwList_pool[:mid_bw_len]
    #     relative_rouge2 = get_rouge2(model, endCnt=outline_endCnt, inputSens=inputSens) / origin_rouge2
    #     print("relative rouge2",relative_rouge2)
    #     if relative_rouge2 > 0.95:
    #         exp_size = mid_bw_len
    #     else:
    #         bw_len = mid_bw_len + 1
