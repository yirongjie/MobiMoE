import torch 
from transformers import AutoTokenizer, SwitchTransformersForConditionalGenerationYRJ, SwitchTransformersForConditionalGeneration, ExpertManager
import os
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 6
ExpertManager.FTpwd = "-samsum"
# ExpertManager.FTpwd = "-xsum"
tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
# tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8")
# model = SwitchTransformersForConditionalGeneration.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"11/models/switch-base-8")

# input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
# input_text = "cola sentence: The course is jumping well."
# input_text = "summarize: studies have shown that owning a dog is good for you"
# input_text = "Please answer the following question. What is the boiling point of Nitrogen?"
# input_text = "summarize: Peter and Elizabeth took a taxi to attend the night party in the city."
# input_text = "What station aired the Super Bowl?"
# input_text = "I can't believe Holly won't eat cabbage. What does not Helly eat?"
input_text = "Eric: MACHINE! Rob: That's so great! Eric: I know! And shows how Americans see Russian ;) Rob: And it's really funny! Eric: I know! I especially like the train part! Rob: Hahaha! No one talks to the machine like that! Eric: Is this his only stand-up? Rob: Idk. I'll check. Eric: Sure. Rob: Turns out no! There are some of his stand-ups on youtube. Eric: Gr8! I'll watch them now! Rob: Me too! Eric: MACHINE! Rob: MACHINE! Eric: TTYL? Rob: Sure :)"

input_ids = tokenizer(input_text, return_tensors="pt").input_ids
with torch.no_grad():  # Avoid saving intermediate layer outputs
    outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
# >>> <pad> <extra_id_0> man<extra_id_1> beer<extra_id_2> a<extra_id_3> salt<extra_id_4>.</s>





######################333333


# from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

# tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models//switch-base-8-samsum")
# model = SwitchTransformersForConditionalGeneration.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models//switch-base-8-samsum")

# input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))
# # >>> <pad> <extra_id_0> man<extra_id_1> beer<extra_id_2> a<extra_id_3> salt<extra_id_4>.</s>




