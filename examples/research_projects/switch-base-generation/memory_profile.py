from memory_profiler import profile


from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration, SwitchTransformersModel, SwitchTransformersModelYRJ, SwitchTransformersForConditionalGenerationYRJ, ExpertManager
import os
ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__))
ExpertManager.CacheLimit = 10
ExpertManager.FTpwd = "-xsum"
ExpertManager.FTpwd = ""
ExpertManager.FTpwd = "-samsum"
import torch


@profile
def switch_genrate():
    
    ExpertManager.Device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
    model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
    #model = SwitchTransformersModel.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"")
    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
    decoder_input_ids = model._shift_right(decoder_input_ids)
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    # ExpertManager.Device = "cuda"
    # tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj")
    # model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8"+ExpertManager.FTpwd+"-yrj", device_map="auto", offload_folder="offload")
    # input_ids = tokenizer(
    #     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    # ).input_ids  # Batch size 1
    # decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids.to(0)  # Batch size 1
    # decoder_input_ids = model._shift_right(decoder_input_ids)
    # outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


    # tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj")
    # model = SwitchTransformersForConditionalGenerationYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj")
    # input_text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # outputs = model.generate(input_ids)



switch_genrate()
