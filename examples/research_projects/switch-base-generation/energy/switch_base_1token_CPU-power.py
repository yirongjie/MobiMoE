
###
# 在执行改脚本前，需要先执行以下命令
# sudo python power_listener.py




from transformers import AutoTokenizer, SwitchTransformersModel, SwitchTransformersModelYRJ, ExpertManager
import time
import os
import torch
from p_usg import Timer
from p_usg import PowerUsage

ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__)) + "/../"
ExpertManager.Device = "cpu"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-samsum"
base_num = 8

def run():
    time_ = 0
    loop = 1
    for i in range(0, loop):
        start = time.time()
        with torch.no_grad():  # Avoid saving intermediate layer outputs
            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        end = time.time()
        time_ += (end - start)/loop

    print("Inference Time:", time_, "s")

# ExpertManager.bwList = ['2_0', '3_5', '0_0', '0_5', '5_5', '4_0', '5_0', '4_5', '1_0', '4_3', '2_3', '3_3', '1_5', '3_2', '0_2', '0_3', '2_2', '4_2', '2_5', '3_0', '5_2', '1_2', '1_3', '5_3', '3_6', '4_6', '0_6', '2_6', '2_4', '3_4', '1_6', '5_6', '1_4', '5_1', '1_1', '3_1', '0_4', '0_1', '1_7', '4_4', '2_1', '4_1', '2_7', '5_7', '5_4', '4_7', '0_7', '3_7']
ExpertManager.bwList = []
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/../models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")
    model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/../models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")

    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("", return_tensors="pt").input_ids  # Batch size 1

    decoder_input_ids = model._shift_right(decoder_input_ids)

    power_usage = PowerUsage()
    power_usage.analyze_start()
    # ----------------------
    # xxx 某一段待分析的代码
    run()
    # ----------------------
    time_used, power_usage_gpu, power_usage_cpu = power_usage.analyze_end()
    print(f'time_used: {time_used}')
    print(f'power_usage_gpu: {power_usage_gpu}')
    print(f'power_usage_cpu: {power_usage_cpu}')


