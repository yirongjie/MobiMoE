from transformers import AutoTokenizer, SwitchTransformersModel, SwitchTransformersModelYRJ, ExpertManager
import time
import os
import torch
import subprocess
import pyRAPL 
from pyRAPL import Measurement
pyRAPL.setup()

ExpertManager.Pwd = os.path.dirname(os.path.realpath(__file__)) + "/../"
ExpertManager.Device = "cuda"
ExpertManager.CacheLimit = 12
ExpertManager.FTpwd = "-samsum"
base_num = 8



ExpertManager.bwList = ['2_0', '3_5', '0_0', '0_5', '5_5', '4_0', '5_0', '4_5', '1_0', '4_3', '2_3', '3_3', '1_5', '3_2', '0_2', '0_3', '2_2', '4_2', '2_5', '3_0', '5_2', '1_2', '1_3', '5_3', '3_6', '4_6', '0_6', '2_6', '2_4', '3_4', '1_6', '5_6', '1_4', '5_1', '1_1', '3_1', '0_4', '0_1', '1_7', '4_4', '2_1', '4_1', '2_7', '5_7', '5_4', '4_7', '0_7', '3_7']
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/../models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj")
    model = SwitchTransformersModelYRJ.from_pretrained(os.path.dirname(os.path.realpath(__file__))+"/../models/switch-base-"+str(base_num)+ExpertManager.FTpwd+"-yrj", device_map="auto", offload_folder="offload")

    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(0)  # Batch size 1

    decoder_input_ids = model._shift_right(decoder_input_ids)

    time_ = 0
    loop = 1
    # Get the initial GPU power usage

    initial_gpu_power = subprocess.check_output("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", shell=True)

    # Start measuring CPU power usage
    measurement = Measurement('MyMeasurement')
    with measurement:
        # Your code here
        time_ = 0
        loop = 1
        for i in range(0, loop):
            torch.cuda.synchronize()  # Wait for all CUDA cores to finish their tasks
            start = time.time()
            with torch.no_grad():  # Avoid saving intermediate layer outputs
                outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            torch.cuda.synchronize()  # Wait for all CUDA cores to finish their tasks
            end = time.time()
            time_ += (end - start)/loop
    print("Inference Time:", time_, "s")


    # Get the final GPU power usage
    final_gpu_power = subprocess.check_output("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", shell=True)

    # Calculate the energy consumption
    gpu_energy_consumption = float(final_gpu_power) - float(initial_gpu_power)
    # Get the energy consumption
    cpu_energy_consumption = measurement.result.pkg  # in microjoules

    print("GPU Power Consumption:",  float(final_gpu_power), "W")
    
    print("GPU Energy Consumption:", float(final_gpu_power) * time_, "J")
    # Get the energy consumption
    cpu_energy_consumption = sum(measurement.result.pkg) / len(measurement.result.pkg)  # in microjoules
    print("CPU Energy Consumption:", cpu_energy_consumption / 1000000, "J")  # Convert to joules

    peak_memory = torch.cuda.max_memory_allocated()  # Get peak GPU memory usage
    print("Inference Time:", time_, "s")
    print("Peak GPU Memory Usage:", peak_memory / 1024 ** 2, "MB")  # Print peak GPU memory usage
    print("Peak GPU Memory Usage:", peak_memory / 1024 ** 3, "GB")  # Print peak GPU memory usage