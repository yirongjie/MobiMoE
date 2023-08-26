import torch
import copy
import pickle
import os

bit_num =8
model_dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-8-xsum/pytorch_model.bin')
t5_model_dict = {}
for key in model_dict:
     #   qcint8
    fp = model_dict[key].float().cpu()
    # scales = []
    # zero_points = []
    # for j in range(fp.shape[1]):
    #     ch_data = fp[:,j]
    #     scale_ = (ch_data.max() - ch_data.min())/2**bit_num
    #     zero_point_ = int((ch_data.max() + ch_data.min())/2)
    #     scales.append(scale_)
    #     zero_points.append(zero_point_)
    # Q_tensor = torch.quantize_per_channel(fp, torch.tensor(scales), torch.tensor(zero_points), 1, torch.qint8)

    scale_ = (float(torch.max(fp))-float(torch.min(fp)))/2**bit_num
    zero_point_ = int((float(torch.max(fp))+float(torch.min(fp)))/2)
    Q_tensor = torch.quantize_per_tensor(fp, scale=scale_, zero_point=zero_point_, dtype=torch.qint8)

    dqT = Q_tensor.dequantize()
    t5_model_dict[key] = dqT

# for key in t5_model_dict:
#     print(key)
torch.save(t5_model_dict, os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-8-xsum-int/'+str(bit_num) +'/pytorch_model.bin')

