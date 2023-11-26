import torch
import copy
import pickle
import os

# mtype = "32"
mtype = "16-samsum"
# mtype = "8-samsum"
# mtype = "8-xsum"
# mtype = "8"
model_dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-'+mtype+'/pytorch_model.bin')
dir_cpu= os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-'+mtype+'-yrj/cpu/'
if not os.path.isdir(dir_cpu):
    os.mkdir(dir_cpu)
dir_gpu= os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-'+mtype+'-yrj/cuda/'
if not os.path.isdir(dir_gpu):
    os.mkdir(dir_gpu)
t5_model_dict = {}
for key in model_dict:
    key_split = key.split('.')
    if 'experts' in key_split:
        # print(model_dict[key].device)
        dict_ ={key:model_dict[key].cpu()}
        f_save = open(dir_cpu+key+'.bin', 'wb')
        pickle.dump(dict_, f_save)
        f_save.close()

        dict_ ={key:model_dict[key].cuda()}
        f_save = open(dir_gpu+key+'.bin', 'wb')
        pickle.dump(dict_, f_save)
        f_save.close()
 
    else:
        t5_model_dict[key] = model_dict[key]

# for key in t5_model_dict:
#     print(key)
torch.save(t5_model_dict, os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-'+mtype+'-yrj/pytorch_model.bin')
