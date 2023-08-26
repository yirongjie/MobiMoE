import torch
model_dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-8/pytorch_model.bin')
base_model_dict = {}
for key in model_dict:
    if 'experts' not in key.split('.'):
        base_model_dict[key] = model_dict[key]
    else:
        layer_dict = {key:model_dict[key]}
        torch.save(base_model_dict, os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-8-yrj/'+key+'.bin')

torch.save(base_model_dict, os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-8-yrj/base.bin')