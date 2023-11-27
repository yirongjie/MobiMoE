import torch
import copy
model_dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-8/pytorch_model.bin')
t5_model_dict = {}
for key in model_dict:
    # mlp  -> DenseReluDense
    # mlp.exports.expert_7 -> DenseReluDense
    # [remove] mlp.exports.expert_0~6    mlp.router.classifier
    # [remove] encoder.embed_tokens.weight, decoder.embed_tokens.weight, lm_head.weight
    key_split = key.split('.')
    if 'mlp' in key_split and 'experts' not in key_split and 'classifier' not in key_split:# mlp  -> DenseReluDense
        t5_key=copy.deepcopy(key)
        t5_key = t5_key.replace('mlp', 'DenseReluDense')
        t5_model_dict[t5_key] = model_dict[key]
        # print(t5_key)
    elif 'mlp' in key_split and 'expert_7' in key_split:# mlp.exports.expert_7 -> DenseReluDense
        t5_key=copy.deepcopy(key)
        t5_key = t5_key.replace('mlp.experts.expert_7', 'DenseReluDense')
        t5_model_dict[t5_key] = model_dict[key]
        # print(t5_key)
    elif 'mlp' in key_split and 'experts' in key_split:# [remove] mlp.exports.expert_0~6 
        continue 
    elif 'mlp' in key_split and 'classifier' in key_split:# [remove] mlp.router.classifier 
        continue 
    elif 'encoder.embed_tokens.weight' in key_split or 'decoder.embed_tokens.weight' in key_split:# [remove] encoder.embed_tokens.weight, decoder.embed_tokens.weight
        continue  
    elif 'lm_head.weight'  ==  key: # [remove] lm_head.weight
        continue
    else:
        t5_model_dict[key] = model_dict[key]

# for key in t5_model_dict:
#     print(key)
torch.save(t5_model_dict, os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-8-t5/pytorch_model.bin')