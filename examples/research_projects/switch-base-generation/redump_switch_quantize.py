import torch
import pickle
from sklearn.mixture import GaussianMixture 
import numpy as np

def uint8_pack(intweight, bits):
    qweight = np.zeros((intweight.shape[0] // 8 * bits, 1), dtype=np.uint8)
    i = 0
    row = 0
    while row < qweight.shape[0]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (8 // bits)):
                qweight[row] |= intweight[j] << (bits * (j - i))
            i += 8 // bits
            row += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    qweight = qweight.astype(np.int8)
    # qweight = torch.from_numpy(qweight)
    return qweight

def gobo_quantize(fp, bits):
    weights = torch.Tensor([]) 
    # original dimension and pointer address to restore to
    weights = torch.cat([weights, torch.flatten(fp)])
    _weights = weights.reshape(-1, 1)
    gm = GaussianMixture(n_components=1, random_state=0).fit(_weights)
    scores = gm.score_samples(_weights)
    outliers = []
    _weights = _weights.numpy()
    for i in range(0, len(scores)):
        if scores[i] <= -4.0:
            outliers.append(_weights[i][0])
    mask = np.zeros(weights.shape, dtype=bool)
    mask[np.where(scores <= -4.0)] = True
    # return np.array(outliers), mask
    o_idx = mask
    g_group = weights[~o_idx]
    g_group = np.sort(g_group)
    bins = []
    n_bins = pow(2, bits)
    step = int(len(g_group)/n_bins)
    centroids = []
    # calculate centroids in G group
    for i in range(n_bins):
        start = i * step
        centroids.append(np.average(g_group[start: start + step]))
        bins.append(g_group[start])
        
    # boundary 
    bins.append(g_group[-1])
    centroids.append(-99999.0) 
    centroids = np.array(centroids)
    # assign quantized values
    quantized = np.digitize(weights, bins, right = True) -1 # return the idx of the centroids
    quantized[np.where(quantized==-1)]=0
    quantized[np.where(quantized==16)]=0
    load_weight = uint8_pack(quantized, bits)
    # a = quantized - 
    centroids = centroids[0:-1]

    return load_weight, centroids, weights[o_idx], o_idx


def quantize_channel(data, bit_num, channel_num=1):# ONLY 1
    quantized_data = torch.zeros(data.shape).type(torch.int8)
    scale = torch.zeros((1, data.shape[channel_num]))
    zero_point = torch.zeros((1, data.shape[channel_num]))
    # print(quantized_data.shape)
    for j in range(data.shape[channel_num]):
        ch_data = data[:,j]
        scale_ = (ch_data.max() - ch_data.min())/2**bit_num
        # zero_point_ = (ch_data.max() + ch_data.min())/2
        zero_point_ = int((ch_data.max() + ch_data.min())/2)
        quantized_data[:,j] = torch.round(((ch_data-torch.full_like(ch_data,zero_point_))/torch.full_like(ch_data,scale_))).type(torch.int8)
        scale[:,j] = scale_
        zero_point[:,j] = zero_point_

    quantized = torch.flatten(quantized_data)
    quantized = torch.clamp(quantized, -int(2**bit_num/2), int(2**bit_num/2)-1)
    quantized = quantized.numpy() + int(2**bit_num/2)
    # print(quantized.max(),quantized.min())
    # print(quantized.shape)
    load_weight = uint8_pack(quantized, bit_num)
    
    return load_weight,scale,zero_point#np.array,tensor, tensor

def quantize_channel_8(data, channel_num=1):# ONLY 1
    quantized_data = torch.zeros(data.shape).type(torch.int8)
    scale = torch.zeros((1, data.shape[channel_num]))
    zero_point = torch.zeros((1, data.shape[channel_num]))
    # print(quantized_data.shape)
    for j in range(data.shape[channel_num]):
        ch_data = data[:,j]
        scale_ = (ch_data.max() - ch_data.min())/2**8
        # zero_point_ = (ch_data.max() + ch_data.min())/2
        zero_point_ = int((ch_data.max() + ch_data.min())/2)
        quantized_data[:,j] = torch.round(((ch_data-torch.full_like(ch_data,zero_point_))/torch.full_like(ch_data,scale_))).type(torch.int8)
        scale[:,j] = scale_
        zero_point[:,j] = zero_point_ 
    return quantized_data, scale, zero_point#tensor,tensor, tensor

def quantize_tensor(data, bit_num):
    scale = (data.max() - data.min())/2**bit_num
    zero_point = 0#int( (data.max() + data.min())/2)
    quantized_data = torch.round(((data-torch.full_like(data,zero_point))/torch.full_like(data,scale)))
    # accumulated_error = data-(quantized_data*scale+zero_point)
    quantized_data = quantized_data.type(torch.int8)

    quantized = torch.flatten(quantized_data)
    quantized = torch.clamp(quantized, -int(2**bit_num/2), int(2**bit_num/2)-1)
    quantized = quantized.numpy() + int(2**bit_num/2)
    # print(quantized.max(),quantized.min())
    # print(quantized.shape)
    load_weight = uint8_pack(quantized, bit_num)
    
    return load_weight,scale,zero_point# np.array, float, int

def quantize_tensor_8(data):
    scale = (data.max() - data.min())/2**8
    zero_point =0#int( (data.max() + data.min())/2)
    quantized_data = torch.round(((data-torch.full_like(data,zero_point))/torch.full_like(data,scale)))
    # accumulated_error = data-(quantized_data*scale+zero_point)
    quantized_data = quantized_data.type(torch.int8)

    # print(quantized_data)

    return quantized_data, torch.tensor(scale) #tensor,tensor, tensor



import os
# mtype = "32"
# mtype = "16-samsum"
mtype = "8-samsum"
# mtype = "8-xsum"
# mtype = "8"

# quantize_type = "gobo"
quantize_type = "ic"
# quantize_type = "iccpu"
bit_num = 2

device_ = "cuda"  

dir_ = os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-'+mtype+'-yrj/'+quantize_type+str(bit_num)+'/'
if not os.path.isdir(dir_):
    os.mkdir(dir_)

model_dict = torch.load(os.path.dirname(os.path.realpath(__file__))+'/models/switch-base-'+mtype+'/pytorch_model.bin')
for key in model_dict:
    key_split = key.split('.')
    if 'experts' in key_split:
        fp = model_dict[key].float().cpu()

        if quantize_type == "gobo":
            #    gobo`
            load_weight, centroids, weights_o_idx, o_idx = gobo_quantize(fp, bit_num)
            dict_ = {"load_weight":load_weight, "centroids":centroids, "weights_o_idx":weights_o_idx, "o_idx":o_idx}

        elif quantize_type == "ic":
            #  ic
            if bit_num ==8:
                quant_x, scale, zerop = quantize_channel_8(fp)
                dict_ ={"weight":quant_x.to(device_), "scale":scale.to(device_), "zero_point": zerop.to(device_)}
            else:
                quant_x, scale, zerop = quantize_channel(fp, bit_num)
                dict_ ={"weight":quant_x, "scale":scale.to(device_), "zero_point": zerop.to(device_)}
        
        elif quantize_type == "iccpu":
            device_ = "cpu"
            #  ic
            if bit_num ==8:
                quant_x, scale, zerop = quantize_channel_8(fp)
                dict_ ={"weight":quant_x.to(device_), "scale":scale.to(device_), "zero_point": zerop.to(device_)}
            else:
                quant_x, scale, zerop = quantize_channel(fp, bit_num)
                dict_ ={"weight":quant_x, "scale":scale.to(device_), "zero_point": zerop.to(device_)}



        elif quantize_type == "it":
            # it
            if bit_num ==8:
                quant_x, scale = quantize_tensor_8(fp)
                dict_ ={"weight":quant_x.to(device_), "scale":scale.to(device_)}
            else:
                quant_x, scale, zerop = quantize_tensor(fp, bit_num)
                dict_ ={"weight":quant_x, "scale":scale.to(device_)}#float(scale)}#, "zero_point": zerop}


        f_save = open(dir_+key+'.bin', 'wb')
        pickle.dump(dict_, f_save)
        f_save.close()