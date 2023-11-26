import torch
import pickle
import numpy as np
import os
import time




fp_dir =os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-xsum-yrj/cpu/decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight.bin"
f_read = open(fp_dir, 'rb')
model = pickle.load(f_read)
fp = model["decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight"].float()#




# from sklearn.mixture import GaussianMixture 
# import numpy as np

# def pack(self, linear, scales, zeros, g_idx=None):
def uint32_pack(intweight, bits):
    qweight = np.zeros((intweight.shape[0] // 32 * bits, 1), dtype=np.uint32)
    i = 0
    row = 0
    while row < qweight.shape[0]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (32 // bits)):
                qweight[row] |= intweight[j] << (bits * (j - i))
            i += 32 // bits
            row += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    qweight = qweight.astype(np.int32)
    # qweight = torch.from_numpy(qweight)
    return qweight

#qweight[0] |= (intweight[j]& (2**bits-1)) << (bits * (j))

def int32_pack(intweight, bits):
    qweight = np.zeros((intweight.shape[0] // 32 * bits, 1), dtype=np.int32)
    i = 0
    row = 0
    while row < qweight.shape[0]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (32 // bits)):
                qweight[row] |= (intweight[j]& (2**bits-1))  << (bits * (j - i))
            i += 32 // bits
            row += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    qweight = qweight.astype(np.int32)
    # qweight = torch.from_numpy(qweight)
    return qweight

# def gobo_quantize(fp, bits):
#     weights = torch.Tensor([]) 
#     # original dimension and pointer address to restore to
#     weights = torch.cat([weights, torch.flatten(fp)])
#     _weights = weights.reshape(-1, 1)
#     gm = GaussianMixture(n_components=1, random_state=0).fit(_weights)
#     scores = gm.score_samples(_weights)
#     outliers = []
#     _weights = _weights.numpy()
#     for i in range(0, len(scores)):
#         if scores[i] <= -4.0:
#             outliers.append(_weights[i][0])
#     mask = np.zeros(weights.shape, dtype=bool)
#     mask[np.where(scores <= -4.0)] = True
#     # return np.array(outliers), mask
#     o_idx = mask


#     g_group = weights[~o_idx]
#     g_group = np.sort(g_group)
#     bins = []
#     n_bins = pow(2, bits)
#     step = int(len(g_group)/n_bins)
#     centroids = []
#     # calculate centroids in G group
#     for i in range(n_bins):
#         start = i * step
#         centroids.append(np.average(g_group[start: start + step]))
#         bins.append(g_group[start])
        
#     # boundary 
#     bins.append(g_group[-1])
#     centroids.append(-99999.0) 
#     centroids = np.array(centroids)
#     # assign quantized values
#     quantized = np.digitize(weights, bins, right = True) -1 # return the idx of the centroids
#     quantized[np.where(quantized==-1)]=0
#     quantized[np.where(quantized==16)]=0
#     load_weight = uint32_pack(quantized, bits)
#     # a = quantized - 
#     centroids = centroids[0:-1]

#     # print(quantized,  quantized.shape)
#     # print(centroids,  centroids.shape)
#     # print(weights[o_idx], weights[o_idx].shape)
#     # print(o_idx)
#     return load_weight, centroids, weights[o_idx], o_idx

def gobo_dequantize(bits, load_weight, centroids, weights_o_idx, o_idx, fp_size):
    ss = time.time()
    bis_numM = (2**bits-1)
    out = []
    for j in range(0, 0 + (8 // bits)):
        in32_ = ((load_weight>> (bits * (j))) & bis_numM )
        out.append(in32_)
    # ee = time.time()
    quantized_ = np.concatenate(out, axis = 1)
    # eee = time.time()
    quantized_ = quantized_.flatten()
    # eeee = time.time()
    ####### dequantize ############
    timeA = time.time()
    new_weights = centroids[quantized_]
    timeB = time.time()
    new_weights[o_idx] = weights_o_idx
    startT = time.time()
    this_module = torch.from_numpy(new_weights).float().view(fp_size)
    endT = time.time()
    # print("4bit2int         ",  ee - ss)
    # print("4bit2int concat  ",  eee - ee)
    # print("4bit2int flatten ",  eeee - eee)
    print("centroids        ", timeB-timeA)
    print("o_idx            ", startT-timeB)
    # print("dequantize       ", endT-startT)
    # print("total            ", endT-ss)


    print("totensor       ", timeA-ss)
    print("dequantize       ", endT-timeA)
    return this_module


def quantize_tensor(data, bit_num):
    scale = (data.max() - data.min())/2**bit_num
    zero_point = (data.max() + data.min())/2
    quantized_data = torch.round(((data-torch.full_like(data,zero_point))/torch.full_like(data,scale)))
    # accumulated_error = data-(quantized_data*scale+zero_point)
    quantized_data = quantized_data.type(torch.int8)
    return quantized_data,scale,zero_point

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

    quantized = torch.flatten(quantized_data).numpy()
    # print(quantized)
    load_weight = int32_pack(quantized, bit_num)
    
    return load_weight,scale,zero_point

def dequantize_channel(load_weight,scale,zero_point, bits, fp_size):

    ss = time.time()
    bis_numM = (2**bits-1)
    out = []
    for j in range(0, 0 + (32 // bits)):
        tmp_ = ((load_weight>> (bits * (j))) & bis_numM)
        in32_ = ((tmp_| (tmp_>> (bits -1)) * 0xfffffff0))#.astype(np.int32))
        # in32_ = ((load_weight>> (bits * (j))) & bis_numM )
        out.append(in32_)
    # ee = time.time()
    quantized_ = np.concatenate(out, axis = 1)
    # eee = time.time()
    quantized_ = quantized_.flatten()
    # eeee = time.time()    
    this_module = torch.from_numpy(quantized_).float().view(fp_size)
    #dequantize
    eeee1 = time.time()    
    result = this_module*scale+zero_point
    eeee2 = time.time()    
    # print("4bit2int         ",  ee - ss)
    # print("4bit2int concat  ",  eee - ee)
    # print("4bit2int flatten ",  eeee - eee)
    # print("totensor         ",  eeee1 - eeee)


    print("totensor         ",  eeee1 - ss)
    print("dequantize       ",  eeee2 - eeee1)

    return result



print("                     float")
st = time.time()
f_read = open(fp_dir, 'rb')
model = pickle.load(f_read)
et = time.time()
fp = model["decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight"].float()#
end_time = time.time()
cudatensor = fp.cuda()
end_c_time = time.time()
print("loadcpu:", et-st , "+", end_time-et, "=", end_time-st)
print("tocuda", end_c_time - end_time)


print("                     int8")
st = time.time()
in_dir =os.path.dirname(os.path.realpath(__file__))+"/testquant/tmp8.bin"#encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight.bin"
f_read = open(in_dir, 'rb')
model = pickle.load(f_read)
et = time.time()
in_ = model["decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight"].dequantize()#
end_time = time.time()
cudatensor = in_.cuda()
end_c_time = time.time()
print("loadcpu:", et-st , "+", end_time-et, "=", end_time-st)
print("tocuda", end_c_time - end_time)


print("                     gobo 4")
# load_weight, centroids, weights_o_idx, o_idx = gobo_quantize(fp, 4)
# dict_ = {"load_weight":load_weight, "centroids":centroids, "weights_o_idx":weights_o_idx, "o_idx":o_idx}
# f_save = open(os.path.dirname(os.path.realpath(__file__))+'/testquant/tmp4.bin', 'wb')
# pickle.dump(dict_, f_save)
# f_save.close()

st = time.time()
f_read = open(os.path.dirname(os.path.realpath(__file__))+'/testquant/tmp4.bin', 'rb')
load_ = pickle.load(f_read)
et = time.time()
this_module = gobo_dequantize(4, load_["load_weight"], load_["centroids"], load_["weights_o_idx"], load_["o_idx"], fp.size())
end_time = time.time()
cudatensor = this_module.cuda()
end_c_time = time.time()
print("dequantize:",  et-st , "+", end_time-et, "=", end_time-st)
print("tocuda", end_c_time - end_time)

# # print((this_module-fp).abs().max())
# # print(this_module.shape)




print("                     linear 4")
# quant_x, scale, zerop = quantize_channel(fp, 4)
# dict_ ={"weight":quant_x, "scale":scale, "zero_point": zerop}
# f_save = open(os.path.dirname(os.path.realpath(__file__))+'/testquant/tmpc4.bin', 'wb')
# pickle.dump(dict_, f_save)
# f_save.close()

st = time.time()
f_read = open(os.path.dirname(os.path.realpath(__file__))+'/testquant/tmpc4.bin', 'rb')
wi_dict = pickle.load(f_read)
et = time.time()
this_module = dequantize_channel(wi_dict["weight"], wi_dict['scale'], wi_dict["zero_point"], 4, fp.size())
end_time = time.time()
cudatensor = this_module.cuda()
end_c_time = time.time()
print("dequantize:", et-st , "+", end_time-et, "=", end_time-st)
print("tocuda", end_c_time - end_time)


