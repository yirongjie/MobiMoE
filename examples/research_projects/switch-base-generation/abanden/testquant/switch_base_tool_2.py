import torch
import pickle
import numpy as np
import os


import time


# def pack(self, linear, scales, zeros, g_idx=None):
def pack(intweight, bits):
    # self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

    # scales = scales.t().contiguous()
    # zeros = zeros.t().contiguous()
    # scale_zeros = zeros * scales
    # self.scales = scales.clone().half()
    # if linear.bias is not None:
    #     self.bias = linear.bias.clone().half()

    # intweight = []
    # for idx in range(768):
    #     intweight.append(torch.zeros(3072).to(torch.int)[:, None])
    # intweight = torch.cat(intweight, dim=1)
    # intweight = intweight.t().contiguous()
    # intweight = intweight.numpy().astype(np.uint32)
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

    # zeros -= 1
    # zeros = zeros.numpy().astype(np.uint32)
    # qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
    # i = 0
    # col = 0
    # while col < qzeros.shape[1]:
    #     if self.bits in [2, 4, 8]:
    #         for j in range(i, i + (32 // self.bits)):
    #             qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
    #         i += 32 // self.bits
    #         col += 1
    #     else:
    #         raise NotImplementedError("Only 2,4,8 bits are supported.")

    # qzeros = qzeros.astype(np.int32)
    # self.qzeros = torch.from_numpy(qzeros)


fp_dir = "/home/ey/personal/program/MoE/huggingface//transformers/examples/research_projects/switch-base-generation/models/switch-base-8-xsum-yrj/cpu/decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight.bin"
f_read = open(fp_dir, 'rb')
model = pickle.load(f_read)
fp = model["decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight"].float()

bit_num = 4

# scales = []
# zero_points = []
# for j in range(fp.shape[1]):
#     ch_data = fp[:,j]
#     scale_ = (ch_data.max() - ch_data.min())/2**bit_num
#     zero_point_ = int((ch_data.max() + ch_data.min())/2)
#     scales.append(scale_)
#     zero_points.append(zero_point_)
# # print(fp.cpu().device)
# Q_tensor = torch.quantize_per_channel(fp, torch.tensor(scales), torch.tensor(zero_points), 1, torch.qint8)
# dict_ ={"w":Q_tensor.cpu()}


# dict_ ={"w":fp.type(torch.int8)}
# f_save = open(os.path.dirname(os.path.realpath(__file__))+'/testquant/tmp_8.bin', 'wb')
# pickle.dump(dict_, f_save)
# f_save.close()



# qmod = pack(bit_num)
# dict_ ={"w":qmod}
# f_save = open(os.path.dirname(os.path.realpath(__file__))+'/testquant/tmp_q.bin', 'wb')
# pickle.dump(dict_, f_save)
# f_save.close()












############## signed
bits = 4
qweight = np.zeros((1), dtype=np.uint32)
# intweight = [1,1,1,1,1,1,1,1]
# intweight = [1,2,3,4,5,6,7,0]
# intweight = [-2,-1,-3,-1,-1,-1,-1,-1]
intweight = [-3,-3,-3,-3,-3,-3,8,-8]
for j in range(0, 0 + (32 // bits)):
    qweight[0] |= (intweight[j]& (2**bits-1)) << (bits * (j))

print(qweight)
print(bin(qweight[0]))

for iw in intweight:
    print(bin(iw & (2**bits-1)), type(iw),bin((2**bits-1)) )


st = time.time()
out = []
for j in range(0, 0 + (32 // bits)):
    # # out.append((qweight[0]>> (bits * (j))) & (2**bits-1))
    # print(bin(((qweight[0]>> (bits * (j))) & (2**bits-1) )>> (bits -1)))
    tmp_ = ((qweight[0]>> (bits * (j))) & (2**bits-1) )
    sign = (tmp_>> (bits -1))
    in32_ = ((tmp_| sign * 0xfffffff0).astype(np.int32))
    out.append((in32_))

et = time.time()
print(et - st)


for iw in out:
    print(iw,  type(iw))#, 0xfffffff0, 2**32-1)



#################  unsigned

bits = 4
qweight = np.zeros((1), dtype=np.uint32)
intweight = [1,2,3,4,5,15,7,0]
for j in range(0, 0 + (32 // bits)):
    qweight[0] |= intweight[j]<< (bits * (j))

print(qweight)
print(bin(qweight[0]))

for iw in intweight:
    print(bin(iw), type(iw) )


st = time.time()
out = []
for j in range(0, 0 + (32 // bits)):
    in32_ = ((qweight[0]>> (bits * (j))) & (2**bits-1) ).astype(np.float32)
    out.append((in32_))

et = time.time()
print(et - st)


for iw in out:
    print(iw,  type(iw))#, 0xfffffff0, 2**32-1)



# st = time.time()
# fp_dir = "/home/ey/personal/program/MoE/huggingface//transformers/examples/research_projects/switch-base-generation/models/switch-base-8-xsum-yrj/qcint8/decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight.bin"#encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight.bin"
# f_read = open(fp_dir, 'rb')
# model = pickle.load(f_read)
# it_= model["decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight"]#
# et = time.time()

# it_4 = it_.int_repr()
# print(it_4, it_4.max(), it_4.min())
# print(et - st)





st = time.time()
fp_dir = "/home/ey/personal/program/MoE/huggingface//transformers/examples/research_projects/switch-base-generation/models/switch-base-8-xsum-yrj/cpu/decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight.bin"#encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight.bin"
f_read = open(fp_dir, 'rb')
model = pickle.load(f_read)
fp = model["decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight"].float()#
et = time.time()
print(et - st)


from sklearn.mixture import GaussianMixture 
import numpy as np

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
    # a = quantized - 

    print(quantized,  quantized.shape)
    print(weights[o_idx], weights[o_idx].shape)
    print(o_idx)
    # m = weights[o_idx]

    load_weight = pack(quantized, bits)
    # print(load_weight)

    out = []
    for qweight_ in load_weight:
        # print(qweight_[0])
        for j in range(0, 0 + (32 // bits)):
            in32_ = ((qweight_[0]>> (bits * (j))) & (2**bits-1) ).astype(np.int32)
            out.append((in32_))

    quantized_ = np.array(out)
    



    print(centroids[0], centroids[-2])
    start = time.time()
    n_bins = pow(2, bits)
    step = int(len(g_group)/n_bins)
    centroids = []
    # calculate centroids in G group
    for i in range(n_bins):
        start = i * step
        centroids.append(np.average(g_group[start: start + step]))
    centroids = np.array(centroids)
    new_weights = centroids[quantized_]
    # recover corresponding outlier weights
    new_weights[o_idx] = weights[o_idx]
    end = time.time()
    # # sanity check manually patch some values...
    # for idx,d in enumerate(new_weights):
    #     if d < -100.0:
    #         # there are values that are not in outlier idx 
    #         if idx not in o_idx:
    #             new_weights[idx] = centroids[0]
    this_module = torch.from_numpy(new_weights).float().view(fp.size())
    return this_module

this_module = gobo_quantize(fp, 4)
print((this_module-fp).abs().max())
print(this_module.shape)