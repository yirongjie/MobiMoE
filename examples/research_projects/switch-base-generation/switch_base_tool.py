import torch
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt

# model = torch.load(os.path.dirname(os.path.realpath(__file__))+"/models/switch-base-8-yrj/pytorch_model.bin")

# for layer in model:
#     print(layer,model[layer].shape)

torch.set_printoptions(precision=4,sci_mode=False)
int_dir = "/home/ey/personal/program/MoE/huggingface//transformers/examples/research_projects/switch-base-generation/models/switch-base-8-yrj/int8/encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight.bin"
fp_dir = "/home/ey/personal/program/MoE/huggingface//transformers/examples/research_projects/switch-base-generation/models/switch-base-8-yrj/decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight.bin"#encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight.bin"
f_read = open(fp_dir, 'rb')
model = pickle.load(f_read)
fp = model["decoder.block.1.layer.2.mlp.experts.expert_0.wi.weight"].float()#'encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].float()

############################33

# t1 = time.time()
# f_read = open(int_dir, 'rb')
# Qmodel = pickle.load(f_read)
# t2 = time.time()
# Qfp = Qmodel['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].dequantize()
# # print(Qfp)
# t3 = time.time()
# f_read = open(fp_dir, 'rb')
# model = pickle.load(f_read)
# t4 = time.time()
# fp = model['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].float()
# t5 = time.time()
# print(t2-t1, t4-t3)


############################33

# print(fp)
scale_ = (float(torch.max(fp))-float(torch.min(fp)))/2**8
zero_point_ = int((float(torch.max(fp))+float(torch.min(fp)))/2)
Q_tensor = torch.quantize_per_tensor(fp, scale=scale_, zero_point=zero_point_, dtype=torch.qint8)
qfpt = Q_tensor.dequantize()
# print(qfpt)

scales = []
zero_points = []
for j in range(fp.shape[1]):
    ch_data = fp[:,j]
    scale_ = (ch_data.max() - ch_data.min())/2**8
    zero_point_ = int((ch_data.max() + ch_data.min())/2)
    scales.append(scale_)
    zero_points.append(zero_point_)

Q_ = torch.quantize_per_channel(fp, torch.tensor(scales), torch.tensor(zero_points), 1, torch.qint8)
aasss = torch.tensor(scales)
aazzz = torch.tensor(zero_points)
qfp = Q_.dequantize()
# print(qfp)



############################33
from pytorch_quantization import tensor_quant
x = fp
num_bits = 8
quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max(), num_bits, False, True)
dequant_x = quant_x/scale.to(quant_x.dtype)


############################33
# print(fp.shape[1])
# for i in fp:
#     print(i.max()-i.min(), i.shape)

# r = []
# for j in range(fp.shape[1]):
#     i = fp[:,j]
#     r.append(float(i.abs().max()))
#     # print(i.max()-i.min(), i.shape)
# print(max(r), min(r))

# np.random.seed(123)

# # x1 = np.random.normal(0, 1, size=1000)
# # print(type(x1))
# x1 = fp.permute([1,0]).numpy()

# # 当在同一幅图表中创建多个直方图，最好使用'stepfilled'，并调整透明度
# kwargs = {
#     "bins": 40,
#     "histtype": "stepfilled",
#     "alpha": 0.5
# }

# fig, ax = plt.subplots(figsize=(10, 7))
# ax.hist(x1, label="x1", **kwargs)
# ax.set_title("Histogram for multiple variables")
# ax.legend()
# plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/testquant/test.jpg")

############################
def quantize_tensor(data, bit_num):
    scale = (data.max() - data.min())/2**bit_num
    zero_point = (data.max() + data.min())/2
    quantized_data = torch.round(((data-torch.full_like(data,zero_point))/torch.full_like(data,scale)))
    accumulated_error = data-(quantized_data*scale+zero_point)
    quantized_data = quantized_data.type(torch.int8)
    return quantized_data,scale,zero_point

def dequantize_tensor(qt,scale,zero_point):
    return qt.data.type(torch.float32)*scale+zero_point

def quantize_channel(data, bit_num, channel_num=1):# ONLY 1
    quantized_data = torch.zeros(data.shape).type(torch.int8)
    scale = torch.zeros((1, data.shape[channel_num]))
    zero_point = torch.zeros((1, data.shape[channel_num]))
    # print(quantized_data.shape)
    for j in range(data.shape[channel_num]):
        ch_data = data[:,j]
        scale_ = (ch_data.max() - ch_data.min())/2**bit_num
        zero_point_ = int((ch_data.max() + ch_data.min())/2)
        # quantized_data[:,j] = torch.round(((ch_data-torch.full_like(ch_data,zero_point_))/torch.full_like(ch_data,scale_))).type(torch.int)
        quantized_data[:,j] = torch.round(((ch_data-torch.full_like(ch_data,zero_point_))/torch.full_like(ch_data,scale_))).type(torch.int)
        scale[:,j] = scale_
        zero_point[:,j] = zero_point_
    return quantized_data,scale,zero_point

def dequantize_channel(qt,scale,zero_point):
    return qt.data.type(torch.float32)*scale+zero_point

qq, ss, zz = quantize_channel(fp, 8)
ff = dequantize_channel(qq, ss, zz)
# print(ff)
# for j in range(len(ff)):
#     print(j, end = ": ")
#     for i in range(len(ff[j])):
#         if (float(ff[j][i] -  qfp[j][i])) != 0.0:
#             print(i, ff[j][i] ,  qfp[j][i], end = "    ")
#     print()


qq, ss, zz = quantize_tensor(fp, 8)
ff = dequantize_tensor(qq, ss, zz)
# print((qfpt - ff).abs().max())


############################################
# # lwg: gobo implementation 
# # returns quantized matrix
# def detect_outliers(weights):
#     from sklearn.mixture import GaussianMixture 
#     print("detecting outliers...")
#     # find outliers
#     _weights = weights.reshape(-1, 1)
#     gm = GaussianMixture(n_components=1, random_state=0).fit(_weights)
#     scores = gm.score_samples(_weights)
#     outliers = []
#     _weights = _weights.numpy()
#     for i in range(0, len(scores)):
#         if scores[i] <= -4.0:
#             outliers.append(_weights[i][0])
#     print("masked: ", len(outliers))
#     print(weights.shape, len(scores))
#     mask = np.zeros(weights.shape, dtype=bool)
#     mask[np.where(scores <= -4.0)] = True
#     return np.array(outliers), mask

# def gobo_quantize(weights, o_idx, bits):
#     print("gobo qunatization. Bits = ", bits)
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
#     quantized = np.digitize(weights, bins, right = True) - 1 # return the idx of the centroids
#     print("quantzied weights are:", quantized)



#     #save_weight_to_file('/tmp/weight', quantized)
#     start = time.time()
#     new_weights = centroids[quantized]
#     print("new weights are:", new_weights)
#     # print("o_idx:", o_idx)
#     # print("centroids:", centroids)
#     # recover corresponding outlier weights
#     new_weights[o_idx] = weights[o_idx]
#     end = time.time()
#     print("restoring weight takes " ,(end-start) * 1000, "ms")
#     print("centroids size: " , centroids.shape)
#     print("quantized weight size: ", quantized.shape)
#     print("original weight size: ", weights.size())
#     # sanity check manually patch some values...
#     for idx,d in enumerate(new_weights):
#         if d < -100.0:
#             # print(idx, weights[idx],"fail to be binned??")
#             # there are values that are not in outlier idx 
#             if idx not in o_idx:
#                 new_weights[idx] = centroids[0]
#                 # print("manually patch", weights[idx], "to", centroids[0])
#     return new_weights

# def gobo_quant__(fp, bits):
#     weights = torch.Tensor([]) 
#     # original dimension and pointer address to restore to
#     weights = torch.cat([weights, torch.flatten(fp)])
#     o_group, o_idx = detect_outliers(weights)
#     bits=8
#     new_weight =  gobo_quantize(weights, o_idx, bits)
#     print("weights:", weights)#, weights.shape)
#     print("new_weight:", new_weight)#, new_weight.shape)
#     this_module = torch.from_numpy(new_weight).float().view(fp.size())
#     print(this_module)
#     print((this_module-fp).abs().max())
#     return this_module


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
    quantized = np.digitize(weights, bins, right = True) - 1 # return the idx of the centroids
    start = time.time()
    new_weights = centroids[quantized]
    # recover corresponding outlier weights
    new_weights[o_idx] = weights[o_idx]
    end = time.time()
    # sanity check manually patch some values...
    for idx,d in enumerate(new_weights):
        if d < -100.0:
            # there are values that are not in outlier idx 
            if idx not in o_idx:
                new_weights[idx] = centroids[0]
    this_module = torch.from_numpy(new_weights).float().view(fp.size())
    return this_module

this_module = gobo_quantize(fp, 3)
print((this_module-fp).abs().max())