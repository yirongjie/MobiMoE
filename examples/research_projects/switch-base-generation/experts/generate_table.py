# %%
import sys
import numpy as np
print(sys.version)

# %%
def loadMoEData(file_name):
    fileHandler  =open(file_name,  "r")
    filelines = []
    while  True:
        line  =  fileHandler.readline()
        if  not  line  :
            break
        txt_ = line.strip().split("\n")[0]
        if len(txt_):
            filelines.append(txt_)
    fileHandler.close()
    seq_num = int(filelines[-1].split(" ")[0])
    seqs_en =[]
    seqs_de =[]
    for i in range(seq_num+1):
        seqs_en.append([])
        seqs_de.append([])
    for line in filelines:
        if line.split(" ")[1] == 'E':
            seqs_en[int(line.split(" ")[0])].append(line.split(" ")[2:])    
        if line.split(" ")[1] == 'D':
            seqs_de[int(line.split(" ")[0])].append(line.split(" ")[2:])
    result = []    
    for idx in range(0, len(seqs_en)):
        # print(seqs_en, idx)
        seq_en = seqs_en[idx]
        seq_de = seqs_de[idx]

        seqs_en_token =[]
        max_ = 0
        for m in seq_en:
            if int(m[0]) > max_:
                max_ = int(m[0])
        for i in range(max_+1):
            seqs_en_token.append([])
        seqs_de_token =[]
        max_ = 0
        for m in seq_de:
            if int(m[0]) > max_:
                max_ = int(m[0])
        for i in range(max_+1):
            seqs_de_token.append([])
        for et in seq_en:
            # print(seqs_en, et)
            seqs_en_token[int(et[0])].append(int(et[2]))
        for dt in seq_de:
            seqs_de_token[int(dt[0])].append(int(dt[2]))
        dict_ = {'E':seqs_en_token, 'D':seqs_de_token}
        result.append(dict_)
    return result

# %% [markdown]
# 模型说明： google/switch-base-8, 每个MoE包含8个export(标记为0～7号). Encoder和Decoder分别有6个MoE层；
# 
# 数据说明： output.glue.test.csv 测试自GLUE数据集的测试集。含有1083个输入sequence（标记为0～1082号）；
# 
# 函数说明： loadMoEData 函数返回一个四维数据，根据前三个维度索引, 分别为:
# 
#          (1)输入sequence序号(0~1083)
# 
#          (2)Encoder(E)/Decoder(D)
# 
#          (3)Encoder/Decoder的第几个token(0~data[m]['D'], m表示输入sequence序号)
# 
#          最后一个维度为6个MoE层按顺序的选中的export序号（0～7）

# %%
import os

num_experts = 16
FTpwd = "-samsum"

train_data = loadMoEData(os.path.dirname(os.path.realpath(__file__))+f"/st-b{num_experts}{FTpwd}.csv")
print("sequence num:", len(train_data))
print("第0号输入sequence的Decoder部分的第2号token的逐个Gating选择的experts的序号:", train_data[0]['D'][2])


# %%
# decoder_test_data = np.array([item for sublist in test_data for item in sublist['D']])
decoder_train_data = np.array([item for sublist in train_data for item in sublist['D']])
np.set_printoptions(precision=3, suppress=True)

# %%
EXPERT_NUM = num_experts  # 每层专家个数
LAYER_NUM = 6  # Encoder/Decoder的moe层数

def Get_raw_freq(data):
    """
    功能: 计算各层中各个专家出现的概率分布
    
    参数: 
        - data: 
    
    返回: 
        - (LAYER_NUM, EXPERT_NUM)的概率分布矩阵
    """

    L = np.zeros((LAYER_NUM, EXPERT_NUM), dtype=int) # 频次分布矩阵
    for item in data:
        for layer in range(LAYER_NUM):
            expert = item[layer]
            L[layer][expert] += 1
    L_freq = np.zeros((LAYER_NUM, EXPERT_NUM)) # 概率分布矩阵
    L_sum = np.sum(L, 1)
    for i in range(LAYER_NUM):
        for j in range(EXPERT_NUM):
            L_freq[i][j] = L[i][j]/L_sum[i]
    
    return L_freq

raw_freq = Get_raw_freq(decoder_train_data)
print("各层中各个专家的出现概率\n", raw_freq)
raw_var = np.var(raw_freq, axis = 1)
print("各层的方差", raw_var, "\n各层方差的平均值为", np.mean(raw_var))

# %% [markdown]
# **可以看到，在不选定前n层的专家时，直接统计到的各层专家分布比较均匀**

# %%
def Get_freq(data, layer, layer_target):
    """
    功能: 计算在前layer层取各种组合时, 第layer_target层的专家概率分布
    
    参数: 
        - data: 
        - layer: 固定的层
        - layer_target: 统计的层
    
    返回: 
        - (EXPERT_NUM^layer, EXPERT_NUM)的频次分布矩阵
        - (EXPERT_NUM^layer, EXPERT_NUM)的概率分布矩阵
    """
    if layer>LAYER_NUM-1 or layer_target<= layer or layer_target>LAYER_NUM:
        print("Get_freq: Wrong input layer")
        return

    L = np.zeros((pow(EXPERT_NUM, layer), EXPERT_NUM), dtype=int) # 频次分布矩阵
    for item in data:
        idx0 = 0
        for i in range(layer):
            idx0 += item[i] * pow(EXPERT_NUM, layer-1-i)
        idx1 = item[layer_target-1]
        L[idx0][idx1] += 1

    L_freq = np.zeros((pow(EXPERT_NUM, layer), EXPERT_NUM)) # 概率分布矩阵
    L_sum = np.sum(L, 1)
    for idx0 in range(pow(EXPERT_NUM, layer)):
        for idx1 in range(EXPERT_NUM):
            if L_sum[idx0]>0:
                L_freq[idx0][idx1] = L[idx0][idx1]/L_sum[idx0]
            else:
                L_freq[idx0][idx1] = 0
    
    return L, L_freq


# %%
Layer = 2
Layer_target = 3
nums, freq = Get_freq(decoder_train_data, Layer, Layer_target)
print(f"前{Layer}层专家各种组合下,第{Layer_target}层专家频次分布: \n", nums)
print(f"前{Layer}层专家各种组合下,第{Layer_target}层专家概率分布: \n", freq)

# %%
def Show_var(freq, layer, layer_target):
    """
    功能: 根据给定的概率分布矩阵，统计方差信息
    
    参数: 
        - freq: 概率分布矩阵
        - layer: 固定的层(需要和Get_freq的参数一致)
        - layer_target: 统计的层(需要和Get_freq的参数一致)
    """
    temp_freq = []
    for i in range(len(freq)):
        if np.any(freq[i]):  # 全0为“不存在”的一种组合，不计入统计
            temp_freq.append(freq[i])

    var_lst = []
    for i in range(len(temp_freq)):
        var_lst.append(np.var(temp_freq[i]))

    print(f"给定前{layer}层的专家, 第{layer_target}层专家分布方差的平均值为", sum(var_lst)/len(var_lst))
    print(f"不限定前{layer}层，第{layer_target}层专家分布的方差为", raw_var[layer_target-1])
    print(f"增大了{round(sum(var_lst)/len(var_lst)/raw_var[layer_target-1])}倍")

Show_var(freq, Layer, Layer_target)

# %% [markdown]
# **可以看到，在选定前2层专家时，第3层专家的分布就很不均匀了**

# %%
def Get_statistics(freq, layer, layer_target, first_n):    
    """
    功能: 生成专家选择字典和统计数据等

    参数:
        - freq: 概率分布矩阵
        - layer: 固定的层(需要和Get_freq的参数一致)
        - layer_target: 统计的层(需要和Get_freq的参数一致)
        - first_n: 选取的高概率专家个数, 比如frist_n=3是选取概率最高的三个专家
        
    返回: 
        - 前layer层与对应的layer_target层的最高的first_n个频率、对应专家号、概率和
        - 各种统计数据
        - 概率最高的{first_n}个专家的总占比（概率和）
    """

    freq_sort = np.sort(freq, 1)  # 对freq每一行按概率从低到高排序
    freq_sort_arg = np.argsort(freq, 1)  # 对应的专家序号
    freq_firstn = freq_sort[:,(EXPERT_NUM-first_n):]  # 排序后截取概率最高的first_n个数据  
    freq_firstn_arg = freq_sort_arg[:,(EXPERT_NUM-first_n):] 
    freq_firsn_sum = np.sum(freq_firstn, 1)  # 概率最高的{first_n}个专家的总占比（概率和）

    freq_sort_withzero = {}  # 字典，前layer层与对应的layer_target层的最高的first_n个频率、对应专家号、概率和
    sort_withzero = {}  # 字典，前layer层与对应的layer_target层的最高的first_n个频率、对应专家号、概率和
    statistics_nonzero = [] # 去除全0行后
    sum_nonzero = []  # 用于统计去除全0行后的概率和的均值
    for i in range(len(freq)):
        i_oct = ("%o" % i).zfill(layer)  # 把序号转成8进制，每位直接表示该层的专家号
        freq_sort_withzero[i_oct] = (freq_firstn[i], freq_firstn_arg[i], freq_firsn_sum[i])
        sort_withzero[i_oct]=list(freq_firstn_arg[i])

        if not(np.any(freq[i])):  # 全0为“不存在”的一种组合，不计入统计
            continue
        else:
            # print(f"前{layer}层专家为{i_oct}",f"第{layer_target}层最高的{first_n}个概率为", freq_firstn[i], "对应的专家号", freq_firstn_arg[i], "概率和", freq_firsn_sum[i])
            statistics_nonzero.append([i_oct, freq_firstn[i], freq_firstn_arg[i], freq_firsn_sum[i]])
            sum_nonzero.append(freq_firsn_sum[i])

    sum_nonzero_mean = np.mean(np.array(sum_nonzero)) # sum_nonzero均值
    print(f"对于前{layer}层专家各种组合，第{layer_target}层最高的{first_n}个概率和的平均值为", sum_nonzero_mean)
    return freq_sort_withzero, statistics_nonzero, sum_nonzero_mean, sort_withzero

First_n = 3
expert_dic, statistics, avg, s_dic = Get_statistics(freq, Layer, Layer_target, First_n) 

# %% [markdown]
# “对于前2层专家各种组合，第3层最高的3个概率和的平均值为 0.9361372874714965”
# 
# **这就说明，在前2层专家确定后，在第3层只需要预加载特定的3个专家即可**

# %%
print(s_dic)

# %%
print(expert_dic["34"])
for i in range(len(statistics)):
    print(f"前{Layer}层专家为{statistics[i][0]}",f"第{Layer_target}层最高的{First_n}个概率为", statistics[i][1], "对应的专家号", statistics[i][2], "概率和", statistics[i][3])
print(len(statistics))
print(f"\n对于前{Layer}层专家各种组合，第{Layer_target}层最高的{First_n}个概率和的平均值为", avg)

# %%
# 遍历，统计所有结果

for i in range(1, LAYER_NUM):
    for j in range(i+1, LAYER_NUM+1):
        _, freq = Get_freq(decoder_train_data, i, j)
        Show_var(freq, i, j)
        Get_statistics(freq, i, j, 2)  
        Get_statistics(freq, i, j, 3)        
        print("\n")


# %%
# 遍历，统计所有结果
global_dict = {}
for i in range(1, LAYER_NUM):
    j = i+1
    _, freq = Get_freq(decoder_train_data, i, j)
#     Show_var(freq, i, j)
    _, _, _, s_dict = Get_statistics(freq, i, j, 2) 
    global_dict = dict(list(global_dict.items()) + list(s_dict.items()))  
    print(freq.shape)
    # print(s_dict)
#     Get_statistics(freq, i, j, 3)        
# print("\n")
# print(global_dict)




# global_dict_ = {}
# for i in range(1, LAYER_NUM):
#     j = i+1
#     _, freq = Get_freq(decoder_train_data, i, j)
# #     Show_var(freq, i, j)
#     _, _, _, s_dict = Get_statistics(freq, i, j, 2) 
#     print(s_dict)
#     global_dict_ = dict(list(global_dict_.items()) + list(s_dict.items()))  
# exit()




import pickle
# 字典保存
# dict = {'a':1,'b':2,'c':3}
f_save = open(os.path.dirname(os.path.realpath(__file__))+f"/st-b{num_experts}{FTpwd}-table.pkl", 'wb')
pickle.dump(global_dict, f_save)
f_save.close()
 
# # 读取
f_read = open(os.path.dirname(os.path.realpath(__file__))+f"/st-b{num_experts}{FTpwd}-table.pkl", 'rb')
dict2 = pickle.load(f_read)
# print(dict2)
f_read.close()
