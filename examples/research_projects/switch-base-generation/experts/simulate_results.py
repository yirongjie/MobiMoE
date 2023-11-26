import os
import pickle
import math


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

    # result_ = []
    # print(len(result))
    # for r in range(len(result)):
    #     result_.append(result[r-1])
    return result


Pwd = os.path.dirname(os.path.realpath(__file__))
num_experts = 32
FTpwd = "-samsum"
CacheLimit = 90

f_read = open(f'{Pwd}/st-b{num_experts}{FTpwd}-expfreq.pkl', 'rb')
lruTable =  pickle.load(f_read)
f_read.close()
                
# f_read = open(f'{Pwd}/st-b{num_experts}{FTpwd}-table.pkl', 'rb')
# Table = pickle.load(f_read)
# f_read.close()

f_read = open(f'{Pwd}/st-b{num_experts}{FTpwd}-tablec.pkl', 'rb')
TableC = pickle.load(f_read)
f_read.close()


num_sparse_encoder_layers = 6
num_sparse_decoder_layers = 6
encoder_save_exids = [-1 for x in range(num_sparse_encoder_layers)]
decoder_save_exids = [-1 for x in range(num_sparse_decoder_layers)]

self_caches = {}
self_lru_idx = -1

def initialize():
    # for idx in range(CacheLimit):
    #         self_caches[idx] = ['e', 0, 0]#f"e_0_{0}"   
    global self_caches 
    for idx in range(CacheLimit):
        expert_idx = idx%num_experts
        self_caches[idx] = ['e', math.floor(idx/num_experts), expert_idx]#f"e_{math.floor(idx/self.num_experts)}_{expert_idx}"
    

def update_our_idx(id_, exp_id_ = None):# id_[1]:sparse_idx:0,1,2,3,4,5  
    Param = 1
    # distance_s = []
    # select_idx_list = {}
    select_idx = 0
    init_dis_ = -999
    for i in self_caches:
        d = self_caches[i]
        if d[0] == 'e':
            distance_ = 0
        else:
            distance_ = d[1] - id_[1]
            if distance_ > 0:
                distance_ = distance_ - num_sparse_decoder_layers
            elif distance_ == 0:
                distance_ = -1e-6
            if exp_id_:
                if exp_id_ == d:
                    distance_ = -9999
                    # print(d)
            freq_id_ = f'{d[1]}_{d[2]}'
            if freq_id_ in lruTable.keys():
                distance_ = distance_*lruTable[freq_id_]**Param
        # distance_s.append(distance_)
        # select_idx_list[i] = distance_
        if distance_ > init_dis_:
            init_dis_= distance_
            select_idx = i
            if init_dis_ == 0:
                break
    # a1 = sorted(select_idx_list.items(),key = lambda x:x[1],reverse = True)
    # print(distance_s, select_idx)
    # # print(caches, id_)
    # print(select_idx)
    global self_lru_idx
    self_lru_idx = select_idx
    self_caches[self_lru_idx] = id_
    # print(select_idx_list, a1, select_idx)
    # print(self_lru_idx)

def update_lru_idx(id_):
    global self_lru_idx
    self_lru_idx =( self_lru_idx + 1)%CacheLimit
    self_caches[self_lru_idx] = id_


import random
#示例
def update_rand_idx(id_):# 
    self_lru_idx = random.randint(0,CacheLimit)
    self_caches[self_lru_idx] = id_


def update_idx(id_, exp_id_ = None):# id_[1]:sparse_idx:0,1,2,3,4,5  
    # update_rand_idx(id_)
    # update_lru_idx(id_)

    update_our_idx(id_, exp_id_ = exp_id_)


def get_table_1(is_decoder, sparse_idx, expert_idx):
    f_read = open(f'{Pwd}/st-b{num_experts}{FTpwd}-table.pkl', 'rb')
    Table = pickle.load(f_read)
    f_read.close()
    if is_decoder and sparse_idx == num_sparse_encoder_layers-1:
        return -1
    if not is_decoder and sparse_idx == num_sparse_decoder_layers-1:
        return -1
    global decoder_save_exids
    if is_decoder and sparse_idx == 0:
        len_ = len(decoder_save_exids)
        decoder_save_exids  = [-1 for x in range(len_)]
    if not is_decoder and sparse_idx == 0:
        len_ = len(encoder_save_exids)
        encoder_save_exids  = [-1 for x in range(len_)]
    if is_decoder:
        decoder_save_exids[sparse_idx] = expert_idx
    else:
        encoder_save_exids[sparse_idx] = expert_idx
    key_ = ""
    if is_decoder:
        for ei in decoder_save_exids:
            if ei <0:
                break
            key_ = key_ + str(ei)
    else:
        for ei in encoder_save_exids:
            if ei <0:
                break
            key_ = key_ + str(ei)
    if key_ not in Table:
        return -1
    # print(sparse_idx, expert_idx, key_, Table[key_])
    return Table[key_]



last_layer_num = 2
encoder_save_exids_c = [-1 for x in range(last_layer_num)]
decoder_save_exids_c = [-1 for x in range(last_layer_num)]
def get_table_2(is_decoder, sparse_idx, expert_idx):
    if not is_decoder:
        return -1
    global decoder_save_exids_c
    for ii in range(last_layer_num-1):
        decoder_save_exids_c[ii] = decoder_save_exids_c[ii+1]
    decoder_save_exids_c[-1] = expert_idx
    # if is_decoder and sparse_idx == 0:
    #     len_ = len(decoder_save_exids_c)
    #     decoder_save_exids_c  = [-1 for x in range(len_)]
    # decoder_save_exids_c[sparse_idx] = expert_idx
    # print(decoder_save_exids_c)

    key_ = ""
    for ei in decoder_save_exids_c:
        if ei <0:
            break
        key_ = key_ + str(ei) +"_"
    # print(decoder_save_exids_c, key_)
    if key_ not in TableC[(sparse_idx+1)%num_sparse_encoder_layers]:
        return -1
    # print(sparse_idx, expert_idx, key_, TableC[(sparse_idx+1)%num_sparse_encoder_layers][key_])
    return TableC[(sparse_idx+1)%num_sparse_encoder_layers][key_]

## for 2
# loadOneTime = 0.03452599048614502/2
# computeTime = 0.2759261131286621

loadOneTime = 0.01643212636311849
computeTime = 0.21066498756408691
# computeTime = 0.161

                    
def forward(data):
    seqLen = len(data)
    seq_1token_ts = []
    hit_1token = []
    for seq_idx in range(seqLen):
        initialize()
        seq_d_tokens = data[seq_idx]['D'] # tokensize x 6
        for token_idx, seq_d_t_sparses in enumerate(seq_d_tokens):
            total_cnt = 0
            hit_cnt= 0
            # print(seq_idx, token_idx, seq_d_t_sparses)
            for sparse_idx, seq_d_t_s_expert in enumerate(seq_d_t_sparses):
                # seq_idx: sequance id
                # token_idx: token id
                # seq_d_t_s_expert: each sparse selected expert
                ######LOAD
                id_ = ["d", sparse_idx, seq_d_t_s_expert]
                if list(self_caches.values()).count(id_):
                    # print("HIT")
                    hit_cnt += 1
                else:
                    update_idx(id_)
                #####RUN
                total_cnt += 1
                # print(f"seq{seq_idx}", f"token{token_idx}", f"s{sparse_idx}:{seq_d_t_s_expert}")
                #####PRELOAD
                table = get_table_2(True, sparse_idx, seq_d_t_s_expert)
                # table_2 = get_table_1(True, sparse_idx, seq_d_t_s_expert)
                # if table !=-1 and table_2 !=-1:
                #     print(table[0], table_2[0])
                if table !=-1: 
                    next_expert_idx = table[0]
                    # print( f"-->s{sparse_idx+1}:{next_expert_idx}" f"?=s{sparse_idx+1}:{seq_d_t_sparses[(sparse_idx+1)%6]}")
                    next_id_ = ["d", sparse_idx+1, next_expert_idx]
                    if not list(self_caches.values()).count(next_id_):
                        update_idx(next_id_)

                    # next_expert_idx2 = table[1]
                    # next_id_2 = ["d", sparse_idx+1, next_expert_idx2]
                    # if not list(self_caches.values()).count(next_id_2):
                    #     update_idx(next_id_2, exp_id_=next_id_)
                    #     # self_caches[(self_lru_idx+1)%CacheLimit] = next_id_2
                    # print(self_caches, next_id_2, next_id_)
            # hit_cnt = 0
            unhit = total_cnt-hit_cnt
            seq_1token_t = unhit*loadOneTime + computeTime
            seq_1token_ts.append(seq_1token_t)
            hit_1token.append(hit_cnt)
            # (hit_cnt/total_cnt)
            # print(seq_idx, token_idx, f"{hit_cnt}/{total_cnt}",  "   ", seq_1token_t)
    return seq_1token_ts, hit_1token
                    




if __name__ == "__main__":

    data = loadMoEData(f'{Pwd}/st-b{num_experts}{FTpwd}.csv')
    # print("sequence num:", len(data))
    # print("第1号输入sequence的Decoder部分的第2号token的逐个Gating选择的experts的序号:", data[0]['D'][2])
    seq_1token_ts, hit_1token = forward(data)
    # print(avg(seq_1token_ts))
    print("Latency_1token:",f"{sum(seq_1token_ts)/len(seq_1token_ts)}s", f"[{min(seq_1token_ts)}~{max(seq_1token_ts)}]")
    print("HitIn_1token:", f"{sum(hit_1token)/len(hit_1token)}/6", f"[{min(hit_1token)}~{max(hit_1token)}]")
