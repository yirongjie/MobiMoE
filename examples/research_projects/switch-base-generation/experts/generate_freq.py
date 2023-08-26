import os
import pickle

num_experts = 32
FTpwd = "-samsum"

fileHandler  =  open  (os.path.dirname(os.path.realpath(__file__))+f"/st-b{num_experts}{FTpwd}.csv",  "r")
txts = []
while  True:
    # Get next line from file
    line  =  fileHandler.readline()
    # If line is empty then end of file reached
    if  not  line  :
        break;
    # print(line.strip())
    txts.append(line.strip())
    # Close Close    
fileHandler.close()

tmp_l = []
for txt in txts:
    coder_ = txt.split("\n")[-1].split(" ")[1]
    if "D" == coder_:
        sparse_idx = txt.split("\n")[-1].split(" ")[3]
        # sparse_idx = int((int(blk_idx)-1)/2)
        expert_idx = txt.split("\n")[-1].split(" ")[4]
        k_ = f"{sparse_idx}_{expert_idx}"
        tmp_l.append(k_)

dict_ = {}
for key in tmp_l:
    dict_[key] = dict_.get(key, 0) + 1


show = sorted(dict_.items(),key = lambda x:x[1],reverse = True)

print(show)

# dict_ ={key:model_dict[key].cuda()}
f_save = open(os.path.dirname(os.path.realpath(__file__))+f'/st-b{num_experts}{FTpwd}-expfreq.pkl', 'wb')
pickle.dump(dict_, f_save)
f_save.close()