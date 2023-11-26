import os
import pickle
fileHandler  =  open  (os.path.dirname(os.path.realpath(__file__))+"/st-b8-samsum-eachexperts.csv",  "r")
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


if not os.path.exists(os.path.dirname(os.path.realpath(__file__))+"/st-b8-samsum-test.csv"):
    os.mknod(os.path.dirname(os.path.realpath(__file__))+"/st-b8-samsum-test.csv")
else:
    os.remove(os.path.dirname(os.path.realpath(__file__))+"/st-b8-samsum-test.csv")
    os.mknod(os.path.dirname(os.path.realpath(__file__))+"/st-b8-samsum-test.csv")


tmp_l = []
encoder_token_list = [0,0,0,0,0,0]
decoder_token_list = [0,0,0,0,0,0]
last_seq_ = txts[0].split("\n")[-1].split(" ")[0]
for txt in txts:
    txt_l = txt.split("\n")[-1].split(" ")
    seq_ = txt.split("\n")[-1].split(" ")[0]
    if seq_ != last_seq_:
        encoder_token_list = [0,0,0,0,0,0]
        decoder_token_list = [0,0,0,0,0,0]

    coder_ = txt.split("\n")[-1].split(" ")[1]
    blk_idx = txt.split("\n")[-1].split(" ")[2] #mei yong
    sparse_idx = int((int(blk_idx)-1)/2)
    expert_idx = txt.split("\n")[-1].split(" ")[3]
    if coder_ == "encoder":
        out= f"{seq_} E {encoder_token_list[sparse_idx]} {sparse_idx} {expert_idx}\n"
        encoder_token_list[sparse_idx] += 1
    else:
        out= f"{seq_} D {decoder_token_list[sparse_idx]} {sparse_idx} {expert_idx}\n"
        decoder_token_list[sparse_idx] += 1

    with open(os.path.dirname(os.path.realpath(__file__))+"/st-b8-samsum-test.csv", 'a') as f:
        lines = [out]
        f.writelines(lines)
    last_seq_ = seq_
    # if "decoder" == coder_:
    #     k_ = f"{sparse_idx}_{expert_idx}"
    #     tmp_l.append(k_)