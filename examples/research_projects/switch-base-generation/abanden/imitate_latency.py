##########
truth_next_experts = [1,1,1,1,1,1,1,1,1,1,1,1]
def last_l_idx(i):
    return i-1
load_num = 2
def get_next_expert(i):
    return 1



## don't change
Tne = 0.3/6
Te = 0.015/6
Tl = 0.2/6
ETl = [0]
ETne = [Tne]
ETe = [Tne+Te]

for i in range(1, len(truth_next_experts)):
    ETl_i=ETe[last_l_idx(i)]+Tl*load_num
    ETl.append(ETl_i)
    ETne_i = ETe[i-1]+Tne
    ETne.append(ETne_i)
    tmp_Tl = Tl
    if get_next_expert(i) == truth_next_experts[i]:
        tmp_Tl = 0
    ETe_i = max(ETl[i], ETne[i]) + tmp_Tl +Te
    ETe.append(ETe_i)

print(ETne[-1]/(len(truth_next_experts)/6))