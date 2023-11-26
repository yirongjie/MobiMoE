import numpy as np
import matplotlib.pyplot as plt
import os


# it8
save_ = "moe-85-it8"
SloadTimes= []
SmainTimes= [[1691250855.057764, 1691250855.068589], [1691250855.121982, 1691250855.1333132], [1691250855.1829078, 1691250855.1941812], [1691250855.2446005, 1691250855.2557535], [1691250855.3069568, 1691250855.3174133], [1691250855.3672626, 1691250855.3789642]]
SmainNoTimes= [[1691250855.0238955, 1691250855.0238955], [1691250855.0686054, 1691250855.0897684], [1691250855.133321, 1691250855.150135], [1691250855.1941903, 1691250855.212388], [1691250855.25577, 1691250855.2758586], [1691250855.3174207, 1691250855.3370187]]
SupTimes= [[1691250855.0239015, 1691250855.057757], [1691250855.089776, 1691250855.1219785], [1691250855.150142, 1691250855.182903], [1691250855.2123952, 1691250855.2445943], [1691250855.275866, 1691250855.3069532], [1691250855.3370252, 1691250855.3672595]]
# attn+ffn+attn+gate 0.09586143493652344 experts 0.06674051284790039 preload 0
# load 0.19233965873718262 s
# compute: 0.1838359832763672 s
# decoder: 0.3761756420135498 s

# next
save_ = "moe-85-it8next"
SloadTimes= [[1691249943.068006, 1691249943.102619], [1691249943.1467257, 1691249943.1781917], [1691249943.2154107, 1691249943.250616], [1691249943.2864444, 1691249943.3181195], [1691249943.356213, 1691249943.3868566], [1691249943.4231472, 1691249943.42316]]
SmainTimes= [[1691249943.0679603, 1691249943.075319], [1691249943.1466837, 1691249943.1538277], [1691249943.2153075, 1691249943.2226498], [1691249943.2864044, 1691249943.294033], [1691249943.3561716, 1691249943.3636615], [1691249943.423079, 1691249943.430806]]
SmainNoTimes= [[1691249943.0385597, 1691249943.0385597], [1691249943.0753322, 1691249943.1145868], [1691249943.1538396, 1691249943.1852818], [1691249943.2226658, 1691249943.2562153], [1691249943.2940443, 1691249943.3264983], [1691249943.36368, 1691249943.394398]]
SupTimes= [[1691249943.0385706, 1691249943.067879], [1691249943.1146, 1691249943.1466181], [1691249943.1852946, 1691249943.2152328], [1691249943.2562268, 1691249943.2863379], [1691249943.3265097, 1691249943.3561077], [1691249943.394409, 1691249943.4230118]]
# attn+ffn+attn+gate 0.1674184799194336 experts 0.044690608978271484 preload 0.16361570358276367
# load 0.17957663536071777 s
# compute: 0.2339463233947754 s
# decoder: 0.41352295875549316 s

# # nextbest
save_ = "moe-85-it8nextbest"
SloadTimes= [[1691250082.1923993, 1691250082.2497911], [1691250082.258786, 1691250082.2916205], [1691250082.302269, 1691250082.3377905], [1691250082.3489244, 1691250082.3812551], [1691250082.3813682, 1691250082.4111407], [1691250082.418558, 1691250082.418568]]
SmainTimes= [[1691250082.1923375, 1691250082.195431], [1691250082.2587183, 1691250082.2618513], [1691250082.3021982, 1691250082.3052793], [1691250082.3488677, 1691250082.351849], [1691250082.3813345, 1691250082.388459], [1691250082.4185164, 1691250082.4205468]]
SmainNoTimes= [[1691250082.192237, 1691250082.192237], [1691250082.195443, 1691250082.2586298], [1691250082.2618628, 1691250082.30211], [1691250082.3052905, 1691250082.348783], [1691250082.3518608, 1691250082.3807461], [1691250082.3884702, 1691250082.4184515]]
SupTimes= [[1691250082.1922495, 1691250082.1922498], [1691250082.2586455, 1691250082.258646], [1691250082.302125, 1691250082.3021255], [1691250082.3487983, 1691250082.3487988], [1691250082.3813, 1691250082.3813002], [1691250082.4184613, 1691250082.418462]]
# attn+ffn+attn+gate 0.2057933807373047 experts 0.021443843841552734 preload 0.18786120414733887
# load 0 s
# compute: 0.2560157775878906 s
# decoder: 0.25601840019226074 s

# # nextbest
save_ = "moe-85-it2nextbest"
SloadTimes= [[1691483128.720971, 1691483128.7741008], [1691483128.7910702, 1691483128.8252816], [1691483128.8379526, 1691483128.8712454], [1691483128.8819978, 1691483128.9142861], [1691483128.92539, 1691483128.957545], [1691483128.9693484, 1691483128.9693613]]
SmainTimes= [[1691483128.720899, 1691483128.7236884], [1691483128.7910075, 1691483128.7942126], [1691483128.8378947, 1691483128.8401098], [1691483128.8819463, 1691483128.8841534], [1691483128.9253335, 1691483128.927685], [1691483128.9692948, 1691483128.9713805]]
SmainNoTimes= [[1691483128.7208238, 1691483128.7208238], [1691483128.7236962, 1691483128.7909164], [1691483128.794227, 1691483128.8378255], [1691483128.8401182, 1691483128.8818812], [1691483128.884161, 1691483128.9252503], [1691483128.9276938, 1691483128.9692283]]
SupTimes= [[1691483128.7208343, 1691483128.7208345], [1691483128.7909336, 1691483128.7909338], [1691483128.8378384, 1691483128.837839], [1691483128.881893, 1691483128.8818932], [1691483128.925266, 1691483128.9252665], [1691483128.96924, 1691483128.9692407]]
# attn+ffn+attn+gate 0.23520565032958984 experts 0.014853715896606445 preload 0.1850900650024414
# load 0 s
# compute: 0.2701869010925293 s
# decoder: 0.2701892852783203 s

save_ = "moe-85-it8next-usb"
SloadTimes= [[1691294774.130073, 1691294774.6687887], [1691294775.1655726, 1691294775.6230981], [1691294775.8919618, 1691294776.430092], [1691294776.8726997, 1691294777.2590585], [1691294777.6225913, 1691294778.1176195], [1691294778.4924903, 1691294778.4925108]]
SmainTimes= [[1691294774.1300106, 1691294774.140576], [1691294775.1655035, 1691294775.1756163], [1691294775.8919032, 1691294775.9019902], [1691294776.8726416, 1691294776.8826196], [1691294777.622539, 1691294777.6327417], [1691294778.4924319, 1691294778.5028348]]
SmainNoTimes= [[1691294773.6820064, 1691294773.6820064], [1691294774.140586, 1691294774.2185419], [1691294775.1756315, 1691294775.2549863], [1691294775.9020033, 1691294775.9777722], [1691294776.8826299, 1691294776.9649467], [1691294777.6327567, 1691294777.7117803]]
SupTimes= [[1691294773.682018, 1691294774.1298788], [1691294774.6688433, 1691294775.1653876], [1691294775.623158, 1691294775.8918297], [1691294776.430149, 1691294776.8725684], [1691294777.259147, 1691294777.6224725], [1691294778.1177077, 1691294778.4923291]]
# attn+ffn+attn+gate 0.39442014694213867 experts 0.061348676681518555 preload 2.415778875350952
# load 2.3934431076049805 s
# compute: 2.6348094940185547 s
# decoder: 5.028252601623535 s


save_ = "moe-85-it8next-end"
SloadTimes= [[1691631930.7465217, 1691631930.76697], [1691631930.7750466, 1691631930.7964845], [1691631930.806344, 1691631930.8260524], [1691631930.8346086, 1691631930.8639698], [1691631930.905801, 1691631930.9259799], [1691631930.9890785, 1691631930.9890902]]
SmainTimes= [[1691631930.746476, 1691631930.748721], [1691631930.7750072, 1691631930.777212], [1691631930.8062842, 1691631930.8087282], [1691631930.8345685, 1691631930.8367572], [1691631930.905757, 1691631930.9082243], [1691631930.9890356, 1691631930.9961195]]
SmainNoTimes= [[1691631930.7008774, 1691631930.746375], [1691631930.7487278, 1691631930.774907], [1691631930.7772195, 1691631930.8059552], [1691631930.808735, 1691631930.834464], [1691631930.836764, 1691631930.90565], [1691631930.9082344, 1691631930.9716241]]
SupTimes= [[1691631930.7463856, 1691631930.7464142], [1691631930.774918, 1691631930.7749512], [1691631930.8059855, 1691631930.8060923], [1691631930.834475, 1691631930.8345072], [1691631930.9056604, 1691631930.905695], [1691631930.9716349, 1691631930.988957]]
SloadTimes= [[1691632001.4378529, 1691632001.4589849], [1691632001.4669008, 1691632001.4859126], [1691632001.49428, 1691632001.5141444], [1691632001.5227861, 1691632001.5421433], [1691632001.5510976, 1691632001.5721707], [1691632001.5995753, 1691632001.5995877]]
SmainTimes= [[1691632001.437807, 1691632001.4400883], [1691632001.4668467, 1691632001.4691026], [1691632001.4942331, 1691632001.4965193], [1691632001.5227354, 1691632001.5250509], [1691632001.5510352, 1691632001.5533981], [1691632001.599532, 1691632001.6065395]]
SmainNoTimes= [[1691632001.395455, 1691632001.437706], [1691632001.440095, 1691632001.4667506], [1691632001.4691093, 1691632001.4941356], [1691632001.4965274, 1691632001.5226374], [1691632001.525058, 1691632001.5509], [1691632001.5534058, 1691632001.579219]]
SupTimes= [[1691632001.4377172, 1691632001.4377472], [1691632001.4667604, 1691632001.4667952], [1691632001.4941466, 1691632001.4941804], [1691632001.5226488, 1691632001.5226827], [1691632001.550912, 1691632001.5509799], [1691632001.5792289, 1691632001.5994577]]
# attn+ffn+attn+gate 0.17169833183288574  experts 0.018509387969970703 preload 0.10045075416564941
# load 0.020429372787475586
# 

save_ = "moe-85-it8-end"
SloadTimes= []
SmainTimes= [[1691632107.3379033, 1691632107.3395753], [1691632107.3523886, 1691632107.3547447], [1691632107.3811154, 1691632107.3836257], [1691632107.432113, 1691632107.4344568], [1691632107.486024, 1691632107.4884238], [1691632107.5564282, 1691632107.5666633]]
SmainNoTimes= [[1691632107.291638, 1691632107.337866], [1691632107.3395813, 1691632107.352355], [1691632107.3547635, 1691632107.3810775], [1691632107.383644, 1691632107.4320738], [1691632107.4344754, 1691632107.485985], [1691632107.4884465, 1691632107.5380166]]
SupTimes= [[1691632107.3378735, 1691632107.3379023], [1691632107.352361, 1691632107.3523874], [1691632107.3810844, 1691632107.3811145], [1691632107.4320815, 1691632107.432112], [1691632107.485993, 1691632107.4860227], [1691632107.5380244, 1691632107.5564241]]


save_ = "moe-85-it8-test"
SloadTimes= [[1692356062.283021, 1692356062.3352308], [1692356062.3449879, 1692356062.3954012], [1692356062.4054146, 1692356062.4283388], [1692356062.440549, 1692356062.4992957], [1692356062.5032136, 1692356062.5260637], [1692356062.5377266, 1692356062.5377383]]
SmainTimes= [[1692356062.2735639, 1692356062.2828543], [1692356062.3353236, 1692356062.3448484], [1692356062.3955138, 1692356062.4052646], [1692356062.4385545, 1692356062.4404025], [1692356062.501229, 1692356062.5030673], [1692356062.5357063, 1692356062.537587]]
SmainNoTimes= [[1692356062.222846, 1692356062.2535477], [1692356062.28294, 1692356062.3255923], [1692356062.3449152, 1692356062.3869932], [1692356062.4053352, 1692356062.4384732], [1692356062.440474, 1692356062.501179], [1692356062.503135, 1692356062.535653]]
SupTimes= [[1692356062.2535598, 1692356062.2735634], [1692356062.3352852, 1692356062.335323], [1692356062.3954747, 1692356062.395513], [1692356062.438486, 1692356062.438554], [1692356062.5011897, 1692356062.5012286], [1692356062.5356665, 1692356062.5357058]]
# attn+ffn+attn+gate 0.24179315567016602 experts 0.034132957458496094 compute: 0.2759261131286621 1compute: 0.04598768552144369
# preload 0.20715594291687012 1load: 0.03452599048614502
# load 0.020226001739501953 6

save_ = "moe-85-it8-test2"
SloadTimes= [[1692355572.9412155, 1692355572.9702005], [1692355573.0058742, 1692355573.0295603], [1692355573.0383813, 1692355573.038422], [1692355573.0808253, 1692355573.1033773], [1692355573.1406758, 1692355573.1639948], [1692355573.1737013, 1692355573.173711]]
SmainTimes= [[1692355572.9319363, 1692355572.941041], [1692355572.9983754, 1692355573.0057197], [1692355573.036782, 1692355573.038256], [1692355573.0720654, 1692355573.0806737], [1692355573.130775, 1692355573.1405122], [1692355573.1720054, 1692355573.1735573]]
SmainNoTimes= [[1692355572.8840806, 1692355572.912675], [1692355572.9411175, 1692355572.9778209], [1692355573.0057971, 1692355573.0367382], [1692355573.0383172, 1692355573.0529354], [1692355573.0807476, 1692355573.111367], [1692355573.1405933, 1692355573.1719618]]
SupTimes= [[1692355572.912686, 1692355572.931936], [1692355572.9778411, 1692355572.998375], [1692355573.0367486, 1692355573.0367815], [1692355573.0529463, 1692355573.0720644], [1692355573.1113787, 1692355573.1307745], [1692355573.1719718, 1692355573.172005]]
# attn+ffn+attn+gate 0.17284464836120605 experts 0.03782033920288086 compute: 0.21066498756408691 1compute: 0.03511083126068115
# preload 0.09859275817871094 1load: 0.01643212636311849
# load 0.07836365699768066 6






basetime = SupTimes[0][0]
bottom = 0
plt.figure(figsize=(50, 5))
plt.yticks(fontsize=32)


def plt_soc(loadTime, mainTime, upTime, mainNoTime):

    loadSum = 0
    mainSum = 0
    upSum = 0
    mainNoSum = 0

    ax_list = ["load Thread"]
    for i in range(len(loadTime)):
        plt.barh(ax_list, loadTime[i][1] - loadTime[i][0], height=0.6, color="black", left=loadTime[i][0]-basetime,
                 edgecolor='black')
        loadSum += (loadTime[i][1] - loadTime[i][0])

    ax_list = ["main"]
    for i in range(len(mainTime)):
        plt.barh(ax_list, mainTime[i][1] - mainTime[i][0], height=0.6, color="red",
                 left=mainTime[i][0]-basetime, edgecolor='black')
        mainSum += (mainTime[i][1] - mainTime[i][0])

    for i in range(len(upTime)):
        plt.barh(ax_list, upTime[i][1] - upTime[i][0], height=0.6, color="black", left=upTime[i][0]-basetime,
                 edgecolor='black')
        upSum += ( upTime[i][1] - upTime[i][0])

    for i in range(len(mainNoTime)):
        plt.barh(ax_list, mainNoTime[i][1] - mainNoTime[i][0], height=0.6, color="green", left=mainNoTime[i][0]-basetime,
                 edgecolor='black')
        mainNoSum += (mainNoTime[i][1] - mainNoTime[i][0])

    print("attn+ffn+attn+gate", mainNoSum,"experts", mainSum, "compute:",mainNoSum+mainSum, "1compute:", float(mainNoSum+mainSum)/len(mainNoTime))
    print("preload", loadSum, "1load:", loadSum/len(loadTime))
    print("load", upSum, len(loadTime))


plt_soc(SloadTimes, SmainTimes, SupTimes, SmainNoTimes)
# plt.legend(fontsize=32)
plt.gca().invert_yaxis()

# plt.show()
plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/"+save_+".png", bbox_inches="tight")

