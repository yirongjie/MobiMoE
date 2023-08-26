

def loadData(file_name):
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

    d_ = {}
    for i in filelines:
        sp = i.split(" ")
        # print(f'{sp[0]}_{sp[1]}', float(sp[2]))
        d_[f'{sp[0]}_{sp[1]}'] = float(sp[2])

    a = sorted(d_.items(), key=lambda x: x[1])
    keys=[]
    cals=[]
    for i in a:
        # print(i[0], i[1])
        keys.append(i[0])
        cals.append(i[1])
    # print(len(a))
    return keys,cals

import os 
keys,cals =loadData(os.path.dirname(os.path.realpath(__file__)) +'/st-b8-metrics.csv')
print(keys[:10])
print(keys)